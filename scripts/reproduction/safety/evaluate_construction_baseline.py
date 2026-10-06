"""Cache-only matched TOFU baseline; no training, judges, or weight exports.

Run with torchrun --nproc_per_node=2. Rows/generations are private artifacts;
public-summary.json contains only aggregate scores and source hashes.
"""
import argparse
from collections import Counter
from datetime import timedelta
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from construction.baseline import generation_batch, normalize_answer, split_membership
from construction.data import ConstructionDataset, ConstructionCollator
from construction.learning_metrics import teacher_forced_metrics
from construction.protocol import digest_json, read_rows, sha256, validate_manifest, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model-id')
    p.add_argument('--model-revision')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--max-new-tokens', type=int, default=200)
    args = p.parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1:
        raise ValueError('Positive batch size/generation cap required')
    manifest = validate_manifest(json.loads(Path(args.manifest).read_text()))
    if bool(args.model_id) != bool(args.model_revision):
        raise ValueError('Model override requires both id and immutable revision')
    model_id = args.model_id or manifest['model_id']
    revision = args.model_revision or manifest['model_revision']
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('Pin model revision')
    out = Path(args.output)
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    if rank == 0:
        out.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()
    source = manifest['datasets']
    for split in ('full', 'retain95'):
        if sha256(source[split]['path']) != source[split]['sha256']:
            raise ValueError('TOFU export changed')
    rows, retain = read_rows(source['full']['path']), read_rows(source['retain95']['path'])
    membership = split_membership(rows, retain)
    if Counter(membership) != {'forget05': 200, 'retain95': 3800}:
        raise ValueError('Expected complete 4000-row TOFU full with forget05/retain95')
    repo = Path(__file__).resolve().parents[3]
    template = OmegaConf.load(repo/'configs/model/Llama-3.1-8B-Instruct.yaml').template_args
    torch.manual_seed(manifest['construction_seed'])
    tokenizer = AutoTokenizer.from_pretrained(manifest['model_id'], revision=manifest['tokenizer_revision'],
                                              local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = ConstructionDataset(rows, tokenizer, template)
    collator = ConstructionCollator(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, local_files_only=True,
                                               torch_dtype=torch.bfloat16,
                                               attn_implementation='flash_attention_2').to(local).eval()
    base_generation = GenerationConfig.from_pretrained(manifest['model_id'],
                                                       revision=manifest['model_revision'], local_files_only=True)
    generation = GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1,
                                   use_cache=True, eos_token_id=base_generation.eos_token_id,
                                   pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id)
    eos = generation.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    indices = list(range(rank, len(rows), world))
    started = time.perf_counter()
    with (out/f'rows-rank{rank}.jsonl').open('x', encoding='utf-8') as handle, torch.inference_mode():
        for offset in range(0, len(indices), args.batch_size):
            batch_ids = indices[offset:offset+args.batch_size]
            items = [dataset[i] for i in batch_ids]
            batch = {k:v.to(local) for k,v in collator(items).items()}
            logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], use_cache=False).logits
            scores = teacher_forced_metrics(logits, batch['labels'])
            del logits, batch
            prompts = {k:v.to(local) for k,v in generation_batch(items, tokenizer.pad_token_id).items()}
            generated = model.generate(**prompts, generation_config=generation)[:, prompts['input_ids'].shape[1]:].cpu()
            for i, score, tokens in zip(batch_ids, scores, generated.tolist()):
                stop = next((n for n,t in enumerate(tokens) if t in eos), None)
                length = len(tokens) if stop is None else stop+1
                text = tokenizer.decode(tokens[:length], skip_special_tokens=True)
                rouge = scorer.score(rows[i]['answer'], text)['rougeL']
                score.update(rouge_l_recall=rouge.recall, rouge_l_f1=rouge.fmeasure,
                             normalized_exact_match=float(normalize_answer(text)==normalize_answer(rows[i]['answer'])),
                             generation_tokens=length,
                             generation_hit_cap=float(stop is None and length==args.max_new_tokens))
                handle.write(json.dumps(dict(id=str(i), row_sha256=digest_json(rows[i]), split=membership[i],
                                             metrics=score, generation=text), ensure_ascii=False)+'\n')
            handle.flush()
            write_json(out/f'progress-rank{rank}.json', {'completed':min(offset+args.batch_size,len(indices)),
                                                       'total':len(indices),'elapsed_seconds':time.perf_counter()-started})
    write_json(out/f'resources-rank{rank}.json', {'elapsed_seconds':time.perf_counter()-started,
                                                'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated()})
    if world > 1:
        dist.barrier()
    if rank == 0:
        records = [json.loads(line) for r in range(world) for line in (out/f'rows-rank{r}.jsonl').read_text().splitlines()]
        assert sorted(int(row['id']) for row in records)==list(range(len(rows)))
        aggregates = {}
        for split in ('forget05','retain95'):
            selected = [r['metrics'] for r in records if r['split']==split]
            aggregates[split] = {'count':len(selected), 'metrics':{key:{
                'mean':float(np.mean([r[key] for r in selected])),
                'std':float(np.std([r[key] for r in selected], ddof=1)),
            } for key in selected[0]}}
        write_json(out/'public-summary.json', {
            'status':'pass','model_id':model_id,'model_revision':revision,
            'tokenizer_revision':manifest['tokenizer_revision'],'manifest_sha256':sha256(args.manifest),
            'task_data_sha256':source['full']['sha256'], 'retain_data_sha256':source['retain95']['sha256'],
            'template_sha256':sha256(repo/'configs/model/Llama-3.1-8B-Instruct.yaml'),
            'teacher_forcing_max_length':512, 'generation':generation.to_dict(),
            'batch_size_per_rank':args.batch_size,'world_size':world,'aggregates':aggregates,
            'resources':[json.loads((out/f'resources-rank{r}.json').read_text()) for r in range(world)],
            'checkpoint_written':False,'thresholds_frozen':False,
            'limitations':['Descriptive baseline, not a target gate result',
                           'Teacher-forced suffix metric is not free-running extraction',
                           f'Normalized exact match preserves punctuation; generation is capped at {args.max_new_tokens} tokens',
                           'Historical full model is a calibration reference, not the new Standard target'],
        })
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
