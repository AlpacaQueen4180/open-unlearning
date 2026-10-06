"""Cache-only MMLU auxiliary zero-shot option likelihood, not formal MMLU5shot."""
import argparse,hashlib,json,os,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from construction.knowledge import encode_choices,continuation_score,choose_option

def main():
    if (int(os.environ.get('WORLD_SIZE', '1')) != 1
            or int(os.environ.get('RANK', '0')) != 0
            or int(os.environ.get('LOCAL_RANK', '0')) != 0):
        raise ValueError('SPF development adapter requires world1/rank0/local0')
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['dataset','dataset-sha256','manifest','output']:p.add_argument('--'+name,required=True)
    a=p.parse_args();path=Path(a.dataset)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==a.dataset_sha256
    rows=json.loads(path.read_text());assert len(rows)==1024 and len({r['id'] for r in rows})==1024
    m=json.loads(Path(a.manifest).read_text());torch.cuda.set_device(0)
    out=Path(a.output);out.mkdir(exist_ok=False)
    tok=AutoTokenizer.from_pretrained(m['model_id'],revision=m['tokenizer_revision'],local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained('/data/spf-npo-20261006-r4/spf-full-s0',revision=None,local_files_only=True,
                    torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2').to('cuda').eval()
    correct=0;ties=0;start=time.time()
    with (out/'rows.jsonl').open('x') as f,torch.inference_mode():
        for n,row in enumerate(rows,1):
            scores=[]
            for ids,boundary in encode_choices(tok,row):
                if len(ids)>model.config.max_position_embeddings:raise ValueError('Context overflow')
                inputs=torch.tensor([ids],device='cuda')
                logits=model(inputs,use_cache=False).logits[0,boundary-1:-1].float()
                targets=inputs[0,boundary:]
                score=continuation_score(logits,targets)
                scores.append(score)
            prediction=choose_option(scores);correct+=int(prediction==row['answer'])
            ties+=int(scores.count(max(scores))>1)
            f.write(json.dumps(dict(id=row['id'],scores=scores,prediction=prediction,answer=row['answer']))+'\n');f.flush()
            (out/'progress.json').write_text(json.dumps(dict(completed=n,total=len(rows),elapsed_seconds=time.time()-start)))
    result=dict(status='pass',count=len(rows),correct=correct,accuracy=correct/len(rows),ties=ties,tie_policy='first option',
        protocol='auxiliary-zero-shot-ABCD-summed-continuation-logprob-v1',num_fewshot=0,chat_template=False,
        model='/data/spf-npo-20261006-r4/spf-full-s0',revision=None,checkpoint_identity_sha256='c45e87fc69825b75f5d47345fb3c8502a4c1b4524f9d621411068bf3bc64f3a8',world_size=1,tokenizer_revision=m['tokenizer_revision'],dataset_sha256=a.dataset_sha256,
        elapsed_seconds=time.time()-start,research_gate='insufficient_evidence',
        limitations=['Auxiliary subject labels are empty; no subject-specific5shot prompts','Not comparable to historical formal MMLU5shot scores'])
    (out/'public-summary.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
