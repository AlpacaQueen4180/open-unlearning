"""Cache-only two-turn development generation; invoke with torchrun for sharding."""
import argparse,hashlib,json,os,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig

def main():
    if (int(os.environ.get('WORLD_SIZE', '1')) != 1
            or int(os.environ.get('RANK', '0')) != 0
            or int(os.environ.get('LOCAL_RANK', '0')) != 0):
        raise ValueError('SPF development adapter requires world1/rank0/local0')
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['dataset','dataset-sha256','output','manifest']:
        p.add_argument('--'+key,required=True)
    p.add_argument('--max-new-tokens',type=int,required=True)
    args=p.parse_args()
    if args.max_new_tokens<1:raise ValueError('Positive generation cap required')
    source=Path(args.dataset)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==args.dataset_sha256
    rows=json.loads(source.read_text())
    assert rows and len({str(r['id']) for r in rows})==len(rows)
    assert all(len(r['turns'])==2 and all(isinstance(t,str) and t.strip() for t in r['turns']) for r in rows)
    manifest=json.loads(Path(args.manifest).read_text())
    rank=int(os.environ.get('RANK','0')); world=int(os.environ.get('WORLD_SIZE','1'))
    local=int(os.environ.get('LOCAL_RANK','0'))
    torch.cuda.set_device(local);torch.manual_seed(0)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    # Refuse rank-output reuse before loading weights.
    with (out/f'responses-rank{rank}.jsonl').open('x',encoding='utf-8') as f:
        tokenizer=AutoTokenizer.from_pretrained(manifest['model_id'],revision=manifest['tokenizer_revision'],local_files_only=True)
        assert tokenizer.chat_template
        model=AutoModelForCausalLM.from_pretrained('/data/spf-npo-20261006-r4/spf-full-s0',revision=None,local_files_only=True,
                  torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2').to(local).eval()
        base=GenerationConfig.from_pretrained(manifest['model_id'],revision=manifest['model_revision'],local_files_only=True)
        config=GenerationConfig(do_sample=False,max_new_tokens=args.max_new_tokens,use_cache=True,
                  eos_token_id=base.eos_token_id,pad_token_id=tokenizer.eos_token_id,bos_token_id=tokenizer.bos_token_id)
        eos={config.eos_token_id} if isinstance(config.eos_token_id,int) else set(config.eos_token_id)
        indices=list(range(rank,len(rows),world));started=time.time()
        for done,i in enumerate(indices,1):
            row=rows[i]
            messages=[]
            for turn,prompt in enumerate(row['turns'],1):
                messages.append({'role':'user','content':prompt})
                inputs=tokenizer.apply_chat_template(messages,add_generation_prompt=True,return_tensors='pt',return_dict=True)
                if inputs['input_ids'].shape[1]+args.max_new_tokens>model.config.max_position_embeddings:
                    raise ValueError('Context overflow; no silent truncation')
                inputs={k:v.to(local) for k,v in inputs.items()}
                with torch.inference_mode():
                    tokens=model.generate(use_model_defaults=False, do_sample=False, **inputs,generation_config=config)[0,inputs['input_ids'].shape[1]:].tolist()
                stopped=any(t in eos for t in tokens)
                result=dict(id=str(row['id'])+':'+str(turn),conversation_id=str(row['id']),turn=turn,prompt=prompt,response=tokenizer.decode(tokens,skip_special_tokens=True),
                            generated_tokens=len(tokens),hit_cap=not stopped and len(tokens)==args.max_new_tokens)
                f.write(json.dumps(result,ensure_ascii=False)+'\n');f.flush()
                messages.append({'role':'assistant','content':result['response']})
            (out/f'progress-rank{rank}.json').write_text(json.dumps(dict(completed=done,total=len(indices),elapsed_seconds=time.time()-started)))
        (out/f'metadata-rank{rank}.json').write_text(json.dumps(dict(status='pass',count=len(indices),turns_per_conversation=2,model='/data/spf-npo-20261006-r4/spf-full-s0',checkpoint_identity_sha256='c45e87fc69825b75f5d47345fb3c8502a4c1b4524f9d621411068bf3bc64f3a8',world_size=world,
             revision=None,tokenizer_revision=manifest['tokenizer_revision'],dataset_sha256=args.dataset_sha256,
             template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),system_message=None,
             generation=config.to_dict(),seed=0,elapsed_seconds=time.time()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated()),indent=2))

if __name__=='__main__':main()
