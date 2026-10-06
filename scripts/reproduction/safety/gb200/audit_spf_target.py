"""Independent CPU audit of actual construction exposure and complete 8B HF exports."""
import argparse,hashlib,json,math,pathlib,struct

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(16*1024**2),b''):h.update(b)
    return h.hexdigest()

def weights(path):
    config=json.loads((path/'config.json').read_text())
    assert (config['num_hidden_layers'],config['hidden_size'],config['intermediate_size'],config['vocab_size'])==(32,4096,14336,128256)
    expected={'model.embed_tokens.weight':[128256,4096],'lm_head.weight':[128256,4096],'model.norm.weight':[4096]}
    for i in range(32):
        prefix=f'model.layers.{i}.'
        expected.update({prefix+'input_layernorm.weight':[4096],prefix+'post_attention_layernorm.weight':[4096],
                         prefix+'self_attn.q_proj.weight':[4096,4096],prefix+'self_attn.k_proj.weight':[1024,4096],
                         prefix+'self_attn.v_proj.weight':[1024,4096],prefix+'self_attn.o_proj.weight':[4096,4096],
                         prefix+'mlp.gate_proj.weight':[14336,4096],prefix+'mlp.up_proj.weight':[14336,4096],
                         prefix+'mlp.down_proj.weight':[4096,14336]})
    files=sorted(path.glob('*.safetensors'));assert files
    tensors={};proof=[];mapping={}
    for file in files:
        size=file.stat().st_size
        with file.open('rb') as f:
            raw=f.read(8);assert len(raw)==8
            n=struct.unpack('<Q',raw)[0];assert 0<n<10*1024**2
            header=json.loads(f.read(n))
        spans=[]
        for key,item in header.items():
            if key=='__metadata__':continue
            assert key not in tensors and key in expected,key
            assert item['shape']==expected[key] and item['dtype']=='BF16',(key,item)
            lo,hi=item['data_offsets'];assert hi-lo==math.prod(item['shape'])*2
            tensors[key]=item;mapping[key]=file.name;spans.append((lo,hi))
        position=0
        for lo,hi in sorted(spans):assert lo==position;position=hi
        assert position+n+8==size
        proof.append({'name':file.name,'size':size,'sha256':sha(file),'tensors':len(spans)})
    assert set(tensors)==set(expected) and len(tensors)==291
    index=path/'model.safetensors.index.json'
    if index.exists():assert json.loads(index.read_text())['weight_map']==mapping
    else:assert len(files)==1
    for name in ['config.json','tokenizer.json','tokenizer_config.json','trainer_state.json']:
        assert (path/name).is_file(),name
    return {'parameter_tensors':len(tensors),'layers':32,'files':proof,'weights_bytes':sum(p['size'] for p in proof)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--target',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    target=pathlib.Path(a.target);record=json.loads((target/'target.json').read_text())
    n={'full':4000,'retain95':3800}[record['split']];updates=math.ceil(n/32)*5
    resolved=json.loads((target/'resolved_training.json').read_text());state=json.loads((target/'trainer_state.json').read_text())
    assert record['method']=='spf' and record['construction_seed']==0 and record['mode']=='full'
    assert resolved['world_size']==1 and resolved['effective_batch']==32 and resolved['total_updates']==updates
    assert resolved['args']['num_train_epochs']==5 and resolved['warmup_steps']==math.ceil(n/32)
    assert state['global_step']==updates and state['epoch']==5.0
    assert record['reload']['status']=='pass'
    rows=[json.loads(line) for line in (target/'trajectory.jsonl').read_text().splitlines()]
    assert rows[0]['step']==0;rows=rows[1:];assert len(rows)==updates
    expected_window=[32]*(n//32)+([n%32] if n%32 else [])
    assert [r['window_examples'] for r in rows]==expected_window*5
    examples=tokens=0
    for step,r in enumerate(rows,1):
        assert r['step']==r['engine_global_steps']==r['scheduler_steps']==step
        assert abs(r['task_normalized_weight_sum']-1)<1e-10 and r['backward_scale_wrt_gas'] is False
        assert r['window_assistant_tokens']>0 and r['task_microbatches_per_rank']==math.ceil(r['window_examples']/resolved['args']['per_device_train_batch_size'])
        examples+=r['window_examples'];tokens+=r['window_assistant_tokens']
        assert r['examples']==examples and r['tokens']==tokens
        assert all(math.isfinite(r[k]) for k in ['task_loss','global_dot','gradient_norm_before','gradient_norm_after','optimizer_update_norm'])
    assert examples==n*5 and rows[-1]['unique_examples']==n and rows[-1]['lr_after']==0
    final=weights(target);trajectory=[]
    for step in sorted({math.ceil(updates*f) for f in (.25,.5,.75,1)}):
        checkpoint=target/f'checkpoint-{step}';assert json.loads((checkpoint/'trainer_state.json').read_text())['global_step']==step
        trajectory.append({'step':step,**weights(checkpoint)})
    result={'status':'pass','method':'spf','split':record['split'],'construction_seed':0,'actual_epochs':state['epoch'],
            'actual_updates':updates,'actual_examples':examples,'actual_assistant_tokens':tokens,'unique_examples':n,
            'conflicting_updates':sum(r['conflict'] for r in rows),'safety_tokens_computed':rows[-1]['safety_tokens_computed'],
            'fresh_reload':record['reload'],'final_checkpoint':final,'trajectory_checkpoints':trajectory,
            'manifest_sha256':record['manifest_sha256'],'trajectory_sha256':sha(target/'trajectory.jsonl'),
            'resolved_training_sha256':sha(target/'resolved_training.json')}
    out=pathlib.Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ['status','split','actual_epochs','actual_updates','actual_examples','conflicting_updates']}))

if __name__=='__main__':main()
