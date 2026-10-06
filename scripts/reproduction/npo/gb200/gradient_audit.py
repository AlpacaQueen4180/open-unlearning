"""Controlled pre-clip gradient audit of actual NPO -> Trainer -> Accelerate -> DS.

Synthetic fixed tokens avoid dataset/ordering confounds. A tail boundary is
explicitly flushed on the legacy stack to isolate scaling from the known step
bug; this is NOT a measurement of an unmodified legacy epoch loop.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

p=argparse.ArgumentParser()
p.add_argument('--repo',required=True)
p.add_argument('--output',required=True)
p.add_argument('--gas',type=int,default=8)
p.add_argument('--window',type=int,required=True)
p.add_argument('--term',choices=['forget','retain','combined'],required=True)
p.add_argument('--dtype',choices=['fp32','bf16'],default='fp32')
p.add_argument('--unequal-tokens',action='store_true')
p.add_argument('--model-path')
p.add_argument('--batch-file')
p.add_argument('--offset',type=int,default=0)
p.add_argument('--ds-reference',action='store_true',help='Explicit mean control through the SAME DS engine; retains autograd comparisons')
a=p.parse_args()
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(Path(a.repo)/'src'))
import torch
import torch.nn.functional as F
import deepspeed
import transformers
import accelerate
from transformers import LlamaConfig,LlamaForCausalLM,TrainingArguments
from accelerate.utils import DistributedType
from accelerate.utils.deepspeed import DeepSpeedEngineWrapper,DeepSpeedOptimizerWrapper
from deepspeed.utils import safe_get_full_grad
from trainer.unlearn.npo import NPO

torch.manual_seed(4180)
torch.backends.cuda.matmul.allow_tf32=False
dtype=torch.float32 if a.dtype=='fp32' else torch.bfloat16
if a.model_path:
    model=LlamaForCausalLM.from_pretrained(a.model_path,torch_dtype=dtype,attn_implementation='eager').cuda()
else:
    model=LlamaForCausalLM(LlamaConfig(vocab_size=71,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,attention_dropout=0.0,tie_word_embeddings=False)).to('cuda',dtype=dtype)
for m in model.modules():
    if isinstance(m,torch.nn.Dropout):m.p=0.0
model.config.use_cache=False
alpha=0.0 if a.term=='forget' else 1.0
gamma=0.0 if a.term=='retain' else 1.0
args=TrainingArguments(output_dir=str(out.with_suffix(''))+'-trainer',per_device_train_batch_size=4,gradient_accumulation_steps=a.gas,report_to=[],remove_unused_columns=False,bf16=False,max_grad_norm=0.0)
trainer=NPO(model=model,args=args,alpha=alpha,gamma=gamma,beta=0.1)
trainer.ref_model.eval()
rows=4*a.window
vocab=model.config.vocab_size
def inputs(offset):
    ids=(torch.arange(rows*16,device='cuda').reshape(rows,16)+offset)%(vocab-4)+3
    labels=ids.clone();labels[:,:4]=-100
    if a.unequal_tokens:
        for j in range(rows):labels[j,:4+(j//4)%8]=-100
    return dict(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labels)
batch={'forget':inputs(1),'retain':inputs(9)}
if a.batch_file:
    frozen=torch.load(a.batch_file,map_location='cpu',weights_only=True)
    batch={k:{q:v[a.offset:a.offset+rows].cuda() for q,v in x.items() if q in ('input_ids','labels','attention_mask')} for k,x in frozen.items()}
    assert all(x['input_ids'].shape[0]==rows for x in batch.values())
parts=[{k:{q:v[i*4:(i+1)*4] for q,v in x.items()} for k,x in batch.items()} for i in range(a.window)]
names=[n for n,x in model.named_parameters() if x.requires_grad]

def parameter_hash(sharded=False):
    digest=hashlib.sha256()
    for n,x in model.named_parameters():
        if not x.requires_grad:continue
        context=deepspeed.zero.GatheredParameters([x]) if sharded else __import__('contextlib').nullcontext()
        with context:
            digest.update(n.encode())
            flat=x.detach().reshape(-1)
            for start in range(0,flat.numel(),1000000):
                digest.update(flat[start:start+1000000].float().cpu().numpy().tobytes())
    return digest.hexdigest()

initial_parameter_hash=parameter_hash() if a.ds_reference else None
def gradients():
    return torch.cat([x.grad.detach().float().cpu().reshape(-1) if x.grad is not None else torch.zeros(x.numel()) for x in model.parameters() if x.requires_grad])
model.zero_grad(set_to_none=True)
# Reference objective A: arithmetic mean of the SAME microbatch losses.
for b in parts:(trainer.compute_loss(model,b)/a.window).backward()
mean_grad=gradients();model.zero_grad(set_to_none=True)
# Reference objective B: single large batch (token-weighting can differ).
trainer.compute_loss(model,batch).backward()
large_grad=gradients();model.zero_grad(set_to_none=True)
world=int(os.environ.get('WORLD_SIZE','1'))
config={'train_batch_size':4*a.gas*world,'train_micro_batch_size_per_gpu':4,'gradient_accumulation_steps':a.gas,'gradient_clipping':0.0,'zero_allow_untested_optimizer':True,'zero_optimization':{'stage':3,'overlap_comm':False,'contiguous_gradients':True,'reduce_bucket_size':1000000,'stage3_prefetch_bucket_size':1000000,'stage3_param_persistence_threshold':100000},'bf16':{'enabled':a.dtype=='bf16'},'fp16':{'enabled':False},'steps_per_print':1000000}
optimizer=torch.optim.SGD(model.parameters(),lr=0.0)
engine,_,_,_=deepspeed.initialize(model=model,optimizer=optimizer,config=config)
trainer.model_wrapped=engine
trainer.is_deepspeed_enabled=True
trainer.optimizer=DeepSpeedOptimizerWrapper(engine.optimizer)
trainer.accelerator.state.distributed_type=DistributedType.DEEPSPEED
trainer.accelerator.deepspeed_engine_wrapped=DeepSpeedEngineWrapper(engine)
trainer.current_gradient_accumulation_steps=a.window
trainer.current_flos=0
traces=[];captured=[]
original_backward=engine.backward
def backward(loss,*args,**kwargs):
    traces.append({'loss_to_engine':float(loss.detach()),'kwargs':kwargs,'boundary':engine.is_gradient_accumulation_boundary()})
    return original_backward(loss,*args,**kwargs)
engine.backward=backward
original_step=engine.step
def step(*args,**kwargs):
    if engine.is_gradient_accumulation_boundary():
        full=[]
        for n,x in engine.module.named_parameters():
            if x.requires_grad:
                g=safe_get_full_grad(x)
                assert g is not None,n
                full.append(g.detach().float().cpu().reshape(-1))
        captured.append(torch.cat(full))
    return original_step(*args,**kwargs)
engine.step=step
for i,b in enumerate(parts):
    sync=i==a.window-1
    trainer.accelerator.gradient_state._set_sync_gradients(sync)
    if transformers.__version__.startswith('4.'):
        engine.set_gradient_accumulation_boundary(sync)
    trainer.training_step(engine,b,num_items_in_batch=None)
assert len(captured)==1,len(captured)
observed=captured[0]
observed_updates=engine.global_steps
observed_traces=list(traces)
ds_mean_grad=None
if a.ds_reference:
    # Same kernels, buffers, parameters and NPO loss; explicit K division.
    # This isolates Trainer scaling from the original autograd/DS discrepancy.
    for i,b in enumerate(parts):
        sync=i==a.window-1
        engine.set_gradient_accumulation_boundary(sync)
        engine.backward(trainer.compute_loss(engine,b)/a.window,scale_wrt_gas=False)
        if sync:engine.step()
    assert len(captured)==2,len(captured)
    ds_mean_grad=captured[1]
    assert engine.global_steps-observed_updates==1
    final_parameter_hash=parameter_hash(sharded=True)
    assert initial_parameter_hash==final_parameter_hash,'lr0 control changed parameters'
def comparison(x,y):
    import math
    xx=yy=xy=0.0
    for start in range(0,x.numel(),1000000):
        u=x[start:start+1000000].double();v=y[start:start+1000000].double()
        xx+=float(u.dot(u));yy+=float(v.dot(v));xy+=float(u.dot(v))
    scale=xy/yy
    err=scaled=0.0
    for start in range(0,x.numel(),1000000):
        u=x[start:start+1000000].double();v=y[start:start+1000000].double()
        err+=float((u-v).square().sum());scaled+=float((u-scale*v).square().sum())
    return {'norm_ratio':math.sqrt(xx/yy),'projected_scale':scale,'cosine':xy/math.sqrt(xx*yy),'relative_error':math.sqrt(err/yy),'relative_error_after_scale':math.sqrt(scaled/yy),'reference_norm':math.sqrt(yy),'observed_norm':math.sqrt(xx)}
report={'versions':{x:__import__(x).__version__ for x in ['torch','transformers','accelerate','deepspeed']},'gpu':torch.cuda.get_device_name(),'world_size':1,'gas':a.gas,'window':a.window,'term':a.term,'dtype':a.dtype,'unequal_tokens':a.unequal_tokens,'model':a.model_path or 'fixed tiny Llama (2 layers, hidden32)','model_accepts_loss_kwargs':trainer.model_accepts_loss_kwargs,'num_items_in_batch':None,'accelerator_gas':trainer.accelerator.gradient_accumulation_steps,'engine_gas':engine.gradient_accumulation_steps(),'engine_override_scale_wrt_gas':getattr(engine,'_scale_wrt_gas',None),'legacy_tail_boundary_forced':transformers.__version__.startswith('4.') and a.window!=a.gas,'clipping':0,'optimizer_lr':0,'engine_updates':engine.global_steps,'engine_backward_trace':traces,'vs_mean_microbatch_objective':comparison(observed,mean_grad),'vs_large_batch_objective':comparison(observed,large_grad),'mean_micro_vs_large':comparison(mean_grad,large_grad),'gradient_sha256':hashlib.sha256(observed.numpy().tobytes()).hexdigest(),'parameter_names':names,'synthetic_input_sha256':hashlib.sha256(batch['forget']['input_ids'].cpu().numpy().tobytes()+batch['retain']['labels'].cpu().numpy().tobytes()).hexdigest()}
report['world_size']=world
report['engine_updates']=observed_updates
report['engine_backward_trace']=observed_traces
report['measurement_revision']='ds_explicit_mean_reference_v2' if a.ds_reference else 'autograd_reference_v1'
if a.ds_reference:
    report.update(vs_ds_explicit_mean_reference=comparison(observed,ds_mean_grad),ds_reference_vs_autograd_mean=comparison(ds_mean_grad,mean_grad),ds_reference_backward_trace=traces[len(observed_traces):],ds_reference_updates=engine.global_steps-observed_updates,engine_total_updates=engine.global_steps,initial_parameter_sha256=initial_parameter_hash,final_parameter_sha256=final_parameter_hash)
report['frozen_batch_file']=a.batch_file
report['batch_offset']=a.offset
report['rank']=int(os.environ.get('RANK','0'))
report['distributed_input_policy']='identical fixed inputs on each rank; DS averaging compared with identical local reference'
if world>1:out=out.with_name(out.stem+f'_rank{report["rank"]}.json')
if not a.model_path:
    torch.save({'observed':observed,'mean_micro':mean_grad,'large_batch':large_grad},out.with_suffix('.pt'))
out.write_text(json.dumps(report,indent=2,default=str))
expected=a.window if transformers.__version__.startswith('4.') else 1
r=report['vs_mean_microbatch_objective']
tolerance=.05 if a.dtype=='bf16' else .002
report['autograd_reference_gate_passed']=abs(r['projected_scale']/expected-1)<tolerance and r['relative_error_after_scale']/expected<tolerance
report['autograd_micro_vs_large_gate_passed']=report['mean_micro_vs_large']['relative_error']<tolerance
out.write_text(json.dumps(report,indent=2,default=str))
if a.ds_reference:r=report['vs_ds_explicit_mean_reference']
assert abs(r['projected_scale']/expected-1)<tolerance and r['relative_error_after_scale']/expected<tolerance,r
if not a.unequal_tokens and not a.batch_file and not a.ds_reference:
    assert report['mean_micro_vs_large']['relative_error']<tolerance,report
print(json.dumps(report,default=str))
