import hashlib,importlib.machinery,json,os,sys,types
from pathlib import Path
sys.dont_write_bytecode=True
assert sys.executable=='/data/npo-gb200-20261004/validation-20261006/venv/bin/python' and os.environ['CUDA_VISIBLE_DEVICES']==''
V=Path('/data/spf-hybrid01-rank40-full-20261012-r1/validation-code');BASE=Path('/data/spf-npo-20261006-r4');sha=lambda raw:hashlib.sha256(raw).hexdigest()
original=(V/'verify_spf_anchor_hybrid_rank40.py').read_bytes();assert sha(original)=='548814b7d849fc9e58f40441f176ad463f1aa71854c235fed1daf998d5884be1'
text=original.decode()
assert text.count('cross_entropy(logits.reshape(-1, 64), targets.reshape(-1))')==1;text=text.replace('cross_entropy(logits.reshape(-1, 64), targets.reshape(-1))','cross_entropy(logits.double().reshape(-1, 64), targets.reshape(-1))')
assert text.count('spf_projection_rank=40, safety_loss_coefficient=.1, actual_cpu_updates=2')==1;text=text.replace('spf_projection_rank=40, safety_loss_coefficient=.1, actual_cpu_updates=2',"loss_oracle_dtype='torch.float64',parameter_gradient_projection_dtype='torch.float32',original_tolerances_unchanged=True,spf_projection_rank=40, safety_loss_coefficient=.1, actual_cpu_updates=2")
assert sha(text.encode())=='a998a42a44109da78c1789478881f1da8fc5753b53788844141b4bc5e65b447b'
manifest=json.loads((BASE/'frozen.json').read_bytes());pins={}
for name in ('src/construction/backend.py','src/construction/projection.py','src/construction/numerics.py'):
    pins[name]=sha((BASE/'repo'/name).read_bytes());assert pins[name]==manifest['construction_code'][name]
sys.path.insert(0,str(BASE/'repo/src'))
import torch
assert not torch.cuda.is_initialized()
m=types.ModuleType('rank40_new_scalar_oracle');m.__file__=str(V/'verify_spf_anchor_hybrid_rank40_cpu_oracle_r2.py');m.__spec__=importlib.machinery.ModuleSpec(m.__name__,loader=None);sys.modules[m.__name__]=m
exec(compile(text,'<new_rank40_cpu_oracle_r2>','exec'),m.__dict__)
helper=m.load_helper(V/'spf_anchor_hybrid.py','fdb2e1254dcae25c33ee788123eb2a0ca29e2d23edae0db7971c72d8c15b45cc')
result=m.cpu_check(helper);assert result['actual_cpu_updates']==2 and result['spf_projection_rank']==40 and not torch.cuda.is_initialized()
result.update(verifier_sha256=sha(text.encode()),helper_sha256='fdb2e1254dcae25c33ee788123eb2a0ca29e2d23edae0db7971c72d8c15b45cc',
 frozen_dependency_sha256=pins,torch_version=torch.__version__,execution_mode='in_memory_exact_source_bound_cpu_function_only',verifier_file_deployed=False,
 cuda_initialized=False,hf_weights_read=False,actual8b_training=0,api_calls=0,credential_reads=0,bytecode_writes_disabled=True)
print(json.dumps(result))
