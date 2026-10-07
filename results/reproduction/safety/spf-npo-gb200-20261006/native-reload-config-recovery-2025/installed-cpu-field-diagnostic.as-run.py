from pathlib import Path
import ast, copy, hashlib, json, sys
from types import SimpleNamespace
import pydantic
from pydantic import BaseModel, Field, ValidationError

site=Path('/data/npo-gb200-20261004/validation-20261006/venv/lib/python3.12/site-packages')
paths={'hf':site/'transformers/integrations/deepspeed.py','zero':site/'deepspeed/runtime/zero/config.py',
       'utils':site/'deepspeed/runtime/config_utils.py','base':Path('/data/spf-development-20261007-r1/repo/src/trainer/unlearn/base.py')}
expected=dict(hf='194c012cde03afdf8acacf0b94acc3e4e475f08cdaf9fa1f5be0ef6ac5a16bc3',
 zero='1c5e4e47b9d5adec5051e94aba1512f56a6835773ca771611eadc707f40cea33',
 utils='9297e58edfa7ffa7b419b207fdc4fa62667f5aad0ed7488da303bcb82d50cdb1',
 base='fbd938156f4bd86a33582d18e2363de0fb53a72788904a6c6a4bc803486a07a6')
trees={}
for k,path in paths.items():
    raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==expected[k];trees[k]=ast.parse(raw)
hf=next(n for n in ast.walk(trees['hf']) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='fill_only' and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value=='zero_optimization.stage3_prefetch_bucket_size')
value=eval(compile(ast.Expression(hf.args[1]),'<installed-HF-integer-expression>','eval'),{'hidden_size':4096})
assert type(value) is int and value==15099494
z=next(n for n in trees['zero'].body if isinstance(n,ast.ClassDef) and n.name=='DeepSpeedZeroConfig')
field=next(n for n in z.body if isinstance(n,ast.AnnAssign) and n.target.id=='prefetch_bucket_size')
cls=ast.ClassDef(name='IsolatedInstalledPrefetchField',bases=[ast.Name(id='BaseModel',ctx=ast.Load())],keywords=[],body=[field],decorator_list=[])
ns=dict(BaseModel=BaseModel,Field=Field,pp_int=int)
exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),'<isolated-installed-field>','exec'),ns)
model=ns['IsolatedInstalledPrefetchField']
try:model(stage3_prefetch_bucket_size=.9*4096*4096)
except ValidationError as e:
    errors=[dict(type=x['type'],loc=list(x['loc'])) for x in e.errors()];assert errors==[dict(type='int_from_float',loc=['stage3_prefetch_bucket_size'])]
else:raise AssertionError('Fractional value unexpectedly accepted')
assert model(stage3_prefetch_bucket_size=value).prefetch_bucket_size==value
base=next(n for n in trees['base'].body if isinstance(n,ast.ClassDef) and n.name=='UnlearnTrainer')
prepare=next(n for n in base.body if isinstance(n,ast.FunctionDef) and n.name=='_prepare_deepspeed')
seen=[]
fake=SimpleNamespace(initialize=lambda **kw:(seen.append(copy.deepcopy(kw['config'])) or SimpleNamespace(eval=lambda:None),))
bn=dict(deepcopy=copy.deepcopy,deepspeed=fake)
exec(compile(ast.Module(body=[prepare],type_ignores=[]),'<native-prepare-with-initialize-stub>','exec'),bn)
cfg=dict(zero_optimization=dict(stage=3,reduce_bucket_size=4096**2,stage3_prefetch_bucket_size=value,stage3_param_persistence_threshold=40960))
host=SimpleNamespace(accelerator=SimpleNamespace(state=SimpleNamespace(deepspeed_plugin=SimpleNamespace(deepspeed_config=cfg))))
bn['_prepare_deepspeed'](host,SimpleNamespace(config=SimpleNamespace(hidden_size=4096)))
assert seen[0]['zero_optimization']==cfg['zero_optimization']
assert seen[0]['zero_optimization.stage3_prefetch_bucket_size']==15099494.4
assert model(**seen[0]['zero_optimization']).prefetch_bucket_size==value
assert 'torch' not in sys.modules and 'deepspeed' not in sys.modules
print(json.dumps(dict(status='ACTUAL_INSTALLED_PYDANTIC_CPU_FIELD_AND_SOURCE_AST_PASS_ONLY',
 pydantic_version=pydantic.__version__,source_sha256=expected,corrected_prefetch=value,
 fractional_error=errors,native_prepare_nested_integer_preserved=True,native_prepare_dotted_float_retained=True,
 original_reference_defaults_differ_from_resolved_target=True,full_deepspeed_initialization_tested=False,
 torch_imports=0,cuda_initialized=False,weights_read=False,real_initialize_calls=0,model_reload_executed=False)))
