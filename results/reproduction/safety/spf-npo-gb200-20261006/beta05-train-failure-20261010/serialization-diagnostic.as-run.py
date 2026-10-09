from pathlib import Path
import ast,hashlib,json,sys
from types import SimpleNamespace
sys.path.insert(0,'/data/npo-gb200-20261004/validation-20261006/venv/lib/python3.12/site-packages')
from omegaconf import OmegaConf
path=Path('/data/npo-gb200-20261004/validation-20261006/venv/lib/python3.12/site-packages/transformers/training_args.py')
raw=path.read_bytes();sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha(raw)=='67e63d9b8a68a1547c0f3d17eac51034592da669d01cf03b4471b0b77f22a756'
tree=ast.parse(raw)
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TrainingArguments')
post=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__post_init__')
node=next(n for n in post.body if isinstance(n,ast.If) and ast.unparse(n.test)=="self.report_to == 'all' or self.report_to == ['all']")
body=ast.Module(body=[node],type_ignores=[])
value=SimpleNamespace(report_to=OmegaConf.create([]))
exec(compile(body,'<bound_report_to_normalization>','exec'),{'self':value})
actual={'report_to':value.report_to}
baseline_path=Path('/data/spf-npo-20261006-r4/spf-full-s0/resolved_training.json')
baseline_raw=baseline_path.read_bytes();assert sha(baseline_raw)=='d3aeba5342e5102124d7a12096cb4190c56c9617dabb75ed1df5a3b155bc354d'
baseline=json.loads(baseline_raw)['args']['report_to']
serialized=json.loads(json.dumps(actual,default=str))['report_to']
assert actual['report_to']!=baseline and serialized==baseline==['[]']
print(json.dumps(dict(status='SOURCE_BOUND_REPORT_TO_SERIALIZATION_MISMATCH_REPRODUCED',
 original_raw_comparison_matches=False,original_baseline_json_serialization_matches=True,
 actual_outer_type=type(value.report_to).__name__,actual_inner_type=type(value.report_to[0]).__name__,
 saved_baseline_report_to=baseline,matched_serialization=serialized,
 training_args_source_sha256=sha(raw),baseline_resolved_sha256=sha(baseline_raw),
 torch_imported='torch' in sys.modules,CUDA_initialized=False,weights_read=False,
 training_arguments_instance_created=False,optimizer_created=False,training_steps=0,paid_API_calls=0,
 original_failed_GPU_setup_loaded291_weights=True,original_failed_GPU_setup_optimizer_created=True,
 original_failure_no_checkpoint_and_no_training_steps=True)))
