from pathlib import Path
import json, subprocess, os

code = r'''
from pathlib import Path
from argparse import ArgumentParser, REMAINDER
from contextlib import redirect_stderr
import argparse, ast, hashlib, importlib.util, inspect, io, json, sys
sha=lambda b:hashlib.sha256(b).hexdigest()
assert 'torch' not in sys.modules
spec=importlib.util.find_spec('torch')
base=Path(list(spec.submodule_search_locations)[0])/'distributed'
run_raw=(base/'run.py').read_bytes();util_raw=(base/'argparse_util.py').read_bytes()
assert sha(run_raw)=='0dce0257f4c6fe4809b95981f98add58dd3ee6a888847385329f0fff977e0699'
assert sha(util_raw)=='345c63c366ac62dd3a6857df72eeab849a42dc516d87028bbbdc2f5849bfa905'
ns={'ArgumentParser':ArgumentParser,'REMAINDER':REMAINDER}
exec(compile(util_raw,'bound-argparse-util','exec'),ns)
node=next(n for n in ast.parse(run_raw).body if isinstance(n,ast.FunctionDef) and n.name=='get_args_parser')
exec(compile(ast.Module(body=[node],type_ignores=[]),'bound-get-args-parser','exec'),ns)
verifier='/data/spf-npo-smoke-20261007-r1/validation-code/verify_spf_npo_smoke_native_reload.py'
original=['--standalone','--nproc_per_node=1',verifier,'--source-sha256','5364672a92d23fdc13b25809ae2364973ab8d81195729880988732b06ca2d2e0','--run','/data/spf-npo-smoke-20261007-r1/runs/reload-native-1312']
err=io.StringIO()
try:
    with redirect_stderr(err): ns['get_args_parser']().parse_args(original)
except SystemExit as error:
    assert error.code==2 and 'ambiguous option: --run' in err.getvalue()
else: raise AssertionError('Remote original parser failure did not reproduce')
repaired=original[:2]+['--']+original[2:]
a=ns['get_args_parser']().parse_args(repaired)
assert a.training_script==verifier and a.training_script_args==original[3:]
assert a.standalone and a.nproc_per_node=='1' and not a.run_path
assert 'torch' not in sys.modules
print(json.dumps(dict(status='ACTUAL_REMOTE_SOURCE_BOUND_CPU_ARGPARSE_REPRODUCTION_AND_SEPARATOR_PASS_ONLY',
    python=sys.version, interpreter=sys.executable, argparse_source_sha256=sha(Path(inspect.getfile(argparse)).read_bytes()),
    torchrun_source_sha256=sha(run_raw),argparse_util_source_sha256=sha(util_raw),
    original_arguments=original,original_parser_exit_code=2,original_parser_stderr_text=err.getvalue(),
    proposed_arguments=repaired,parsed_script=a.training_script,parsed_script_arguments=a.training_script_args,
    separator_only=True,script_arguments_unchanged=True,torch_imported=False,workers_started=0,
    cuda_used=False,weights_read=False,model_reload_executed=False,new_api_calls=0,
    deployed=False,actual_remote_model_reload_pass=False)))
'''
env=dict(os.environ, CUDA_VISIBLE_DEVICES='')
for key in list(env):
    if key.startswith('PET_'): del env[key]
r=subprocess.run(['/data/npo-gb200-20261004/validation-20261006/venv/bin/python','-c',code],
                 capture_output=True,text=True,env=env)
assert r.returncode==0, r.stderr
result=json.loads(r.stdout)
assert not r.stderr
print(json.dumps(dict(status='READONLY_CPU_PARSER_DIAGNOSTIC',result=result,
    child_returncode=r.returncode,stderr_text=r.stderr,progress_snapshot=False,
    remote_mutations=0,gpu_submissions=0,credential_reads=0)))
