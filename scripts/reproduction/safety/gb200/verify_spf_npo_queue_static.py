"""Synthetic queue precondition and journal checks; never launch a subprocess."""
import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import tempfile

import run_spf_npo_smoke_queue as queue


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',required=True);p.add_argument('--adapter-audit',required=True)
    p.add_argument('--private-output',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.package)
    c=json.loads((root/'contract.private.json').read_bytes())
    package=json.loads((root/'package.private.json').read_bytes())
    commands=json.loads((root/'commands.private.json').read_bytes())
    expected=queue.exact_commands(c,package['contract_sha256'])
    assert all(commands[k]==value for k,value in expected.items())
    assert '--resume_from_checkpoint' not in expected['train']
    assert expected['train'][-4:]==['--config-path',c['config_dir'],'--config-name','spf_npo_smoke']
    hydra=queue.hydra_command(c,Path('/synthetic/hydra-compose.private.json'))
    compile(hydra[2],'<sealed-hydra-compose>','exec')
    assert 'from hydra import compose' in hydra[2] and 'model = OmegaConf.load' in hydra[2]
    assert 'torch' not in hydra[2] and 'from data' not in hydra[2]
    now=datetime(2026,10,7,0,0,tzinfo=timezone.utc)
    names=['spf-retain95-s0-'+suffix for suffix in ('train','reload','audit')]
    audit=dict(status='pass',method='spf',split='retain95',construction_seed=0,
        actual_epochs=5.0,actual_updates=595,actual_examples=19000,unique_examples=3800,
        manifest_sha256=queue.FREEZE_SHA,fresh_reload=dict(status='pass'),
        final_checkpoint=dict(parameter_tensors=291,layers=32,weights_bytes=16060556616,
                              files=[dict(size=16060556616,tensors=291)]),
        trajectory_checkpoints=[dict(step=step) for step in (149,298,447,595)])
    status=dict(status='CONSTRUCTION_DONE_PENDING_GATE_NPO',pid=45219,completed=names,
                stages=[dict(name=name,returncode=0,finished_at=1) for name in names])
    snapshot=dict(snapshot_utc=now.isoformat(),launch=dict(pid=45219),status=status,
        reference_audit=audit,process=dict(returncode=1,stdout='    PID STAT COMMAND\n',stderr=''),
        compute_processes=dict(returncode=0,stdout='',stderr=''),
        gpu=dict(returncode=0,stdout='0 MiB, 0 %\n',stderr=''),
        filesystem=dict(returncode=0,stdout='synthetic',stderr=''))
    queue.validate_snapshot(snapshot,now)
    ref_cases=[('reference_running',lambda s,r:s.update(status='RUNNING')),
        ('different_original_pid',lambda s,r:s.update(pid=1)),
        ('reference_stage_missing',lambda s,r:s['completed'].pop()),
        ('reference_stage_failed',lambda s,r:s['stages'][0].update(returncode=9)),
        ('reference_stage_unfinished',lambda s,r:s['stages'][1].pop('finished_at')),
        ('epochs_incomplete',lambda s,r:r.update(actual_epochs=4)),
        ('updates_incomplete',lambda s,r:r.update(actual_updates=594)),
        ('exposure_incomplete',lambda s,r:r.update(actual_examples=18976)),
        ('wrong_split',lambda s,r:r.update(split='full')),
        ('wrong_seed',lambda s,r:r.update(construction_seed=1)),
        ('wrong_freeze',lambda s,r:r.update(manifest_sha256='0'*64)),
        ('reload_failed',lambda s,r:r['fresh_reload'].update(status='fail')),
        ('partial_tensors',lambda s,r:r['final_checkpoint'].update(parameter_tensors=290)),
        ('partial_layers',lambda s,r:r['final_checkpoint'].update(layers=31)),
        ('wrong_shard_bytes',lambda s,r:r['final_checkpoint']['files'][0].update(size=1)),
        ('partial_trajectory',lambda s,r:r['trajectory_checkpoints'].pop())]
    rejected=[]
    for name,mutate in ref_cases:
        s,r=deepcopy(status),deepcopy(audit);mutate(s,r)
        try:queue.completed_reference(s,r)
        except ValueError:rejected.append(name)
        else:raise AssertionError('Accepted synthetic invalid reference: '+name)
    snapshot_cases=[('stale_snapshot',lambda s:s.update(snapshot_utc=(now-timedelta(seconds=601)).isoformat())),
        ('future_snapshot',lambda s:s.update(snapshot_utc=(now+timedelta(seconds=1)).isoformat())),
        ('naive_snapshot',lambda s:s.update(snapshot_utc='2026-10-07T00:00:00')),
        ('absent_audit',lambda s:s.update(reference_audit=None)),
        ('wrong_launch_pid',lambda s:s['launch'].update(pid=1)),
        ('original_queue_live',lambda s:s['process'].update(returncode=0,stdout='PID STAT COMMAND\n45219 Ss python\n')),
        ('ambiguous_ps',lambda s:s['process'].update(returncode=0)),
        ('gpu_process_alive',lambda s:s['compute_processes'].update(stdout='46358, 80000 MiB\n')),
        ('gpu_query_failed',lambda s:s['gpu'].update(returncode=1)),
        ('filesystem_query_failed',lambda s:s['filesystem'].update(returncode=1))]
    snapshot_rejected=[]
    for name,mutate in snapshot_cases:
        s=deepcopy(snapshot);mutate(s)
        try:queue.validate_snapshot(s,now)
        except ValueError:snapshot_rejected.append(name)
        else:raise AssertionError('Accepted synthetic invalid snapshot: '+name)
    zombie=deepcopy(snapshot)
    zombie['process'].update(returncode=0,stdout='PID STAT COMMAND\n45219 Zs [python] <defunct>\n')
    queue.validate_snapshot(zombie,now)
    adapter=queue.bound_json(a.adapter_audit,queue.ADAPTER_SHA)
    acceptance=dict(status='DEVELOPMENT_CUDA_EXECUTORS_PASS',checkpoint_identity_sha256=queue.IDENTITY,
        checkpoint_audit_sha256=queue.FULL_AUDIT_SHA,adapter_audit_sha256=queue.ADAPTER_SHA,
        world_size=1,generation_caps=adapter['generation_caps'],
        base_tokenizer_revision=adapter['base_tokenizer_revision'],external_judge_executed=False,
        executors={name:dict(returncode=0,gpu_runtime_validated=True,process_pid=1,examples=1,
            source_sha256=adapter['sources'][name]['adapter_sha256'],raw_log_sha256='a'*64)
            for name in queue.EXECUTORS})
    queue.development_accepted(acceptance,adapter)
    dev_cases=[('static_only',lambda x:x.update(status='prepared_not_evaluated')),
        ('wrong_identity',lambda x:x.update(checkpoint_identity_sha256='0'*64)),
        ('wrong_adapter',lambda x:x.update(adapter_audit_sha256='0'*64)),
        ('changed_caps',lambda x:x.update(generation_caps={})),
        ('changed_tokenizer',lambda x:x.update(base_tokenizer_revision='main')),
        ('missing_executor',lambda x:x['executors'].pop(queue.EXECUTORS[0])),
        ('no_cuda',lambda x:x['executors'][queue.EXECUTORS[0]].update(gpu_runtime_validated=False)),
        ('executor_failed',lambda x:x['executors'][queue.EXECUTORS[0]].update(returncode=9)),
        ('missing_actual_log',lambda x:x['executors'][queue.EXECUTORS[0]].update(raw_log_sha256=''))]
    dev_rejected=[]
    for name,mutate in dev_cases:
        x=deepcopy(acceptance);mutate(x)
        try:queue.development_accepted(x,adapter)
        except ValueError:dev_rejected.append(name)
        else:raise AssertionError('Accepted synthetic missing CUDA acceptance: '+name)
    private=Path(a.private_output)
    private.mkdir(parents=True,exist_ok=False)
    original_popen,original_command=queue.subprocess.Popen,queue.command_result
    simulated=[]
    try:
        queue.command_result=lambda command:dict(returncode=0,stdout='',stderr='')
        for failed in (None,'train','fresh-reload'):
            run=private/('pass' if failed is None else 'failed-'+failed);run.mkdir()
            state=dict(completed=[],stages=[]);calls=[]
            class StubProcess:
                pid=123
                def __init__(self,command,**kwargs):
                    calls.append(command[0]);kwargs['stdout'].write(b'new synthetic queue journal fixture\n')
                    self.code=9 if command[0]==failed else 0
                def wait(self):return self.code
            queue.subprocess.Popen=StubProcess
            for name in ('hydra-compose','train','fresh-reload'):
                try:queue.stage(state,run/'status.json',name,[name],run,
                    dict(overrides={},values={}),require_idle=name!='hydra-compose')
                except RuntimeError:break
            expected_calls=(['hydra-compose','train'] if failed=='train' else ['hydra-compose','train','fresh-reload'])
            assert calls==expected_calls
            assert state['completed']==([x for x in expected_calls if x!=failed])
            assert all('raw_log_sha256' in row for row in state['stages'])
            simulated.append(dict(failure=failed,calls=calls,completed=state['completed']))
    finally:
        queue.subprocess.Popen,queue.command_result=original_popen,original_command
    source=Path(__file__).resolve().parent
    names=('run_spf_npo_smoke_queue.py','verify_spf_npo_queue_static.py')
    for name in names:compile((source/name).read_bytes(),name,'exec')
    result=dict(status='PASS_SYNTHETIC_QUEUE_PRECONDITIONS_AND_STUB_JOURNALS_ONLY',
        reference_rejections=rejected,snapshot_rejections=snapshot_rejected,
        development_acceptance_rejections=dev_rejected,stub_stage_sequences=simulated,
        sealed_package_sha256=hashlib.sha256((root/'package.private.json').read_bytes()).hexdigest(),
        source_sha256={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in names},
        subprocesses_launched=0,remote_deployment_executed=False,hydra_runtime_composition_validated=False,
        gpu_runtime_validation='not_run',npo_training_executed=False,
        fresh_process_reload_verified=False,pilot_candidates_frozen=False,
        limits=['All successful reference, process and CUDA metadata here are synthetic rejection fixtures.',
                'Existing prepared package was compared with queue commands; completed old static suites were not rerun.',
                'Subprocess and GPU query functions were stubbed; this is not Linux flock or cluster runtime acceptance.',
                'Actual completed retain95 and separate development CUDA executor evidence are still required.'])
    output=Path(a.output)
    if output.exists():raise FileExistsError('Refuse to replace static evidence')
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status=result['status'],reference_rejections=len(rejected),
        snapshot_rejections=len(snapshot_rejected),development_rejections=len(dev_rejected),subprocesses_launched=0)))


if __name__=='__main__':
    main()
