"""Static package, immutable config and synthetic reload rejection checks only."""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

from prepare_spf_npo_smoke import canonical_config, canonical_deepspeed, validate_contract_config
from verify_spf_npo_smoke_reload import check_reload_metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.package)
    c=json.loads((root/'contract.private.json').read_bytes())
    package=json.loads((root/'package.private.json').read_bytes())
    config=json.loads((root/'configs/spf_npo_smoke.yaml').read_bytes())
    ds=json.loads((root/'deepspeed.json').read_bytes())
    for name,spec in package['files'].items():
        raw=(root/name).read_bytes()
        assert len(raw)==spec['size'] and hashlib.sha256(raw).hexdigest()==spec['sha256'],name
    validate_contract_config(c,config,ds)
    rejected=[]
    cases=[('alpha1',lambda q,f,d:f['trainer']['method_args'].update(alpha=1)),
        ('multi_epoch',lambda q,f,d:f['trainer']['args'].update(num_train_epochs=2)),
        ('forced_max_steps',lambda q,f,d:f['trainer']['args'].update(max_steps=2)),
        ('lr_changed',lambda q,f,d:f['trainer']['args'].update(learning_rate=1e-6)),
        ('gas_changed',lambda q,f,d:f['trainer']['args'].update(gradient_accumulation_steps=4)),
        ('eval_enabled',lambda q,f,d:f['trainer']['args'].update(do_eval=True)),
        ('model_changed',lambda q,f,d:f['model']['model_args'].update(pretrained_model_name_or_path='other')),
        ('tokenizer_revision_changed',lambda q,f,d:f['model']['tokenizer_args'].update(revision='main')),
        ('native_id_mask_collator',lambda q,f,d:f['collator']['supervised'].update(handler='DataCollatorForSupervisedDataset')),
        ('retain_pool_changed',lambda q,f,d:q['inputs']['retain'].update(sha256='0'*64)),
        ('unsafe_task_root',lambda q,f,d:q.update(task_root='/data/spf-npo-20261006-r4')),
        ('zero_stage_changed',lambda q,f,d:d['zero_optimization'].update(stage=2)),
        ('global_batch_changed',lambda q,f,d:d.update(train_batch_size=16)),
        ('missing_code_guard',lambda q,f,d:q['code_sha256'].pop('spf_npo_runtime.py')),
        ('missing_template_guard',lambda q,f,d:q['config_sha256'].pop('configs/model/Llama-3.1-8B-Instruct.yaml')),
        ('spurious_config_guard',lambda q,f,d:q['config_sha256'].update(other='0'*64))]
    for name,mutate in cases:
        q,f,d=deepcopy(c),deepcopy(config),deepcopy(ds);mutate(q,f,d)
        try:validate_contract_config(q,f,d)
        except ValueError:rejected.append(name)
        else:raise AssertionError('Invalid config accepted: '+name)
    commands=json.loads((root/'commands.private.json').read_bytes())
    assert commands['train'][-4:]==['--config-path',c['config_dir'],'--config-name','spf_npo_smoke']
    assert '--nproc_per_node=1' in commands['train']
    assert not any('corrected' in x or 'force' in x or 'resume' in x for x in commands['train'])
    assert 'torch.distributed.run' in commands['train']
    assert commands['environment']['SPF_NPO_SMOKE_CONTRACT_SHA256']==package['contract_sha256']
    runtime=dict(trainer_global_step=2,engine_global_steps=2,microsteps=10,epoch=1.0,
        world_size=1,micro_batch=4,configured_gas=8,alpha=0.0,gamma=1.0,beta=0.1,
        retain_loss_type='NLL',learning_rate=1e-7,dataset_examples=40,retain_pool_examples=3800,
        forget_exposures=40,retain_exposures=40,unique_forget_examples=40,
        attention='flash_attention_2',optimizer='paged_adamw_32bit',zero_stage=3,
        initial_target_reference_equal=True,reference_fingerprint_unchanged=True,
        fresh_process_reload_verified=False,contract_sha256=package['contract_sha256'],
        loss_scaling_trace=[dict(raw_npo_loss=13.0,loss_entering_ds_backward=13/window,
            actual_window_microbatches=window,boundary=i in (7,9),
            kwargs=dict(scale_wrt_gas=False),mask_valid=True) for i,window in enumerate([8]*8+[2]*2)])
    record=dict(contract_sha256=package['contract_sha256'],capture_process_pid=os.getpid()+1,
        comparison=dict(atol=.05,rtol=.01),checkpoint=c['output_dir'],
        numerical_settings_or_rotary_buffers_modified=False)
    state=dict(global_step=2,epoch=1.0)
    check_reload_metadata(c,runtime,record,state)
    reload_cases=[('same_process',lambda r,s:r.update(capture_process_pid=os.getpid())),
        ('relaxed_atol',lambda r,s:r['comparison'].update(atol=.1)),
        ('wrong_epoch',lambda r,s:s.update(epoch=2.0)),
        ('wrong_steps',lambda r,s:s.update(global_step=1)),
        ('wrong_checkpoint',lambda r,s:r.update(checkpoint=c['target'])),
        ('numerics_modified',lambda r,s:r.update(numerical_settings_or_rotary_buffers_modified=True))]
    reload_rejected=[]
    for name,mutate in reload_cases:
        r,s=deepcopy(record),deepcopy(state);mutate(r,s)
        try:check_reload_metadata(c,runtime,r,s)
        except ValueError:reload_rejected.append(name)
        else:raise AssertionError('Invalid reload accepted: '+name)
    directory=Path(__file__).resolve().parent
    sources=('prepare_spf_npo_smoke.py','verify_spf_npo_smoke_reload.py',
             'spf_npo_native_probe.py','verify_spf_npo_smoke_static.py')
    for name in sources:
        compile((directory/name).read_bytes(),name,'exec')
    result=dict(status='PASS_PACKAGE_BYTES_CONFIG_AND_SYNTHETIC_RELOAD_METADATA_ONLY',
        package_files_byte_verified=len(package['files']),contract_sha256=package['contract_sha256'],
        bounded_config_rejections=rejected,synthetic_reload_metadata_rejections=reload_rejected,
        source_sha256={n:hashlib.sha256((directory/n).read_bytes()).hexdigest() for n in sources},
        torch_runtime_validation='not_run',hydra_runtime_composition_validated=False,
        gpu_runtime_validation='not_run',npo_training_executed=False,
        fresh_process_reload_verified=False,pilot_candidates_frozen=False,
        limits=['JSON syntax is valid YAML; actual Hydra/default composition is still required in the remote runtime.',
                'Synthetic counts/reference flags and process IDs are rejection fixtures, not model/reload evidence.',
                'No Torch import, weight read, GPU operation, deployment or launch was performed.'])
    output=Path(a.output)
    if output.exists():raise FileExistsError('Refuse to replace static evidence')
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status=result['status'],package_files=len(package['files']),
        config_rejections=len(rejected),reload_metadata_rejections=len(reload_rejected))))


if __name__=='__main__':
    main()
