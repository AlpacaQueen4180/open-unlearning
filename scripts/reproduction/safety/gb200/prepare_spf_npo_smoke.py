"""Prepare a private bounded smoke package, without deploying or launching it."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

from spf_npo_native_probe import AUDIT_SHA, FROZEN_SHA, INPUT_SHAS, NATIVE_SOURCE_FILES

IDENTITY = 'c45e87fc69825b75f5d47345fb3c8502a4c1b4524f9d621411068bf3bc64f3a8'
TARGET = '/data/spf-npo-20261006-r4/spf-full-s0'
META = 'meta-llama/Llama-3.1-8B-Instruct'
REVISION = '0e9e39f249a16976918f6564b8830bc894c89659'
TEMPLATE_SHA = '7d3e75bbebc8ca06f05579ff90e1730db88e7982d6c4ff524c55c736b21a10a0'
CODE = ('spf_npo_native_probe.py', 'spf_npo_runtime.py',
        'prepare_spf_npo_smoke.py', 'verify_spf_npo_smoke_reload.py', 'audit_spf_target.py')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False)+'\n').encode('utf8')


def canonical_config(c):
    dataset = lambda name: {name: dict(handler='QADataset', args=dict(
        hf_args=dict(path='json', data_files=c['inputs'][name]['path'], split='train'),
        question_key='question', answer_key='answer', max_length=512,
        predict_with_generate=False))}
    return dict(defaults=[{'model':'Llama-3.1-8B-Instruct'}, '_self_'], mode='unlearn',
        model=dict(model_args=dict(pretrained_model_name_or_path=TARGET,
            local_files_only=True, use_cache=False),
            tokenizer_args=dict(pretrained_model_name_or_path=META, revision=REVISION,
                                local_files_only=True)),
        data=dict(anchor='forget', forget=dataset('forget'), retain=dataset('retain')),
        collator=dict(supervised=dict(handler='SPFSourceMaskCollator',
                                     args=dict(index='index', padding_side='right'))),
        trainer=dict(handler='NPO', method_args=dict(alpha=0.0, gamma=1.0, beta=0.1,
                                                    retain_loss_type='NLL'),
            args=dict(output_dir=c['output_dir'], logging_dir=c['output_dir']+'/logs',
                per_device_train_batch_size=4, per_device_eval_batch_size=4,
                gradient_accumulation_steps=8, learning_rate=1e-7, num_train_epochs=1,
                max_steps=-1, warmup_steps=0, lr_scheduler_type='linear',
                weight_decay=0.0, max_grad_norm=1.0, bf16=True, bf16_full_eval=True,
                optim='paged_adamw_32bit', gradient_checkpointing=True,
                ddp_find_unused_parameters=True, remove_unused_columns=False,
                dataloader_drop_last=False, dataloader_num_workers=0,
                do_train=True, do_eval=False, eval_on_start=False, eval_strategy='no',
                save_strategy='no', save_safetensors=True, save_only_model=True,
                seed=0, data_seed=0, report_to='none', logging_steps=1,
                deepspeed=c['deepspeed'])),
        hydra=dict(run=dict(dir=c['task_root']+'/hydra'), job=dict(chdir=False)))


def canonical_deepspeed():
    return dict(zero_optimization=dict(stage=3,
        offload_optimizer=dict(device='none', pin_memory=True),
        offload_param=dict(device='none', pin_memory=True), overlap_comm=True,
        contiguous_gradients=True, reduce_bucket_size='auto',
        stage3_prefetch_bucket_size='auto', stage3_param_persistence_threshold='auto',
        sub_group_size=1e9, stage3_max_live_parameters=1e9, stage3_max_reuse_distance=1e9,
        stage3_gather_16bit_weights_on_model_save=True), train_batch_size=32,
        train_micro_batch_size_per_gpu=4, gradient_accumulation_steps=8,
        bf16=dict(enabled=True))


def validate_contract_config(c, config, ds):
    root = PurePosixPath(c['task_root'])
    if (root.parent != PurePosixPath('/data') or not root.name.startswith('spf-npo-smoke-')
            or '..' in root.parts or c['output_dir'] != str(root/'checkpoint')
            or c['config_dir'] != str(root/'configs')
            or c['deepspeed'] != str(root/'deepspeed.json')
            or c['repository'] != '/data/spf-development-20261007-r1/repo'
            or c['target'] != TARGET or c['initial_reference'] != TARGET
            or c['checkpoint_identity_sha256'] != IDENTITY
            or c['smoke_only'] is not True or c['pilot_candidates_frozen'] is not False
            or c['frozen'] != '/data/spf-npo-20261006-r4/frozen.json'
            or c['target_audit'] != '/data/spf-npo-20261006-r4/spf-full-s0-audit.json'
            or set(c['inputs']) != set(INPUT_SHAS)):
        raise ValueError('Fixed independent SPF engineering smoke contract required')
    for name, file in [('forget','forget05-smoke40.json'), ('retain','retain95.json')]:
        if c['inputs'][name] != dict(path=str(root/'inputs'/file), sha256=INPUT_SHAS[name]):
            raise ValueError('Fixed engineering inputs required')
    if config != canonical_config(c) or ds != canonical_deepspeed():
        raise ValueError('Bounded native pure-NPO config/DeepSpeed profile changed')
    if set(c['code_sha256']) != set(CODE):
        raise ValueError('Complete observer/reload source guard required')
    if (set(c['config_sha256']) != {'configs/spf_npo_smoke.yaml','deepspeed.json',
                                  'configs/model/Llama-3.1-8B-Instruct.yaml'}
            or c['config_sha256']['configs/model/Llama-3.1-8B-Instruct.yaml'] != TEMPLATE_SHA
            or c['config_sha256']['configs/spf_npo_smoke.yaml'] != sha(encoded(config))
            or c['config_sha256']['deepspeed.json'] != sha(encoded(ds))):
        raise ValueError('Exact bound config, DeepSpeed and Meta template SHA set required')


def prepare(args):
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError('Refuse to replace a smoke preparation')
    inputs_raw = (Path(args.inputs)/'inputs.json').read_bytes()
    inputs = json.loads(inputs_raw)
    frozen_raw = Path(args.frozen).read_bytes()
    audit_raw = Path(args.target_audit).read_bytes()
    if (sha(frozen_raw) != FROZEN_SHA or sha(audit_raw) != AUDIT_SHA
            or inputs['frozen_sha256'] != FROZEN_SHA or inputs['target_audit_sha256'] != AUDIT_SHA
            or inputs['target'] != TARGET or inputs['reference_model'] != TARGET
            or inputs['checkpoint_identity_sha256'] != IDENTITY
            or inputs['pilot_candidates_frozen'] is not False
            or inputs['forget05_count'] != 200 or inputs['retain95_count'] != 3800
            or inputs['smoke']['forget_examples'] != 40):
        raise ValueError('Actual completed SPF target and prepared engineering inputs required')
    directory = Path(__file__).resolve().parent
    template = Path(args.template).read_bytes()
    if sha(template) != TEMPLATE_SHA:
        raise ValueError('Fixed Meta template bytes required')
    root = PurePosixPath(args.task_root)
    c = dict(schema_version=1, task_root=str(root), repository='/data/spf-development-20261007-r1/repo',
        output_dir=str(root/'checkpoint'), config_dir=str(root/'configs'),
        deepspeed=str(root/'deepspeed.json'), target=TARGET, initial_reference=TARGET,
        checkpoint_identity_sha256=IDENTITY, smoke_only=True, pilot_candidates_frozen=False,
        frozen='/data/spf-npo-20261006-r4/frozen.json',
        target_audit='/data/spf-npo-20261006-r4/spf-full-s0-audit.json',
        native_source_sha256={p:json.loads(frozen_raw)['construction_code'][p] for p in NATIVE_SOURCE_FILES},
        inputs={name:dict(path=str(root/'inputs'/file), sha256=INPUT_SHAS[name])
                for name,file in [('forget','forget05-smoke40.json'),('retain','retain95.json')]},
        code_sha256={p:sha((directory/p).read_bytes()) for p in CODE},
        input_preparation_sha256=sha(inputs_raw))
    config, ds = canonical_config(c), canonical_deepspeed()
    data = {'configs/spf_npo_smoke.yaml':encoded(config), 'deepspeed.json':encoded(ds),
            'configs/model/Llama-3.1-8B-Instruct.yaml':template}
    for name,file in [('forget','forget05-smoke40.json'),('retain','retain95.json')]:
        raw = (Path(args.inputs)/file).read_bytes()
        if sha(raw) != INPUT_SHAS[name] or len(json.loads(raw)) != {'forget':40,'retain':3800}[name]:
            raise ValueError('Bound prepared QA bytes changed')
        data['inputs/'+file] = raw
    c['config_sha256'] = {p:sha(data[p]) for p in data if p.startswith('configs/') or p=='deepspeed.json'}
    validate_contract_config(c, config, ds)
    data['contract.private.json'] = encoded(c)
    contract_sha = sha(data['contract.private.json'])
    python = '/data/npo-gb200-20261004/validation-20261006/venv/bin/python'
    commands = dict(cwd=c['repository'], environment=dict(
        SPF_NPO_SMOKE_CONTRACT=str(root/'contract.private.json'),
        SPF_NPO_SMOKE_CONTRACT_SHA256=contract_sha, HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        HF_HOME='/data/smart-mfg/jimmy-lin/hf_cache', TOKENIZERS_PARALLELISM='false',
        WANDB_MODE='disabled', HYDRA_FULL_ERROR='1'),
        train=[python,'-m','torch.distributed.run','--standalone','--nproc_per_node=1',
               str(root/'code/spf_npo_native_probe.py'),'--config-path',c['config_dir'],
               '--config-name','spf_npo_smoke'],
        fresh_reload=[python,str(root/'code/verify_spf_npo_smoke_reload.py'),
                      '--contract',str(root/'contract.private.json'),'--contract-sha256',contract_sha],
        launch_authorization_condition='New complete reference train/reload/audit + original GPU idle snapshot required; no launch by this preparer.',
        training_resume_supported=False, evaluation_only_recovery_after_complete_checkpoint=True)
    data['commands.private.json'] = encoded(commands)
    for name in CODE:
        data['code/'+name] = (directory/name).read_bytes()
    manifest = dict(status='PACKAGE_PREPARED_NOT_DEPLOYED_OR_LAUNCHED',
        task_root=str(root), contract_sha256=contract_sha,
        files={name:dict(size=len(raw),sha256=sha(raw)) for name,raw in data.items()},
        torch_runtime_validated=False, hydra_runtime_composition_validated=False,
        npo_training_executed=False, fresh_process_reload_verified=False, pilot_candidates_frozen=False,
        reload_atol=0.05, reload_rtol=0.01,
        numerical_settings_or_rotary_buffers_modified=False,
        source_sha256=sha(Path(__file__).read_bytes()))
    out.mkdir(parents=True,exist_ok=False)
    for name,raw in data.items():
        path=out/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    (out/'package.private.json').write_bytes(encoded(manifest))
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','frozen','target-audit','template','task-root','output'):
        p.add_argument('--'+name,required=True)
    r=prepare(p.parse_args())
    print(json.dumps({k:r[k] for k in ('status','task_root','contract_sha256','source_sha256')}))


if __name__=='__main__':
    main()
