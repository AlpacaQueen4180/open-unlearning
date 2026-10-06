"""Bounded SPF smoke observer; launch/config and fresh reload are separate stages.

Requires an immutable contract with repository, output, initial target identity,
native source SHA map and the already prepared private input digests. It must be
launched only after a new complete reference/audit/GPU-idle snapshot.
"""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

from spf_npo_runtime import audit_trace, source_mask_collator

NATIVE_SOURCE_FILES = ('src/train.py', 'src/model/__init__.py', 'src/trainer/__init__.py',
    'src/trainer/base.py', 'src/trainer/utils.py', 'src/trainer/unlearn/npo.py',
    'src/trainer/unlearn/grad_diff.py', 'src/trainer/unlearn/base.py',
    'src/data/__init__.py', 'src/data/utils.py', 'src/data/qa.py',
    'src/data/unlearn.py', 'src/data/collators.py', 'configs/model/Llama-3.1-8B-Instruct.yaml')
FROZEN_SHA = '122edbe2c09fcd2903e5b8949657f3dcaf3bf5cbc527e05b4cce91a29fb04d9e'
AUDIT_SHA = '7a7036b903ced64e18307e90523cc7ada73699ddb3d79498bfd4063f86ed8cf9'
INPUT_SHAS = dict(forget='78ec712fc1c1708539d0ba883111293907a562a7eba2a801c86704f6747593de',
                  retain='e1bfcea25c3237064c11676e9d8f52145032ccb45a0e7f858e7f8446475e8a3e')

def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    path = Path(os.environ['SPF_NPO_SMOKE_CONTRACT'])
    raw = path.read_bytes()
    if digest(raw) != os.environ['SPF_NPO_SMOKE_CONTRACT_SHA256']:
        raise ValueError('SPF smoke contract SHA mismatch')
    contract = json.loads(raw)
    if (contract['smoke_only'] is not True or contract['pilot_candidates_frozen'] is not False
            or contract['target'] != contract['initial_reference']
            or any(int(os.environ.get(k, default)) != expected for k,default,expected in
                   [('WORLD_SIZE','1',1),('RANK','0',0),('LOCAL_RANK','0',0)])):
        raise ValueError('Fixed world1 SPF engineering smoke required')
    if (contract['target'] != '/data/spf-npo-20261006-r4/spf-full-s0'
            or contract['checkpoint_identity_sha256'] != 'c45e87fc69825b75f5d47345fb3c8502a4c1b4524f9d621411068bf3bc64f3a8'):
        raise ValueError('Audited seed0 SPF target required')
    frozen_raw = Path(contract['frozen']).read_bytes()
    audit_raw = Path(contract['target_audit']).read_bytes()
    if digest(frozen_raw) != FROZEN_SHA or digest(audit_raw) != AUDIT_SHA:
        raise ValueError('Bound completed SPF full audit/freeze required')
    frozen = json.loads(frozen_raw)
    if contract['native_source_sha256'] != {name:frozen['construction_code'][name] for name in NATIVE_SOURCE_FILES}:
        raise ValueError('Complete native source guard required')
    repository = Path(contract['repository'])
    for name, expected in contract['native_source_sha256'].items():
        if digest((repository/name).read_bytes()) != expected:
            raise ValueError('Native source changed: ' + name)
    if set(contract['inputs']) != set(INPUT_SHAS):
        raise ValueError('Exactly the fixed forget40 and retain95 inputs required')
    for name, spec in contract['inputs'].items():
        data = Path(spec['path']).read_bytes()
        if digest(data) != spec['sha256'] or spec['sha256'] != INPUT_SHAS[name]:
            raise ValueError('Private fixed input changed: ' + name)
    from prepare_spf_npo_smoke import validate_contract_config
    root = Path(contract['task_root'])
    for name, expected in contract['config_sha256'].items():
        if digest((root/name).read_bytes()) != expected:
            raise ValueError('Bound Hydra/DeepSpeed/template bytes changed: '+name)
    for name, expected in contract['code_sha256'].items():
        if digest((root/'code'/name).read_bytes()) != expected:
            raise ValueError('Bound observer/reload source changed: '+name)
    validate_contract_config(contract,json.loads((root/'configs/spf_npo_smoke.yaml').read_bytes()),
                             json.loads((root/'deepspeed.json').read_bytes()))
    if sys.argv[1:] != ['--config-path',contract['config_dir'],'--config-name','spf_npo_smoke']:
        raise ValueError('Exact immutable smoke config required; no Hydra overrides')
    if Path(contract['output_dir']).exists():
        raise FileExistsError('Refuse to restart or replace an existing smoke checkpoint')
    sys.path.insert(0, str(repository/'src'))
    import torch
    import deepspeed
    import transformers
    import accelerate
    from data import _register_collator
    from data.collators import DataCollatorForSupervisedDataset
    from trainer.unlearn.npo import NPO
    if (transformers.__version__ != '5.5.4' or accelerate.__version__ != '1.13.0'
            or deepspeed.__version__ != '0.15.4'):
        raise ValueError('Pinned native Trainer/Accelerate/DeepSpeed required')
    _register_collator(source_mask_collator(DataCollatorForSupervisedDataset))

    def fingerprint(model):
        module = getattr(model, 'module', model)
        manifest = []
        for name, parameter in module.named_parameters():
            with deepspeed.zero.GatheredParameters([parameter], modifier_rank=None):
                tensor = parameter.detach().contiguous()
                content = tensor.view(torch.uint8).cpu().numpy().tobytes()
                manifest.append(dict(name=name,shape=list(tensor.shape),dtype=str(tensor.dtype),
                                     sha256=digest(content)))
        if len(manifest) != 291:
            raise ValueError('Complete 8B reference fingerprint requires 291 tensors')
        return digest(json.dumps(manifest,sort_keys=True).encode('utf8'))

    original_loss, original_step = NPO.compute_loss, NPO.training_step
    observed = None
    first_current = first_reference = None
    microsteps = exposures = 0
    forget_indices = []
    trace = []

    def compute_loss(self, *args, **kwargs):
        result = original_loss(self, *args, **kwargs)
        loss = result[0] if isinstance(result,tuple) else result
        self._spf_smoke_raw_loss = float(loss.detach())
        return result

    def training_step(self, model, inputs, *args, **kwargs):
        nonlocal observed, first_current, first_reference, microsteps, exposures
        observed = self
        if microsteps == 0:
            if (Path(self.args.output_dir) != Path(contract['output_dir'])
                    or self.model.config._name_or_path != contract['target']
                    or self.ref_model is None
                    or len(self.train_dataset.forget) != 40 or len(self.train_dataset.retain) != 3800
                    or self.train_dataset.anchor != 'forget'
                    or not isinstance(self.data_collator, DataCollatorForSupervisedDataset)):
                raise ValueError('Unexpected local SPF model, paired data or collator')
            if self.data_collator.__class__.__name__ != 'SPFSourceMaskCollator':
                raise ValueError('Explicit source-mask collator required')
            first_current, first_reference = fingerprint(model), fingerprint(self.ref_model)
            if first_current != first_reference:
                raise ValueError('Initial SPF target and NPO reference differ')
            original_backward = model.backward

            def backward(loss, *backward_args, **backward_kwargs):
                trace.append(dict(microstep=microsteps, raw_npo_loss=self._spf_smoke_raw_loss,
                    loss_entering_ds_backward=float(loss.detach()),
                    actual_window_microbatches=self.current_gradient_accumulation_steps,
                    boundary=model.is_gradient_accumulation_boundary(),
                    kwargs=backward_kwargs,mask_valid=True))
                return original_backward(loss, *backward_args, **backward_kwargs)
            model.backward = backward
        microsteps += 1
        forget, retain = inputs['forget'], inputs['retain']
        if forget['input_ids'].shape[0] != 4 or retain['input_ids'].shape[0] != 4:
            raise ValueError('Expected four forget and four retain rows per microbatch')
        for batch in (forget, retain):
            if (batch['input_ids'].shape[1] > 512
                    or torch.any((batch['labels'] != -100) & ~batch['attention_mask'].bool())
                    or torch.any((batch['labels'][:,1:] != -100).sum(dim=1) == 0)):
                raise ValueError('Valid assistant token masked or sequence too long')
        exposures += 4
        forget_indices.extend(forget['index'].detach().cpu().tolist())
        return original_step(self, model, inputs, *args, **kwargs)

    NPO.compute_loss, NPO.training_step = compute_loss, training_step
    runpy.run_path(str(repository/'src/train.py'),run_name='__main__')
    if observed is None:
        raise ValueError('No actual NPO training step observed')
    trainer = observed
    engine = trainer.model_wrapped
    final_reference = fingerprint(trainer.ref_model)
    if any(p.grad is not None for p in trainer.ref_model.parameters()):
        raise ValueError('NPO reference accumulated gradients')
    runtime = dict(trainer_global_step=trainer.state.global_step,engine_global_steps=engine.global_steps,
        microsteps=microsteps,epoch=trainer.state.epoch,world_size=trainer.accelerator.num_processes,
        micro_batch=trainer.args.per_device_train_batch_size,configured_gas=trainer.args.gradient_accumulation_steps,
        alpha=trainer.alpha,gamma=trainer.gamma,beta=trainer.beta,retain_loss_type=trainer.retain_loss_type,
        learning_rate=trainer.args.learning_rate,dataset_examples=len(trainer.train_dataset),
        retain_pool_examples=len(trainer.train_dataset.retain),forget_exposures=exposures,
        attention=engine.module.config._attn_implementation,optimizer=trainer.args.optim.value,
        zero_stage=trainer.accelerator.state.deepspeed_plugin.deepspeed_config['zero_optimization']['stage'],
        retain_exposures=exposures,unique_forget_examples=len(set(forget_indices)),
        initial_target_reference_equal=first_current==first_reference,
        reference_fingerprint_unchanged=first_reference==final_reference,
        initial_parameter_fingerprint=first_current,final_reference_fingerprint=final_reference,
        loss_scaling_trace=trace,contract_sha256=digest(raw),
        observer_policy='Native original loss/training_step/backward return values unchanged; no boundary mutation.',
        fresh_process_reload_verified=False,pilot_candidates_frozen=False)
    output = Path(contract['output_dir'])
    (output/'spf-npo-runtime.private.json').write_text(json.dumps(runtime,indent=2)+'\n',encoding='utf8')
    result = audit_trace(runtime)
    (output/'spf-npo-trace-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    # Capture only after the native training and save_model have completed.
    # No buffer precision changes or extra optimizer steps are introduced.
    engine.eval()
    batch = trainer.data_collator([trainer.train_dataset.forget[0]])
    probe_inputs = {k:v.to(trainer.accelerator.device) for k,v in batch.items()
                    if k in ('input_ids','attention_mask')}
    with torch.no_grad():
        expected = engine(**probe_inputs).logits[0,-1].float().cpu()
    probe_path = output/'reload-probe.private.pt'
    torch.save(dict(inputs={k:v.cpu() for k,v in probe_inputs.items()},logits=expected),probe_path)
    context = dict(torch_version=torch.__version__,transformers_version=transformers.__version__,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        tf32_override=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'))
    capture = dict(contract_sha256=digest(raw),checkpoint=str(output),capture_process_pid=os.getpid(),
        probe_sha256=digest(probe_path.read_bytes()),comparison=dict(atol=0.05,rtol=0.01),
        numerical_context=context,buffers={name:dict(shape=list(b.shape),dtype=str(b.dtype))
                for name,b in engine.module.named_buffers()},
        numerical_settings_or_rotary_buffers_modified=False,fresh_process_reload_verified=False)
    (output/'spf-npo-reload-capture.json').write_text(json.dumps(capture,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
