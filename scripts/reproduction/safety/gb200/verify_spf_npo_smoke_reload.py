"""Fresh-process HF reload and full export audit, after the bounded SPF smoke."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from audit_spf_target import weights
from spf_npo_runtime import audit_trace


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(16*1024**2),b''):
            h.update(block)
    return h.hexdigest()


def check_reload_metadata(contract, runtime, record, trainer_state):
    audit_trace(runtime)
    if (runtime['contract_sha256'] != record['contract_sha256']
            or trainer_state['global_step'] != 2 or trainer_state['epoch'] != 1.0
            or record['capture_process_pid'] == os.getpid()
            or record['comparison'] != dict(atol=0.05,rtol=0.01)
            or record['checkpoint'] != contract['output_dir']
            or runtime['fresh_process_reload_verified'] is not False
            or record['numerical_settings_or_rotary_buffers_modified'] is not False):
        raise ValueError('Complete independent smoke, counts and original reload threshold required')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',required=True);p.add_argument('--contract-sha256',required=True)
    a=p.parse_args()
    if sha_file(a.contract) != a.contract_sha256:
        raise ValueError('Bound contract SHA mismatch')
    c=json.loads(Path(a.contract).read_bytes())
    root=Path(c['task_root']);target=Path(c['output_dir'])
    for name,digest in c['code_sha256'].items():
        if sha_file(root/'code'/name) != digest:
            raise ValueError('Observer/reload code changed: '+name)
    from prepare_spf_npo_smoke import validate_contract_config
    validate_contract_config(c,json.loads((root/'configs/spf_npo_smoke.yaml').read_bytes()),
                             json.loads((root/'deepspeed.json').read_bytes()))
    runtime=json.loads((target/'spf-npo-runtime.private.json').read_bytes())
    record=json.loads((target/'spf-npo-reload-capture.json').read_bytes())
    state=json.loads((target/'trainer_state.json').read_bytes())
    check_reload_metadata(c,runtime,record,state)
    if record['contract_sha256'] != a.contract_sha256 or sha_file(target/'reload-probe.private.pt') != record['probe_sha256']:
        raise ValueError('Bound post-training numerical probe changed')
    if any(int(os.environ.get(k,v)) != n for k,v,n in [('WORLD_SIZE','1',1),('RANK','0',0),('LOCAL_RANK','0',0)]):
        raise ValueError('Fresh reload requires world1/rank0/local0')
    export=weights(target)
    import torch
    import transformers
    from transformers import AutoModelForCausalLM
    if transformers.__version__ != '5.5.4' or torch.cuda.device_count()!=1:
        raise ValueError('Pinned Transformers and single GPU required')
    probe=torch.load(target/'reload-probe.private.pt',map_location='cpu',weights_only=True)
    model=AutoModelForCausalLM.from_pretrained(target,torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',local_files_only=True).cuda().eval()
    buffers={name:dict(shape=list(b.shape),dtype=str(b.dtype)) for name,b in model.named_buffers()}
    current_context=dict(torch_version=torch.__version__,transformers_version=transformers.__version__,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        tf32_override=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'))
    if current_context != record['numerical_context']:
        raise ValueError('Original training/reload numerical context differs')
    result=dict(status='FAILED_RELOAD',contract_sha256=a.contract_sha256,
        fresh_reload_pid=os.getpid(),capture_process_pid=record['capture_process_pid'],
        final_checkpoint=export,comparison=record['comparison'],
        numerical_context=current_context,
        buffer_metadata_matches=record['buffers']==buffers,
        numerical_settings_or_rotary_buffers_modified=False,pilot_candidates_frozen=False,
        target_gate_evaluated=False)
    try:
        with torch.no_grad():
            logits=model(**{k:v.cuda() for k,v in probe['inputs'].items()}).logits[0,-1].float().cpu()
        result['max_logit_error']=float((logits-probe['logits']).abs().max())
        torch.save(dict(actual_logits=logits,expected_logits=probe['logits']),target/'reload-comparison.private.pt')
        torch.testing.assert_close(logits,probe['logits'],atol=0.05,rtol=0.01)
        result.update(status='SMOKE_TRAINING_EXPORT_FRESH_RELOAD_PASS',fresh_process_reload_verified=True)
    except Exception as error:
        result.update(error=repr(error),fresh_process_reload_verified=False)
        raise
    finally:
        (target/'spf-npo-fresh-reload-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:result[k] for k in ('status','max_logit_error','fresh_process_reload_verified')}))


if __name__=='__main__':
    main()
