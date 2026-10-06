"""Prepare private SPF NPO data and a bounded smoke input contract, without training.

The 40-row engineering smoke subset is separate from the full 200-row pilot.
This preparer does not freeze pilot candidates, create a checkpoint, or validate
model loading, loss scaling, masking, optimizer updates or GPU execution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_development_adapter import validate_target_audit

META_REVISION = '0e9e39f249a16976918f6564b8830bc894c89659'
TOFU_REVISION = '324592d84ae4f482ac7249b9285c2ecdb53e3a68'
FULL_SHA = 'cf6f9c9bd844b60661f1c923e6b188628f00b92c1834efb8f9f7c552274a9a33'
RETAIN_SHA = 'e1bfcea25c3237064c11676e9d8f52145032ccb45a0e7f858e7f8446475e8a3e'
TEMPLATE_SHA = '7d3e75bbebc8ca06f05579ff90e1730db88e7982d6c4ff524c55c736b21a10a0'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def row_sha(row):
    return sha(json.dumps(row,sort_keys=True,ensure_ascii=False,separators=(',', ':')).encode('utf8'))


def bound_json(path, digest):
    raw = Path(path).read_bytes()
    if sha(raw) != digest:
        raise ValueError('Bound input SHA mismatch: ' + str(path))
    return json.loads(raw), raw


def validate_inputs(audit, audit_sha, frozen, frozen_sha, adapter, full, retain):
    validate_target_audit(audit)
    if (audit.get('manifest_sha256') != frozen_sha
            or frozen.get('model_id') != 'meta-llama/Llama-3.1-8B-Instruct'
            or frozen.get('model_revision') != META_REVISION
            or frozen.get('tokenizer_revision') != META_REVISION
            or frozen.get('tofu_revision') != TOFU_REVISION
            or frozen.get('readiness', {}).get('status') != 'pass'
            or frozen['readiness'].get('world_size') != 1
            or frozen['readiness'].get('global_batch') != 32):
        raise ValueError('Audited fixed Meta/TOFU world1/global32 construction required')
    if (adapter.get('status') != 'prepared_not_evaluated'
            or adapter.get('checkpoint_audit_sha256') != audit_sha
            or adapter.get('world_size') != 1
            or adapter.get('template_sha256') != TEMPLATE_SHA
            or adapter.get('base_tokenizer_revision') != META_REVISION
            or adapter.get('base_generation_config_revision') != META_REVISION
            or adapter.get('checkpoint_identity_sha256') != sha(json.dumps(
                adapter.get('checkpoint_files'),sort_keys=True).encode('utf8'))):
        raise ValueError('SPF M_pre identity and pinned Meta template/tokenizer required')
    checkpoint_files = adapter.get('checkpoint_files', {})
    audited_shards = {entry['name']: entry['sha256'] for entry in audit['final_checkpoint']['files']}
    metadata_files = {'config.json', 'generation_config.json', 'target.json',
                      'tokenizer_config.json', 'trainer_state.json'}
    if len(audited_shards) > 1:
        metadata_files.add('model.safetensors.index.json')
    if (not isinstance(checkpoint_files, dict)
            or set(checkpoint_files) != set(audited_shards) | metadata_files
            or any(not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value)
                   for value in checkpoint_files.values())
            or any(checkpoint_files.get(name) != digest for name,digest in audited_shards.items())
            or not isinstance(adapter.get('checkpoint'), str)
            or not adapter['checkpoint'].startswith('/data/')
            or '..' in Path(adapter['checkpoint']).parts):
        raise ValueError('Adapter shard identities must match the bound completed full audit')
    for split, rows, count, digest in [('full',full,4000,FULL_SHA),('retain95',retain,3800,RETAIN_SHA)]:
        spec = frozen.get('datasets', {}).get(split, {})
        if not isinstance(rows, list) or len(rows) != count or spec.get('count') != count or spec.get('sha256') != digest:
            raise ValueError('Fixed dataset count/identity mismatch: ' + split)
        if any(not isinstance(r,dict) or set(r) != {'question','answer'}
               or any(not isinstance(r[k],str) or not r[k].strip() for k in ('question','answer')) for r in rows):
            raise ValueError('Unexpected QA structure: ' + split)
        hashes = [row_sha(r) for r in rows]
        if len(set(hashes)) != count or hashes != [r['row_sha256'] for r in spec.get('records', [])]:
            raise ValueError('Fixed row hashes/order mismatch: ' + split)
    # Frozen indices are local to each split. Do not assume split-local IDs
    # identify the same QA; join on the SHA of the unchanged QA row instead.
    full_hashes = [row_sha(r) for r in full]
    retain_hashes = {row_sha(r) for r in retain}
    if not retain_hashes <= set(full_hashes):
        raise ValueError('Retain rows must be an exact subset of the full QA rows')
    forget_positions = [i for i,h in enumerate(full_hashes) if h not in retain_hashes]
    if len(forget_positions) != 200:
        raise ValueError('Fixed forget05 must contain exactly 200 rows')
    return forget_positions


def prepare(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Refuse to replace an existing input preparation')
    audit, audit_raw = bound_json(args.target_audit,args.target_audit_sha256)
    adapter, adapter_raw = bound_json(args.adapter_audit,args.adapter_audit_sha256)
    frozen_raw = Path(args.frozen).read_bytes(); frozen = json.loads(frozen_raw)
    full, full_raw = bound_json(args.full_rows,FULL_SHA)
    retain, retain_raw = bound_json(args.retain_rows,RETAIN_SHA)
    positions = validate_inputs(audit,sha(audit_raw),frozen,sha(frozen_raw),adapter,full,retain)
    forget = [full[i] for i in positions]
    smoke = forget[:40]
    data = {'forget05.json':(json.dumps(forget,ensure_ascii=False,indent=2)+'\n').encode('utf8'),
            'forget05-smoke40.json':(json.dumps(smoke,ensure_ascii=False,indent=2)+'\n').encode('utf8'),
            'retain95.json':retain_raw}
    membership = dict(join='unchanged QA row SHA256; split-local IDs are not joined',
                      forget05=[dict(full_index=i,row_sha256=row_sha(full[i])) for i in positions],
                      smoke40_full_indices=positions[:40])
    data['membership.json'] = (json.dumps(membership,indent=2)+'\n').encode('utf8')
    result = dict(status='INPUTS_PREPARED_EXECUTOR_AND_GPU_RUNTIME_NOT_RUN',
                  target=adapter['checkpoint'],reference_model=adapter['checkpoint'],
                  checkpoint_identity_sha256=adapter['checkpoint_identity_sha256'],
                  target_audit_sha256=sha(audit_raw),adapter_audit_sha256=sha(adapter_raw),
                  frozen_sha256=sha(frozen_raw),model_revision=META_REVISION,tofu_revision=TOFU_REVISION,
                  full_input_sha256=sha(full_raw),retain95_input_sha256=sha(retain_raw),
                  full_count=4000,retain95_count=3800,forget05_count=200,
                  data={name:dict(size=len(raw),sha256=sha(raw)) for name,raw in data.items()},
                  smoke=dict(smoke_only=True,used_for_candidate_selection=False,
                    subset_rule='First 40 forget05 rows in frozen full QA order; engineering load/scaling smoke only',
                    forget_examples=40,retain_pool_examples=3800,retain_sampling='1:1 random sampling, seed0',
                    seed=0,alpha=0.0,gamma=1.0,beta=0.1,retain_loss_type='NLL',learning_rate=1e-7,
                    world_size=1,micro_batch=4,gradient_accumulation_steps=8,global_batch=32,
                    planned_epochs=1,expected_updates=2,expected_microbatches=10,
                    expected_window_examples=[32,8],expected_window_microbatches=[8,2],
                    expected_ds_backward_scalars=[0.125]*8+[0.5]*2,
                    spf_projection_in_unlearning=False,max_length=512,
                    counters_exposure_masking_and_scaling_runtime_validated=False),
                  pilot_candidates_frozen=False,gpu_evaluation_started=False,
                  target_gate_executed=False,npo_training_executed=False,
                  full_score_equivalence_verified=False,
                  source_sha256=sha(Path(__file__).read_bytes()),
                  limits=['Input preparation is not native NPO executor, model load, reference, masking, scaling or checkpoint acceptance.',
                          'No weights were read; weight identity is delegated to the bound completed full audit.',
                          'Native executor preparation and full/tail loss-scaling, frozen reference and fresh-process reload checks remain required.',
                          'The 40-row engineering subset must not replace the 200-row forget05 pilot or select a candidate.'])
    output.mkdir(parents=True,exist_ok=False)
    for name,raw in data.items():
        (output/name).write_bytes(raw)
    (output/'inputs.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('target-audit','target-audit-sha256','adapter-audit','adapter-audit-sha256',
                 'frozen','full-rows','retain-rows','output'):
        parser.add_argument('--'+name,required=True)
    result = prepare(parser.parse_args())
    print(json.dumps({k:result[k] for k in ('status','checkpoint_identity_sha256','full_count',
                                         'retain95_count','forget05_count','source_sha256')}))


if __name__ == '__main__':
    main()
