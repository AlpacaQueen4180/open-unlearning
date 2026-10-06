"""Verify private input membership and reject source/identity drift, without models."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_spf_npo_inputs as preparer


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('target-audit', 'adapter-audit', 'frozen', 'full-rows',
                 'retain-rows', 'prepared-inputs', 'output'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    raw = {name: Path(getattr(a, name.replace('-', '_'))).read_bytes()
           for name in ('target-audit', 'adapter-audit', 'frozen', 'full-rows', 'retain-rows')}
    audit, adapter, frozen, full, retain = [json.loads(raw[name]) for name in
        ('target-audit', 'adapter-audit', 'frozen', 'full-rows', 'retain-rows')]
    root = Path(a.prepared_inputs)
    metadata_raw = (root / 'inputs.json').read_bytes()
    metadata = json.loads(metadata_raw)
    assert metadata['target_audit_sha256'] == digest(raw['target-audit'])
    assert metadata['adapter_audit_sha256'] == digest(raw['adapter-audit'])
    assert metadata['frozen_sha256'] == digest(raw['frozen'])
    assert metadata['source_sha256'] == digest(Path(preparer.__file__).read_bytes())
    assert digest(raw['full-rows']) == preparer.FULL_SHA
    assert digest(raw['retain-rows']) == preparer.RETAIN_SHA
    for name, spec in metadata['data'].items():
        data = (root / name).read_bytes()
        assert len(data) == spec['size'] and digest(data) == spec['sha256']
    forget = json.loads((root / 'forget05.json').read_bytes())
    smoke = json.loads((root / 'forget05-smoke40.json').read_bytes())
    # Use exact question/answer pairs independently of the preparer's SHA join.
    key = lambda row: (row['question'], row['answer'])
    full_keys = list(map(key, full))
    retain_keys = set(map(key, retain))
    positions = [i for i, row in enumerate(full_keys) if row not in retain_keys]
    assert len(set(full_keys)) == 4000 and len(retain_keys) == 3800
    assert len(positions) == 200 and forget == [full[i] for i in positions]
    assert smoke == forget[:40] and len(smoke) == 40
    assert (root / 'retain95.json').read_bytes() == raw['retain-rows']
    assert set(map(key, forget)).isdisjoint(retain_keys)
    assert set(map(key, forget)) | retain_keys == set(full_keys)
    membership = json.loads((root / 'membership.json').read_bytes())
    assert [entry['full_index'] for entry in membership['forget05']] == positions
    assert membership['smoke40_full_indices'] == positions[:40]
    full_ids = {r['id'] for r in frozen['datasets']['full']['records']}
    retain_ids = {r['id'] for r in frozen['datasets']['retain95']['records']}
    id_difference_positions = [i for i, r in enumerate(frozen['datasets']['full']['records'])
                               if r['id'] in full_ids - retain_ids]
    actual_id_difference_matches = id_difference_positions == positions
    id_rows_agree = all(next(r['row_sha256'] for r in frozen['datasets']['full']['records']
                            if r['id'] == entry['id']) == entry['row_sha256']
                        for entry in frozen['datasets']['retain95']['records'])
    assert actual_id_difference_matches and id_rows_agree
    # A new synthetic metadata variant proves the SHA join survives a split-ID
    # change. This variant is not an actual construction or checkpoint audit.
    variant = deepcopy(frozen)
    for entry in variant['datasets']['retain95']['records']:
        entry['id'] = 'synthetic-reindexed-' + str(entry['id'])
    variant_sha = digest(json.dumps(variant, sort_keys=True).encode('utf8'))
    variant_audit = deepcopy(audit)
    variant_audit['manifest_sha256'] = variant_sha
    assert preparer.validate_inputs(variant_audit, digest(raw['target-audit']), variant,
        variant_sha, adapter, full, retain) == positions
    assert metadata['target'] == metadata['reference_model'] == adapter['checkpoint']
    assert not metadata['pilot_candidates_frozen'] and not metadata['npo_training_executed']
    assert metadata['smoke']['used_for_candidate_selection'] is False
    assert metadata['smoke']['counters_exposure_masking_and_scaling_runtime_validated'] is False

    cases = []

    def reject(name, mutation):
        changed = deepcopy((audit, frozen, adapter, full, retain))
        mutation(changed)
        try:
            preparer.validate_inputs(changed[0], digest(raw['target-audit']), changed[1],
                                    digest(raw['frozen']), changed[2], changed[3], changed[4])
        except (ValueError, KeyError, TypeError):
            cases.append(dict(case=name, rejected=True))
        else:
            raise AssertionError('Invalid input accepted: ' + name)

    def reidentify(changed):
        changed[2]['checkpoint_identity_sha256'] = digest(
            json.dumps(changed[2]['checkpoint_files'], sort_keys=True).encode('utf8'))

    def weight_change(changed):
        name = audit['final_checkpoint']['files'][0]['name']
        changed[2]['checkpoint_files'][name] = '0' * 64
        reidentify(changed)

    def missing_weight(changed):
        del changed[2]['checkpoint_files'][audit['final_checkpoint']['files'][0]['name']]
        reidentify(changed)

    def extra_weight(changed):
        changed[2]['checkpoint_files']['unexpected.safetensors'] = '1' * 64
        reidentify(changed)

    reject('self_consistent_adapter_with_changed_audited_weight', weight_change)
    reject('self_consistent_adapter_missing_audited_weight', missing_weight)
    reject('self_consistent_adapter_with_unexpected_weight', extra_weight)
    reject('adapter_bound_to_other_audit', lambda c: c[2].update(checkpoint_audit_sha256='0' * 64))
    reject('unbound_frozen_manifest', lambda c: c[0].update(manifest_sha256='0' * 64))
    reject('other_tokenizer_revision', lambda c: c[2].update(base_tokenizer_revision='0' * 40))
    reject('reversed_retain_order', lambda c: c[4].reverse())
    reject('duplicate_full_qa', lambda c: c[3].__setitem__(1, c[3][0]))
    reject('reversed_full_order', lambda c: c[3].reverse())
    reject('unexpected_qa_column', lambda c: c[3][0].update(extra_column='synthetic'))
    reject('non_world1_profile', lambda c: c[1]['readiness'].update(world_size=2))
    reject('wrong_split_count', lambda c: c[1]['datasets']['retain95'].update(count=3799))
    result = dict(status='PASS_PRIVATE_INPUT_MEMBERSHIP_AND_FAIL_CLOSED_CHECKS',
        preparer_sha256=digest(Path(preparer.__file__).read_bytes()),
        verifier_sha256=digest(Path(__file__).read_bytes()),
        target_audit_sha256=digest(raw['target-audit']), adapter_audit_sha256=digest(raw['adapter-audit']),
        frozen_sha256=digest(raw['frozen']), private_inputs_metadata_sha256=digest(metadata_raw),
        counts=dict(full=4000, retain95=3800, forget05=200, engineering_smoke=40),
        exact_qa_pair_join_matches_sha_join=True, retain_bytes_preserved=True,
        actual_split_id_join_verified_by_row_sha=True,
        actual_id_difference_matches_qa_membership=actual_id_difference_matches,
        synthetic_reindexed_retain_ids_preserve_qa_membership=True,
        synthetic_id_variant_is_actual_checkpoint_audit=False, negative_cases=cases,
        gpu_runtime_validation='not_run', native_npo_executor_prepared=False,
        npo_smoke_executed=False, npo_pilot_executed=False, pilot_candidates_frozen=False,
        target_gate_executed=False,
        limits=['These checks establish private fixed input membership and identity rejection only.',
                'No model, gradient, loss-scaling, reference freezing, mask or checkpoint reload was tested.'])
    output = Path(a.output)
    if output.exists():
        raise FileExistsError('Refuse to replace verification evidence')
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(status=result['status'], negative_cases=len(cases), counts=result['counts'])))


if __name__ == '__main__':
    main()
