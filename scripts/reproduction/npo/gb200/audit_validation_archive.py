"""Offline verification of archived hashes and the three native NPO cells."""
import argparse
import hashlib
import json
from pathlib import Path

from audit_native_evidence import audit


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def verify(root):
    mapping = read(root / 'publication-path-map.json')
    omissions = read(root / 'publication-omissions.json')
    manifests = [
        'h100/download-verification.json',
        'gb200/attempt0/download-verification.json',
        'gb200/attempt1/download-verification-0948.json',
        'gb200/native-forget01/download-verification-1003.json',
        'gb200/native-forget05/download-verification-1018.json',
        'gb200/native-forget10/download-verification-1033.json',
        'gb200/as-run/download-verification.json',
    ]
    integrity = []
    for name in manifests:
        manifest = root / name
        omitted = 0
        for item in read(manifest):
            original = (manifest.parent / item['file']).relative_to(root).as_posix()
            if original in omissions:
                assert item['sha256'] == omissions[original]['sha256']
                omitted += 1
                continue
            published = mapping.get(original, original)
            data = (root / published).read_bytes()
            assert hashlib.sha256(data).hexdigest() == item['sha256'], published
        integrity.append({'manifest': name, 'verified_entries': len(read(manifest)) - omitted, 'explicit_large_asset_omissions': omitted})

    queue = read(root / 'gb200/native-forget10/queue-evidence/status.json')
    assert queue['phase'] == 'DONE'
    assert len(queue['stages']) == 4 and all(x['returncode'] == 0 for x in queue['stages'])
    h100 = read(root / 'h100/audit-summary.json')
    tiny = read(root / 'gb200/attempt0/tiny-audit-summary.json')
    real = read(root / 'gb200/attempt1/real1b-gradient-audit-summary.json')
    assert h100['phase'] == 'DONE' and len(h100['cases']) == 8
    assert tiny['verified_cases'] == 20 and real['verified_cases'] == 16
    assert all(x['rank_gradient_hashes_identical'] for x in h100['cases'])
    assert real['matched_initial_old_new_parameters_and_inputs']
    for item in tiny['cases']:
        raw = read(root / ('gb200/attempt0/' + item['case'] + '.json'))
        expected = 1 if item['case'].startswith('new_') else raw['window']
        assert raw['world_size'] == 1 and raw['clipping'] == raw['optimizer_lr'] == 0
        assert abs(item['projected_scale'] - expected) < 1e-5
    for item in real['cases']:
        raw = read(root / ('gb200/attempt1/' + item['case'] + '.json'))
        expected = 1 if item['case'].startswith('new_') else raw['window']
        assert raw['initial_parameter_sha256'] == raw['final_parameter_sha256']
        assert raw['engine_updates'] == raw['ds_reference_updates'] == 1
        assert raw['engine_total_updates'] == 2
        assert item['projected_scale'] == expected and item['cosine'] == 1
        assert item['relative_error_after_dividing_projected_scale'] == 0
        assert not item['autograd_reference_gate_passed']

    native = []
    prior = read(root.parent / 'gb200/followup-independent-audit-20261005.json')
    for split in ['forget01', 'forget05', 'forget10']:
        cell = root / ('gb200/native-' + split)
        ref = cell / ('retain99-reference.json' if split == 'forget01' else 'retain-reference.json')
        proof = cell / ('checkpoint-forget01-verification.json' if split == 'forget01' else 'checkpoint-verification.json')
        result = audit(cell, ref, proof)
        corrected = next(x for x in prior['followup']['cells'] if x['model'] == result['model'] and x['split'] == split and x['seed'] == 0 and x['corrected'])
        result['original_seed0'] = corrected['baseline_seed0']
        result['update_only_corrected_seed0'] = corrected['summary']
        result['delta_update_only'] = {k: v - corrected['summary'][k] for k, v in result['summary'].items()}
        native.append(result)
    report = {
        'finite_validation_phase': 'DONE',
        'upstream_commit': '17cbbc87192e6934deb92875c359c91bbd837fb4',
        'hash_verification': integrity,
        'h100_case_count': 8,
        'gb200_tiny_case_count': 20,
        'gb200_real1b_case_count': 16,
        'native_seed0': native,
        'unresolved_autograd_reference_cases': 16,
        'limits': [
            'Same-DS explicit mean isolates scaling; real1B autograd comparison remains failed.',
            'Gradient ratios are preclip SGD lr0 measurements, not Adam update or LR ratios.',
            'Retained GB200 container Torch and bitsandbytes differ from upstream defaults.',
            'Two evaluator float casts only repair bf16 NumPy serialization.',
            'Single seed, multiple stack changes; no claim of paper-wide reproduction or causal score attribution.',
            'Historical H100/Ada/safety and update-only corrected runs keep their original classification.',
        ],
    }
    (root / 'final-audit-summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[4] / 'results/reproduction/npo/scaling-validation-20261006')
    result = verify(parser.parse_args().root)
    print(json.dumps({'phase': result['finite_validation_phase'], 'manifest_entries': sum(x['verified_entries'] for x in result['hash_verification']), 'native': [x['split'] for x in result['native_seed0']], 'unresolved_autograd_cases': result['unresolved_autograd_reference_cases']}))
