"""Verify finite large-model archive, preserve all failed gates and Git byte hashes."""
import argparse, hashlib, json, subprocess
from pathlib import Path
from audit_large_model_evidence import audit


def verify(root, verify_git=False):
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    original = read(root / 'attempt0/download-verification.json')
    for entry in original:
        assert hashlib.sha256((root / 'attempt0' / entry['file']).read_bytes()).hexdigest() == entry['sha256']
    models = [audit(root / folder, short) for folder, short in [('llama31-8b', 'llama31_8b'), ('llama2-7b', 'llama2_7b')]]
    queue = read(root / 'llama2-7b/status.json')
    assert queue['phase'] == 'DONE' and len(queue['stages']) == 14 and all(s['returncode'] == 0 for s in queue['stages'])
    assert set(queue['models']) == {'Llama-3.1-8B-Instruct', 'Llama-2-7b-chat-hf'}
    launch = read(root / 'llama2-7b/launch.json')
    for model in ('llama31-8b', 'llama2-7b'):
        for name, expected in launch['helpers_sha256'].items():
            assert hashlib.sha256((root / model / 'as-run' / name).read_bytes()).hexdigest() == expected
    assert hashlib.sha256((root / 'llama2-7b/as-run/gradient_audit.py').read_bytes()).hexdigest() == '2d13942540ad74d21804bf06c6e40219f675a9b0605655f9fac26ad18de0d3fb'
    recovery = read(root / 'llama2-7b/recovery-evidence/model-validation-20261006-r1/metadata-recovery-preflight.json')
    frozen = read(root / 'llama31-8b/llama31_8b-tofu-frozen-40.json')
    assert recovery['sha256'] == frozen['sha256'] == launch['frozen_batch_preserved_sha256']
    assert recovery['indices'] == frozen['indices'] and recovery['effective_tokens'] == frozen['effective_tokens']
    count = len(original) + sum(len(read(root / folder / 'evidence-manifest.json')) for folder in ['llama31-8b', 'llama2-7b'])
    report = {'phase': 'DONE', 'raw_manifest_entries_sha_verified': count, 'queue_stage_count': 14,
              'models': models, 'matched_ds_scaling_gates_passed': 4, 'original_autograd_gates_failed': 4,
              'limits': ['Four-update smoke only; no ten-epoch matrix or safety Judge.',
                         'Same DS explicit mean isolates scaling; original autograd comparisons remain failed.',
                         'Measured native full/tail scalar divisors 8/2; no legacy boundary or loss patch.',
                         'Public TOFU full revisions fixed; historical September weight identity not established.',
                         'Container Torch/bitsandbytes and evaluator float casts differ from upstream defaults.',
                         'Remote CPU checkpoint hashes/headers verified; large weights, pt and raw vectors remain remote.']}
    report_path = root / 'final-audit-summary.json'
    if not verify_git:
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        publication = {}
        for path in sorted(root.rglob('*')):
            if path.is_file() and 'raw' not in path.relative_to(root).parts and path.name != 'publication-manifest.json':
                content = path.read_bytes()
                publication[path.relative_to(root).as_posix()] = {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
        (root / 'publication-manifest.json').write_text(json.dumps(publication, indent=2), encoding='utf-8')
    else:
        publication = read(root / 'publication-manifest.json')
        repo = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip())
        for name, meta in publication.items():
            path = root / name
            content = path.read_bytes()
            assert len(content) == meta['bytes'] and hashlib.sha256(content).hexdigest() == meta['sha256']
            blob = subprocess.check_output(['git', 'show', ':' + path.resolve().relative_to(repo).as_posix()])
            assert hashlib.sha256(blob).hexdigest() == meta['sha256'], name
    return {'phase': 'DONE', 'raw_sha_verified': count, 'published_files_sha_verified': len(publication),
            'git_index_bytes_verified': verify_git, 'models': [m['model'] for m in models],
            'matched_ds_scaling_gates_passed': 4, 'original_autograd_gates_failed': 4}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--verify-git', action='store_true')
    a = p.parse_args()
    print(json.dumps(verify(a.root, a.verify_git), indent=2))
