"""Audit completed controls and export small evidence files, excluding weights."""
import base64
import datetime
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

import numpy as np
from scipy.stats import hmean, ks_2samp

root = Path('/data/npo-gb200-20261004')
assets = json.loads((root / 'assets.json').read_text())
retain_path = root / 'hf/hub/datasets--open-unlearning--eval/snapshots' / assets['eval_revision'] / 'tofu_Llama-2-7b-chat-hf_retain95/TOFU_EVAL.json'
retain = json.loads(retain_path.read_text())
assert hashlib.sha256(retain_path.read_bytes()).hexdigest() == assets['retain_sha256']
components = ['retain_Q_A_Prob', 'retain_Q_A_ROUGE', 'retain_Truth_Ratio',
              'ra_Q_A_Prob_normalised', 'ra_Q_A_ROUGE', 'ra_Truth_Ratio',
              'wf_Q_A_Prob_normalised', 'wf_Q_A_ROUGE', 'wf_Truth_Ratio']
report = {'captured_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'assets': assets, 'controls': [], 'weight_manifests': {}}
files = [root / 'assets.json', root / 'preflight.json', retain_path]
for name in ('gb200_1gpu_zero3_flash_attention_2_s0_main', 'gb200_1gpu_zeronone_flash_attention_2_s0_main'):
    run = root / 'runs' / name
    status = json.loads((run / 'status.json').read_text())
    assert status['phase'] == 'DONE' and not status['smoke_only']
    raw = json.loads((run / 'eval/TOFU_EVAL.json').read_text())
    summary = json.loads((run / 'eval/TOFU_SUMMARY.json').read_text())
    score = np.array([v['score'] for v in raw['forget_truth_ratio']['value_by_index'].values()])
    reference = np.array([v['score'] for v in retain['forget_truth_ratio']['value_by_index'].values()])
    assert len(score) == len(reference) == 200
    ks = ks_2samp(score, reference)
    values = {key: float(raw[key]['agg_value']) for key in components}
    recomputed = {'forget_quality': float(ks.pvalue), 'model_utility': float(hmean(list(values.values()))),
                  'forget_truth_ratio': float(np.minimum(score, 1 / (score + 1e-10)).mean())}
    deltas = {key: abs(recomputed[key] - summary[key]) for key in recomputed}
    assert all(np.isfinite(list(recomputed.values())))
    assert max(deltas.values()) < 1e-12, deltas
    runtime = json.loads((run / 'checkpoint/runtime_final_rank0.json').read_text())
    trainer = json.loads((run / 'checkpoint/trainer_state.json').read_text())
    assert trainer['global_step'] == runtime['trainer_global_step'] == 60
    assert runtime['microsteps'] == 432 and trainer['epoch'] == 8.64
    if status['stage'] == '3':
        assert runtime['engine_global_steps'] == 54 and runtime['engine_micro_steps'] == 432
    index = json.loads((run / 'checkpoint/model.safetensors.index.json').read_text())
    shards = sorted(set(index['weight_map'].values()))
    manifest = []
    for shard in shards:
        path = run / 'checkpoint' / shard
        assert path.is_file() and path.stat().st_size > 100_000_000
        manifest.append({'path': str(path), 'bytes': path.stat().st_size})
    report['weight_manifests'][name] = manifest
    report['controls'].append({'run': name, 'summary': summary, 'independent_recomputed': recomputed,
                               'absolute_deltas': deltas, 'ks_D': float(ks.statistic),
                               'samples': [len(score), len(reference)], 'runtime_final': runtime,
                               'utility_components': values,
                               'training_last_log': trainer['log_history'][-1],
                               'start_utc': datetime.datetime.fromtimestamp(status['start'], datetime.timezone.utc).isoformat(),
                               'finish_utc': datetime.datetime.fromtimestamp(status['updated'], datetime.timezone.utc).isoformat()})
    files += [path for path in run.rglob('*') if path.is_file() and path.suffix in ('.json', '.yaml', '.log')]
report_path = root / 'independent_audit.json'
report_path.write_text(json.dumps(report, indent=2))
files.append(report_path)
status_text = subprocess.check_output(['python', str(root / 'repo/scripts/reproduction/npo/gb200/collect_status.py')], text=True)
status_path = root / 'status-final.json'
status_path.write_text(status_text)
files.append(status_path)
buffer = io.BytesIO()
manifest = {}
with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(set(files)):
        if path == retain_path:
            relative = 'reference/retain95_TOFU_EVAL.json'
        else:
            relative = str(path.relative_to(root))
        content = path.read_bytes()
        archive.writestr(relative, content)
        manifest[relative] = {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
    archive.writestr('evidence_manifest.json', json.dumps(manifest, indent=2))
print(base64.b64encode(buffer.getvalue()).decode())
