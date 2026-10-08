from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
root = Path('.').resolve()
p = root / 'work/spf-npo-gb200-20261006/private'
base = p / 'checkpoint313469-completed-20261009'
manifest = json.loads((base / 'manifest.private.json').read_bytes())
mapping = {row['name']: row for row in manifest['files']}
sha = lambda raw: hashlib.sha256(raw).hexdigest()
def value(name):
    row = mapping[name]
    raw = (base / row['local_name']).read_bytes()
    assert len(raw) == row['size'] and sha(raw) == row['sha256']
    return json.loads(raw)
cases = {}
for step in (313, 469):
    prefix = f'checkpoint-{step}/evaluation/'
    tofu = value(prefix + 'tofu/public-summary.json')
    knowledge = value(prefix + 'knowledge/public-summary.json')
    ifbench = value(prefix + 'ifbench-scores.private.json')
    metrics = {split: dict(count=item['count'], metrics={name: item['metrics'][name]['mean'] for name in ('normalized_exact_match', 'rouge_l_f1', 'mean_log_probability')}) for split, item in tofu['aggregates'].items()}
    assert metrics['forget05']['count'] == 200 and metrics['retain95']['count'] == 3800
    assert knowledge['count'] == 1024 and ifbench['count'] == 300
    metrics.update(mmlu_aux=dict(count=1024, correct=knowledge['correct'], accuracy=knowledge['accuracy'], num_fewshot=0),
        ifbench=dict(count=300, summary=ifbench['summary']))
    cases[str(step)] = metrics
record = dict(status='CHECKPOINT313469_FREE_LEARNING_CAPABILITY_COMPLETE', derived_at_utc=datetime.now(timezone.utc).isoformat(),
    archive_sha256=manifest['archive_sha256'], immutable_file_count=len(manifest['files']), cases=cases,
    analysis_source_sha256=sha(Path(__file__).read_bytes()), new_gpu_submissions=0, generation_repeated=False,
    paid_judge_harmfulness_pending=[313,469], target_gate='insufficient_evidence')
target = p / 'checkpoint313469-free-analysis-20261009.private.json'
assert not target.exists()
target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
print(json.dumps(record))
