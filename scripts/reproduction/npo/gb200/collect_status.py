"""Read-only, compact JSON status suitable for RunAI exec and local archiving."""
import datetime
import json
from pathlib import Path
import re

root = Path('/data/npo-gb200-20261004')
result = {'captured_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'runs': []}
for name in ('assets.json', 'preflight.json'):
    path = root / name
    if path.exists():
        result[name.removesuffix('.json')] = json.loads(path.read_text())
for path in sorted((root / 'runs').glob('*/status.json')):
    entry = json.loads(path.read_text())
    trainer = entry.pop('trainer_state', {})
    entry['trainer'] = {k: trainer.get(k) for k in ('global_step', 'max_steps', 'epoch')}
    entry['path'] = str(path.parent)
    entry['runtime_final'] = [json.loads(p.read_text()) for p in sorted((path.parent/'checkpoint').glob('runtime_final_rank*.json'))]
    for filename in ('training.log', 'evaluating.log'):
        log = path.parent / filename
        if log.exists():
            lines = re.split(r'[\r\n]', log.read_text(errors='replace'))
            useful = [line for line in lines if re.search(r'\d+/\d+ \[|grad_norm|Result for metric|Error:|Exception:', line) and 'NPO_RUNTIME' not in line]
            entry[filename] = useful[-3:]
    summary = path.parent / 'eval/TOFU_SUMMARY.json'
    if summary.exists():
        entry['summary'] = json.loads(summary.read_text())
    audit = path.parent / 'auditing.log'
    if audit.exists():
        try:
            entry['audit'] = json.loads(audit.read_text())
        except json.JSONDecodeError:
            entry['audit_log'] = audit.read_text()[-1000:]
    result['runs'].append(entry)
print(json.dumps(result, indent=2))
