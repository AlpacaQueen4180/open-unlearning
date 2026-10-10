from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

task = Path('/data/spf-beta05-lr5e6-full-20261010-r1')
validation = task / 'validation-code'
transfer = validation / 'transfer-20261010-1905/59fb691502f88addeaa48eb50d4b12558fa9cfcc9f247477ccd5372178efc141'
names = ['run_spf_beta05_lr5e6_full_queue.py', 'run_spf_beta05_lr5e6_full.py',
    'spf_beta05_lr5e6_development_executor.py', 'plan.private.json', 'bound-snapshot.private.json',
    'beta05-lr5e6-full-launch-20261010.json', 'beta05-lr5e6-full-launch-20261010.raw.log']
def record(path):
    return dict(size=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()) if path.is_file() else None
print(json.dumps(dict(checked_utc=datetime.now(timezone.utc).isoformat(),
    purpose='Deployment artifact check after preflight transport EOF; not another workload progress snapshot',
    new_root_exists=task.exists(), files={name: record(validation / name) for name in names},
    chunks={str(i): record(transfer / str(i)) for i in range(7)},
    boundaries={name: (task / name).exists() for name in ('status.json', 'spf-beta05-lr5e6-full-s0',
        'candidate-development', 'optimizer-binding.private.json')},
    gpu_queries=0, process_queries=0, model_reads=0, remote_mutations=0)))
