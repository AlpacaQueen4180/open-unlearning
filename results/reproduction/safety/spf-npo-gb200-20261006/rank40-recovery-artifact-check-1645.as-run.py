from pathlib import Path
import hashlib,json
T=Path('/data/spf-hybrid01-rank40-full-20261012-r1');V=T/'validation-code';D=V/'cpu-recovery-transfer-20261012-1645'/'bf963a90b233007f8352dee47185fa4052c2b50fd9d8675fde2e0cbe62e334f8'
def record(path):return dict(size=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) if path.is_file() else None
print(json.dumps(dict(status='READONLY_DEPLOYMENT_ARTIFACT_CHECK_NOT_PROGRESS',chunks={str(i):record(D/str(i)) for i in range(8)},
 destinations={name:record(V/name) for name in ['run_spf_hybrid01_rank40_cpu_recovery_queue.py', 'run_spf_hybrid01_rank40_cpu_recovery.py', 'verify_spf_anchor_hybrid_rank40_cpu_oracle_r2.py', 'rank40-cpu-oracle-proof-1539.private.json', 'plan-cpu-fixture-recovery-1539.private.json', 'bound-snapshot-cpu-recovery-1539.private.json']},launch=record(V/'hybrid01-rank40-cpu-recovery-launch-1539.json'),
 launch_raw=record(V/'hybrid01-rank40-cpu-recovery-launch-1539.raw.log'),journal_exists=(T/'runs/cpu-fixture-recovery-1539').exists(),
 engineering_checkpoint_exists=(T/'engineering-spf-hybrid01-rank40-s0').exists(),full_checkpoint_exists=(T/'spf-hybrid01-rank40-full-s0').exists(),
 no_gpu_or_process_progress_query=True,weights_read=False,new_gpu_submissions=0)))
