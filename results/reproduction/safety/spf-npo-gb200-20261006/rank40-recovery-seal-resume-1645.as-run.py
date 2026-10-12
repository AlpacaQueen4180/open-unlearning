"""All8 chunks verified, sealing absent: newtag only seal and single launch."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
work=Path('work/spf-npo-gb200-20261006');campaign=work/'private/spf-safety-search-20261010';old=campaign/'rank40-cpu-recovery-deployment-20261012-1645';r2=campaign/'rank40-cpu-recovery-deployment-20261012-1645-r2';out=campaign/'rank40-cpu-recovery-deployment-20261012-1645-r3'
sha=lambda raw:hashlib.sha256(raw).hexdigest();dump=lambda v:(json.dumps(v,indent=2)+'\n').encode()
check=json.loads((campaign/'rank40-recovery-artifact-check-1645-r2.private.json').read_bytes());expected=json.loads((campaign/'rank40-recovery-expected-chunks-1645.private.json').read_bytes())
assert check['chunks']==expected and all(v is None for v in check['destinations'].values())
assert check['launch'] is check['launch_raw'] is None and check['journal_exists'] is check['engineering_checkpoint_exists'] is check['full_checkpoint_exists'] is False
snapshot=json.loads((campaign/'heartbeat-20261012-1645-rank40.private.json').read_bytes());assert 0<=(datetime.now(timezone.utc)-datetime.fromisoformat(snapshot['snapshot_utc'])).total_seconds()<=600
for name in ['chunk-'+str(i) for i in range(3,8)]:assert json.loads((r2/(name+'.receipt.json')).read_bytes())['returncode']==0
receipt=json.loads((r2/'seal.receipt.json').read_bytes());raw=(r2/'seal.cli.txt').read_bytes();assert receipt['returncode']==1 and b'EOF' in raw
failure=dict(status='R2_SEAL_CLUSTER_STATUS_EOF_NO_BLIND_RETRY',returncode=1,cli_size=len(raw),cli_sha256=sha(raw),asrun_sha256=sha((r2/'seal.as-run.py').read_bytes()),receipt_sha256=sha((r2/'seal.receipt.json').read_bytes()),capture_format='native strings joined LF UTF8, not original subprocess streambytes',launch_stage_reached=False)
with (campaign/'rank40-recovery-r2-seal-failure-1645.private.json').open('xb') as f:f.write(dump(failure))
commands=[v for v in json.loads((old/'commands.private.json').read_bytes()) if v['name'] in ('seal','launch-once')];assert [v['name'] for v in commands]==['seal','launch-once']
out.mkdir();(out/'commands.private.json').write_bytes(dump(commands))
for item in commands:(out/(item['name']+'.as-run.py')).write_bytes((old/(item['name']+'.as-run.py')).read_bytes())
metadata=json.loads((old/'bundle-metadata.private.json').read_bytes());metadata.update(explicit_new_tag=True,original_deployment_path=str(old),prior_r2_path=str(r2),artifact_check_sha256=sha((campaign/'rank40-recovery-artifact-check-1645-r2.private.json').read_bytes()),completed_chunks_reused=list(range(8)),only_verified_missing_stages=['seal','launch-once'],bundle_reused=True,blind_relaunches=0)
(out/'bundle-metadata.private.json').write_bytes(dump(metadata));(out/'preparation.as-run.py').write_bytes(Path(__file__).read_bytes())
deployment=(work/'deploy_rank40_cpu_recovery_20261012-1645.ps1').read_text(encoding='utf8').replace('rank40-cpu-recovery-deployment-20261012-1645','rank40-cpu-recovery-deployment-20261012-1645-r3')
with (work/'deploy_rank40_cpu_recovery_20261012-1645-r3.ps1').open('xb') as f:f.write(deployment.encode())
(out/'deployment.as-run.ps1').write_bytes(deployment.encode())
print(json.dumps(dict(status='VERIFIED8CHUNKS_REUSED_SEAL_LAUNCH_ONLY_NEW_TAG_PREPARED',stages=['seal','launch-once'],failure=failure)))
