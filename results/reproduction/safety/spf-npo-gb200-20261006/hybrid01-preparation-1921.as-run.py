from pathlib import Path
import ast,hashlib,json
root=Path('.').resolve();work=root/'work/spf-npo-gb200-20261006'
campaign=work/'private/spf-safety-search-20261010';folder=root/'scripts/reproduction/safety/gb200'
old=campaign/'candidate-04-hybrid01-preparation-20261011-1921-r2'
out=campaign/'candidate-04-hybrid01-preparation-20261011-1921-r3';out.mkdir()
sha=lambda raw:hashlib.sha256(raw).hexdigest()
dump=lambda value:(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()
queue=folder/'run_spf_hybrid01_lr5e6_full_queue.py';raw=queue.read_bytes()
assert raw.count(b'/data/spf-hybrid01-full-20261010-r2/validation-code/audit_spf_target.py')==1
(old/'queue-before-auditor-path-review.py').write_bytes(raw)
updated=raw.replace(b'/data/spf-hybrid01-full-20261010-r2/validation-code/audit_spf_target.py',b'/data/spf-beta05-full-20261010-r2/validation-code/audit_spf_target.py')
queue.write_bytes(updated)
records={}
evidence=json.loads((old/'evidence.private.json').read_bytes())
for name in evidence['sources']:
    raw=(folder/name).read_bytes();compile(raw,name,'exec');(out/name).write_bytes(raw)
    records[name]=dict(sha256=sha(raw),size=len(raw))
evidence.update(sources=records,auditor_original_path_preserved=True,
    r2_completed_schema_negatives_reused=True,review_changed_only_auditor_path_literal=True)
(out/'evidence.private.json').write_bytes(dump(evidence))
(out/'candidate-plan-core.private.json').write_bytes((old/'candidate-plan-core.private.json').read_bytes())
(out/'preparation.as-run.py').write_bytes(Path(__file__).read_bytes())
source=(work/'prepare_spf_beta05_lr3e6_deployment_20261011.py').read_text(encoding='utf-8')
for old_text,new_text in [
    ('candidate-03-beta05-lr3e6-preparation-20261011-1118','candidate-04-hybrid01-preparation-20261011-1921-r3'),
    ('beta05_lr3e6','hybrid01_lr5e6'),('beta05-lr3e6','hybrid01-lr5e6'),
    ('/data/spf-beta05-lr5e6-full-20261010-r1','/data/spf-beta05-lr3e6-full-20261011-r1'),
    ('spf-beta05-lr5e6-full-s0-audit.json','spf-beta05-lr3e6-full-s0-audit.json'),
    ('b42e934ac66d38a2918bd45302b362176653f49b1003e3f44d310f6f822118f5','839156fd5c6d0e28847991188a95061d513727cf012ec912465697ca9450c181'),
    ('deploy_beta05_lr3e6_','deploy_hybrid01_lr5e6_')]:source=source.replace(old_text,new_text)
old_line="names=['run_spf_hybrid01_lr5e6_full_queue.py','run_spf_hybrid01_lr5e6_full.py','spf_hybrid01_lr5e6_development_executor.py']"
assert source.count(old_line)==1
source=source.replace(old_line,old_line[:-1]+",'spf_anchor_hybrid.py','verify_spf_anchor_hybrid.py']")
source=source.replace("scope='New independent full SPF candidate from original Meta M0; beta1=.5 and lower learning rate3e-6; train then same six development stages.'",
    "scope='New independent SPF plus training-anchor-loss hybrid from original MetaM0; beta1=.5, LR5e-6, lambda.1. New CPU check and real8B two-update/reload acceptance must pass before full625 train and six development stages.'")
source=source.replace('learning rate3e-6','learning rate5e-6')
source=source.replace('10758,10761,10829,11297,11577):','10758,10761,10829,11297,11577,12367,12370,12438):')
source=source.replace("assert sha(Path('/data/spf-npo-20261006-r4/frozen.json').read_bytes())", "import os\nvfs=os.statvfs('/data');assert vfs.f_bavail*vfs.f_frsize>=160*1024**3\nassert sha(Path('/data/spf-npo-20261006-r4/frozen.json').read_bytes())")
compile(source,'new_hybrid_deployment','exec')
with (work/'prepare_spf_hybrid01_deployment_20261011.py').open('xb') as stream:stream.write(source.encode())
snapshot=(work/'snapshot_spf_search_lr3e6_20261011.py').read_text(encoding='utf-8')
snapshot=snapshot.replace("previous=Path('/data/spf-beta05-lr5e6-full-20261010-r1')", "previous=Path('/data/spf-beta05-lr3e6-full-20261011-r1')")
snapshot=snapshot.replace("candidate=Path('/data/spf-beta05-lr3e6-full-20261011-r1')", "candidate=Path('/data/spf-hybrid01-lr5e6-full-20261011-r1')")
start=snapshot.index('    beta05_lr5e6_candidate=branch(')
snapshot=snapshot[:start]+"""    beta05_lr3e6_candidate=branch(previous,'spf-beta05-lr3e6-full-s0','beta05-lr3e6-full-launch-20261011.json',
        ['run_spf_beta05_lr3e6_full.py','run_spf_beta05_lr3e6_full_queue.py','spf_beta05_lr3e6_development_executor.py','plan.private.json','bound-snapshot.private.json']),
    hybrid01_lr5e6_candidate=branch(candidate,'spf-hybrid01-lr5e6-full-s0','hybrid01-lr5e6-full-launch-20261011.json',
        ['run_spf_hybrid01_lr5e6_full.py','run_spf_hybrid01_lr5e6_full_queue.py','spf_hybrid01_lr5e6_development_executor.py',
         'spf_anchor_hybrid.py','verify_spf_anchor_hybrid.py','plan.private.json','bound-snapshot.private.json']),
    hybrid_engineering_cpu_proof=record(candidate/'hybrid-cpu-proof.private.json'),
    hybrid_engineering_proof=record(candidate/'hybrid-engineering-proof.private.json'),
    hybrid_full_acceptance=record(candidate/'hybrid-full-acceptance.private.json'),
    completed_train_reload_development_raw_or_weight_bytes_not_repeated=True)))
"""
compile(snapshot,'new_hybrid_snapshot','exec')
with (work/'snapshot_spf_search_hybrid01_20261011.py').open('xb') as stream:stream.write(snapshot.encode())
print(json.dumps(dict(status='HYBRID_DEPLOYMENT_PRODUCER_AND_MONITOR_PREPARED',sources=records,
    snapshot_source_sha256=sha(snapshot.encode()),runai_queries=0,remote_mutations=0)))
