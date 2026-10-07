from pathlib import Path
from collections import Counter, defaultdict
import base64, hashlib, io, json, tarfile

root = Path('.').resolve()
p = root/'work/spf-npo-gb200-20261006/private'
sha = lambda b: hashlib.sha256(b).hexdigest()
j = lambda f: json.loads(f.read_bytes())
write = lambda f, d: f.write_bytes((json.dumps(d, ensure_ascii=False, indent=2)+'\n').encode())
response = j(p/'native-reload-launch-failure-export-1936.json')
assert response['status'] == 'IMMUTABLE_NATIVE_RELOAD_LAUNCH_FAILURE_EXPORT'
raw = base64.b64decode(response['archive_base64'], validate=True)
assert sha(raw) == response['archive_sha256']
out = p/'native-reload-launch-failure-20261007-1936'
out.mkdir(exist_ok=False)
(out/'archive.tar.gz').write_bytes(raw)
records = response['files']
with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as tar:
    members = tar.getmembers()
    assert {m.name for m in members} == {r['name'] for r in records}
    for index, record in enumerate(records):
        member = tar.getmember(record['name'])
        assert member.isfile() and not member.issym() and not member.islnk()
        data = tar.extractfile(member).read()
        assert len(data) == record['size'] and sha(data) == record['sha256']
        name = str(index).zfill(2)+'-'+Path(member.name).name
        (out/name).write_bytes(data)
        record['local_name'] = name
write(out/'manifest.private.json', dict(archive_sha256=sha(raw), files=records,
      source_response_sha256=sha((p/'native-reload-launch-failure-export-1936.json').read_bytes()),
      byte_verification=True, weights_read=False, model_evaluation_executed=False,
      training_executed=False, progress_snapshot=False))

# Derive complete enum counts and nullable-score denominators from sealed labels.
a = p/'judge-result-audit-20261007-1936'
audit = j(a/'audit.private.json')
labels_raw = (a/'verified-labels.private.json').read_bytes()
assert sha(labels_raw) == audit['verified_labels_sha256']
labels = json.loads(labels_raw)
enums = defaultdict(Counter)
scores = defaultdict(list)
for r in labels:
    for field, value in r['judgment'].items():
        if field in ('generation_quality', 'safety_label', 'response_behavior', 'quality'):
            enums[r['kind']+':'+field][value] += 1
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            scores[r['kind']+':'+field].append(value)
stats = dict(status='DERIVED_DESCRIPTIVE_STATISTICS_OF_VERIFIED_765_SUBSET',
      parent_audit_sha256=sha((a/'audit.private.json').read_bytes()),
      verified_labels_sha256=sha(labels_raw), count=len(labels),
      label_counts={k:dict(v) for k,v in enums.items()},
      scores={k:dict(non_null_count=len(v), total=sum(v), mean=sum(v)/len(v)) for k,v in scores.items()},
      harmful_assistance_fraction=135/289,
      full_792_results=False, paired_safety_gate=False, human_adjudication_complete=False,
      new_api_calls=0, credential_reads=0, original_records_unchanged=True)
write(a/'descriptive-statistics.private.json', stats)
print(json.dumps(dict(archive_sha256=sha(raw), files=len(records),
      native_failure_sources=[dict(local_name=r['local_name'],sha256=r['sha256']) for r in records],
      subset_counts=stats['label_counts'], scores=stats['scores']), ensure_ascii=False))
