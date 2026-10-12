from pathlib import Path
import base64,hashlib,io,json,tarfile
work=Path('work/spf-npo-gb200-20261006');campaign=work/'private/spf-safety-search-20261010';sha=lambda raw:hashlib.sha256(raw).hexdigest()
response=json.loads((campaign/'rank40-engineering-export-1846.private.json').read_bytes());raw=base64.b64decode(response['archive_base64'])
assert response['status']=='IMMUTABLE_NEW_RANK40_ENGINEERING_COMPLETION_EXPORT' and sha(raw)==response['archive_sha256']=='c65793ec7a88a2e537d415c3d37d53b0ed6d286759281f40ebb02e7496759862'
assert response['weights_read'] is response['progress_snapshot'] is False and response['live_training_status_or_logs_not_read'] is True
expected={v['name']:v for v in response['files']};assert len(expected)==20
out=campaign/'rank40-engineering-completed-20261012-1846';out.mkdir();(out/'archive.private.tar.gz').write_bytes(raw)
values={};records=[]
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
    assert len(archive.getmembers())==len(expected)
    for index,item in enumerate(archive.getmembers()):
        assert item.isfile() and item.name in expected and not item.name.startswith('/') and '..' not in Path(item.name).parts
        data=archive.extractfile(item).read();record=expected[item.name]
        assert len(data)==record['size'] and sha(data)==record['sha256']
        assert Path(item.name).suffix not in ('.safetensors','.bin','.pt')
        local=f'{index:03d}-'+Path(item.name).name
        with (out/local).open('xb') as f:f.write(data)
        records.append(dict(record,local_name=local));values[item.name]=data
assert set(values)==set(expected)
proof=json.loads(values['hybrid-engineering-proof.private.json']);assert proof==response['engineering_proof'] and sha(values['hybrid-engineering-proof.private.json'])=='a68b9782be482c1b6c160a1c88beafb747f8805492a6d6e70cf3b9d7de64844c'
assert proof['fresh_reload']=={'status':'pass','max_logit_error':0.0} and proof['updates']==2 and proof['spf_projection_rank']==40
rows=[json.loads(v) for v in values['engineering-spf-hybrid01-rank40-s0/trajectory.jsonl'].splitlines()][1:]
assert len(rows)==2 and [r['window_examples'] for r in rows]==[32,8]
assert all(r['spf_projection_rank']==40 and r['declared_projection_rank_verified_before_original_combine'] and r['safety_loss_coefficient']==.1 for r in rows)
assert [r['step'] for r in rows]==[1,2] and all(r['step']==r['engine_global_steps']==r['scheduler_steps'] for r in rows)
assert all(r['backward_scale_wrt_gas'] is False and r['extra_model_forwards']==r['extra_model_backwards']==0 and r['projection_before_anchor_addition'] and r['anchor_gradient_added_before_original_clip_and_step'] for r in rows)
manifest=dict(archive_sha256=sha(raw),files=records,engineering_proof=proof,completed_stage_records=response['completed_stage_records'],
 actual_rank40_and_anchor_addition_all_two_steps_verified=True,weights_read=False,engineering_rerun=False,original_cpu_proof_sha_reused_not_downloaded=True)
with (out/'manifest.private.json').open('xb') as f:f.write((json.dumps(manifest,indent=2)+'\n').encode())
(out/'export.as-run.py').write_bytes((work/'export_rank40_engineering_20261012_1846.py').read_bytes())
print(json.dumps(dict(status='NEW_ENGINEERING20_FILES_IMMUTABLE_BYTE_SHA_VERIFIED',archive_sha256=sha(raw),proof_sha256=sha(values['hybrid-engineering-proof.private.json']),updates=2,window_examples=[32,8],rank=40,reload_max_logit_error=0.0,weights_read=False)))
