"""Check existing private GB200 Meta cache against independently hashed H100 M0."""
import hashlib,json,pathlib,os
task=pathlib.Path(os.environ['SPF_TASK'])
manifest=json.loads((task/'overlay/private/m0-files.json').read_text())
assert manifest['revision']=='0e9e39f249a16976918f6564b8830bc894c89659'
cache=pathlib.Path('/data/smart-mfg/jimmy-lin/hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots')/manifest['revision']
rows=[]
for e in manifest['files']:
    p=cache/e['name'];assert p.stat().st_size==e['size']
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(16*1024**2),b''):h.update(b)
    assert h.hexdigest()==e['sha256'],e['name']
    rows.append({'name':p.name,'size':p.stat().st_size,'sha256':h.hexdigest(),'path':str(p)})
(task/'m0-assets.json').write_text(json.dumps({'status':'pass','model_id':manifest['model_id'],'revision':manifest['revision'],'h100_manifest_sha256':hashlib.sha256((task/'overlay/private/m0-files.json').read_bytes()).hexdigest(),'files':rows},indent=2))
print(json.dumps({'status':'pass','verified_files':len(rows),'bytes':sum(r['size'] for r in rows)}))
