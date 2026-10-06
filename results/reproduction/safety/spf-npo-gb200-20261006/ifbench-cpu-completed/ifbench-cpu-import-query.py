import hashlib,importlib,importlib.metadata as md,json,os,pathlib,platform,sys
runtime=pathlib.Path(sys.argv[1]);audit=json.loads((runtime/'audit.json').read_text())
assert audit['revision']=='1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d'
for name,entry in audit['files'].items():
    assert hashlib.sha256((runtime/name).read_bytes()).hexdigest()==entry['sha256'],name
os.environ['NLTK_DATA']=str(runtime/'nltk_data')
import nltk
nltk.data.path.insert(0,os.environ['NLTK_DATA'])
def no_download(*a,**k):raise RuntimeError('Offline scorer forbids NLTK download')
nltk.download=no_download
sys.path.insert(0,str(runtime))
roots=['absl','langdetect','nltk','immutabledict','emoji','syllapy','evaluation_lib']
for name in roots:importlib.import_module(name)
pins=pathlib.Path(sys.argv[2]).read_text().splitlines();versions={}
for line in pins:
    if '==' in line and not line.startswith('#'):
        name,expected=line.split('==');actual=md.version(name);assert actual==expected,(name,actual,expected);versions[name]=actual
record=dict(status='IMPORT_PASS_PENDING_SCORE_EQUIVALENCE',python=sys.version,platform=platform.machine(),
            installed_versions=versions,import_roots=roots,source_revision=audit['revision'],
            historical_scoring_repeated=False,target_scoring_executed=False,score_equivalence_verified=False,
            gpu_runtime_validated=False,external_judge_executed=False)
pathlib.Path(sys.argv[3]).write_text(json.dumps(record,indent=2));print(json.dumps(record))
