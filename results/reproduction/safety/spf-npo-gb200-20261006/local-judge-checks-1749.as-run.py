from pathlib import Path
import asyncio, copy, importlib.util, json, os, tempfile
from types import SimpleNamespace

root = Path('.').resolve()
run = root / 'work/spf-npo-gb200-20261006/private/spf-judge-local-20261007-r1'
spec = importlib.util.spec_from_file_location('local_judge_test', run / 'run_spf_development_judge_local.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
plan = json.loads((run / 'plan.private.json').read_bytes())
m.validate_profile(plan)
rejected = []
for field, value in [('api_base_url', 'https://example.invalid/v1'), ('retries', 1), ('new_calls', 793),
                     ('credential_policy', 'existing_env_file'), ('model', 'other-model')]:
    negative = copy.deepcopy(plan)
    negative[field] = value
    try:
        m.validate_profile(negative)
    except ValueError:
        rejected.append(field)
    else:
        raise AssertionError(field)
args = SimpleNamespace(run=str(run), source_sha256=m.digest((run / 'run_spf_development_judge_local.py').read_bytes()),
                       plan_sha256=m.digest((run / 'plan.private.json').read_bytes()), preflight=False)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.pop('SPF_JUDGE_NEW_KEY_INPUT', None)
try:
    asyncio.run(m.execute(args))
except ValueError:
    rejected.append('missing_new_masked_key')
else:
    raise AssertionError('missing key accepted')
assert not (run / 'execution.lock').exists()

class MockError(Exception):
    code = 'invalid_api_key'
    body = {'message': 'synthetic_secret_that_must_never_be_serialized'}

class FakeResponses:
    def __init__(self): self.calls = 0
    async def parse(self, **kwargs):
        self.calls += 1
        raise MockError('synthetic_secret_that_must_never_be_serialized')

with tempfile.TemporaryDirectory(prefix='spf-local-judge-check-') as tmp:
    folder = Path(tmp)
    (folder / 'raw-responses').mkdir()
    fake = SimpleNamespace(responses=FakeResponses())
    state = dict(status='RUNNING', attempted=0, completed=0)
    rows = [dict(id=str(i), blind_id=str(i), kind='harmful',
                 messages=[dict(content='synthetic question'), dict(content='synthetic response')]) for i in range(2)]
    harmful = SimpleNamespace(is_policy_code=lambda code: False)
    asyncio.run(m.judge_rows(fake, folder, plan, rows, {'harmful': ('synthetic rubric', object, 'fixture')},
                            {'harmful': 'fixture'}, harmful, state))
    assert state['status'] == 'STOPPED_ON_ERROR_NO_RETRY'
    assert fake.responses.calls == state['attempted'] == state['completed'] == 1
    assert 'synthetic_secret' not in (folder / 'judge-results.jsonl').read_text()
    assert 'synthetic_secret' not in (folder / 'job.json').read_text()
evidence = dict(status='LOCAL_JUDGE_NEW_GUARDS_AND_ERROR_STOP_MOCK_PASS', negatives_rejected=rejected,
                mock_requests=1, real_api_calls=0, credential_reads=0, model_generation=0,
                error_message_and_body_not_serialized=True, second_mock_row_not_attempted=True)
(run / 'local-checks.private.json').write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf8')
print(json.dumps(evidence))
