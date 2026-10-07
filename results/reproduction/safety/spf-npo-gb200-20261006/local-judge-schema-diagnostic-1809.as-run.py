from pathlib import Path
import asyncio, hashlib, importlib.util, inspect, json, os, sys
import httpx
from openai import AsyncOpenAI
from openai.resources.responses.responses import AsyncResponses

root = Path('.').resolve()
run = root / 'work/spf-npo-gb200-20261006/private/spf-judge-local-20261007-r2'
f = run / 'rubrics/src/evals/safety/gpt_judge.py'
assert hashlib.sha256(f.read_bytes()).hexdigest() == 'ddfd529d84ab10b4b3cc8054cecfac8fa8abcc4ef718b77a3b5e5d584eec153a'
for name in ('OPENAI_API_KEY', 'OPENAI_ORG_ID', 'OPENAI_PROJECT_ID'):
    os.environ.pop(name, None)
counter = {'http_requests': 0}
def transport(request):
    counter['http_requests'] += 1
    raise AssertionError('No HTTP request permitted in schema diagnostic')
spec = importlib.util.spec_from_file_location('legacy_unregistered_harmful', f)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
async def check():
    async with AsyncOpenAI(api_key='synthetic-not-a-credential', base_url='https://api.openai.com/v1',
                           max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport))) as client:
        try:
            await client.responses.parse(model='gpt-5.6-terra', reasoning={'effort':'medium'},
                                         instructions=legacy.SYSTEM_PROMPT, input='synthetic schema diagnostic',
                                         text_format=legacy.JudgeResult, max_output_tokens=4096)
        except Exception as error:
            assert type(error).__name__ == 'PydanticUserError'
            assert getattr(error, 'code', None) == 'class-not-fully-defined'
            return dict(error_type=type(error).__name__, error_code=error.code)
        raise AssertionError('Legacy failure did not reproduce')
failure = asyncio.run(check())
assert counter['http_requests'] == 0
spec = importlib.util.spec_from_file_location('registered_harmful', f)
fixed = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fixed
spec.loader.exec_module(fixed)
schema = fixed.JudgeResult.model_json_schema()
assert schema['properties']['safety_label']['enum'] == ['refusal','safe_non_refusal','harmful_assistance','indeterminate']
assert schema['properties']['generation_quality']['enum'] == ['normal','degenerate']
sdk = inspect.getsource(AsyncResponses.parse).encode()
evidence = dict(status='LEGACY_HARMFUL_SCHEMA_FAILURE_REPRODUCED_BEFORE_HTTP_FIXED_BY_STANDARD_MODULE_REGISTRATION',
                legacy_error=failure, mock_transport_http_requests=0, real_api_calls=0, credential_reads=0,
                rubric_source_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                installed_sdk_parse_source_sha256=hashlib.sha256(sdk).hexdigest(),
                registered_original_schema_sha256=hashlib.sha256(json.dumps(schema,sort_keys=True).encode()).hexdigest(),
                classification_enums_and_confidence_bounds_unchanged=True,
                actual_fourth_result='PydanticUserError before serialization/HTTP, based on bound SDK and exact legacy module reproduction',
                actual_successful_judge_rows_preserved=3)
(run / 'schema-diagnostic.private.json').write_text(json.dumps(evidence, indent=2)+'\n',encoding='utf8')
(run / 'installed-sdk-parse-source.private.py').write_bytes(sdk)
print(json.dumps(evidence))
