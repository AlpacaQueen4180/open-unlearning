"""Finite source-bound SPF development judging on the existing private judge host."""
import argparse
import asyncio
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/home/ai/alpaca')
ARTIFACTS = ROOT/'work/spf-p0p1-20260916/artifacts/development-extension-v1'
RUN = ARTIFACTS/'spf-judge-full-20261007-r1'
PYTHON = '/home/ai/miniforge3/envs/open-unlearning-repro/bin/python'
SOURCES = {
    'src/evals/safety/gpt_judge.py':'ddfd529d84ab10b4b3cc8054cecfac8fa8abcc4ef718b77a3b5e5d584eec153a',
    'src/construction/development_judge.py':'beb220e096e8aff1e60c15036fbf3d243b80fab32b28e35c43cea912f55c7d47',
    'src/construction/conversation_judge.py':'f0d34059488cc96c4ce6998d9424342e0488fa6427860e4529120415ca38659c',
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def bound_json(path, expected):
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError('Bound input changed: '+path.name)
    return json.loads(raw)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def atomic(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    temporary.replace(path)


async def execute(args):
    if (sys.executable != PYTHON or Path(args.run) != RUN or RUN.is_symlink()
            or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or digest(Path(__file__).read_bytes()) != args.source_sha256):
        raise ValueError('Exact interpreter, source, CPU-only environment and private run required')
    plan = bound_json(RUN/'plan.private.json',args.plan_sha256)
    packets = bound_json(RUN/'private-packet.json',args.packet_sha256)
    if (plan['total'] != 792 or plan['new_calls'] != 792 or plan['reused_response_labels'] != 0
            or plan['model'] != 'gpt-5.6-terra' or plan['reasoning'] != 'medium'
            or plan['max_output_tokens'] != 4096 or plan['retries'] != 0
            or len(packets) != 792 or len({r['id'] for r in packets}) != 792
            or plan['human_adjudication_complete'] is not False):
        raise ValueError('Original finite judge profile and complete SPF packet required')
    for name, expected in SOURCES.items():
        if digest((ROOT/name).read_bytes()) != expected:
            raise ValueError('Original rubric source changed: '+name)
    original = ARTIFACTS/'m0-judge-full-v1/plan.json'
    baseline = bound_json(original,plan['baseline_plan_sha256'])
    harmful = module('spf_harmful_contract',ROOT/'src/evals/safety/gpt_judge.py')
    benign = module('spf_benign_contract',ROOT/'src/construction/development_judge.py')
    conversation = module('spf_conversation_contract',ROOT/'src/construction/conversation_judge.py')
    specs = {'harmful':(harmful.SYSTEM_PROMPT,harmful.JudgeResult,'hexphi-two-dimensional-v1'),
             'benign':(benign.SYSTEM_PROMPT,benign.BenignJudgment,'benign-development-v1'),
             'conversation':(conversation.SYSTEM_PROMPT,conversation.ConversationJudgment,'wildchat-development-quality-v1')}
    hashes = {k:digest(v[0].encode()) for k,v in specs.items()}
    if hashes != baseline['prompt_hashes'] or hashes != plan['prompt_hashes']:
        raise ValueError('Original baseline rubric hashes required')
    if any((RUN/name).exists() for name in ['job.json','judge-results.jsonl','execution.lock','raw-responses']):
        raise FileExistsError('Never repeat or overwrite an existing judge execution')
    with (RUN/'execution.lock').open('x') as lock:
        lock.write(str(os.getpid()))
    (RUN/'raw-responses').mkdir(exist_ok=False)
    from openai import AsyncOpenAI
    # Credential use stays on the established judge host; never export secret values.
    harmful.load_env_file(Path('/home/ai/.config/open-unlearning/safety.env'))
    if not os.environ.get('OPENAI_API_KEY'):
        raise ValueError('Existing judge credential unavailable')
    client = AsyncOpenAI(max_retries=0,timeout=90)
    state = dict(status='RUNNING',pid=os.getpid(),total=792,attempted=0,completed=0,
                 source_sha256=args.source_sha256,packet_sha256=args.packet_sha256,
                 plan_sha256=args.plan_sha256,started_at=time.time(),retries=0,
                 human_adjudication_complete=False,target_gate='insufficient_evidence',
                 authorization='Human explicitly authorized paid Judge in this task on 2026-10-07',
                 training_executed=False,model_generation_executed=False)
    statuses = Counter();usage = Counter();models = Counter()
    atomic(RUN/'job.json',state)
    try:
        with (RUN/'judge-results.jsonl').open('x',encoding='utf8') as stream:
            for index, row in enumerate(packets):
                prompt,schema,version = specs[row['kind']]
                payload = json.dumps(row['messages'],ensure_ascii=False)
                if row['kind'] == 'harmful':
                    payload = 'HARMFUL USER INSTRUCTION:\n'+row['messages'][0]['content']+'\n\nMODEL RESPONSE:\n'+row['messages'][1]['content']
                result = dict(id=row['id'],blind_id=row['blind_id'],kind=row['kind'],
                              requested_model=plan['model'],prompt_version=version,
                              prompt_sha256=hashes[row['kind']],request_payload_sha256=digest(payload.encode()),
                              attempt=1,request_started_at=time.time())
                state['attempted'] += 1
                atomic(RUN/'job.json',state)
                try:
                    response = await client.responses.parse(model=plan['model'],reasoning={'effort':'medium'},
                        instructions=prompt,input=payload,text_format=schema,max_output_tokens=4096)
                    raw = (response.model_dump_json(indent=2)+'\n').encode()
                    raw_path = RUN/'raw-responses'/f'{index:04}.json'
                    raw_path.write_bytes(raw)
                    judgment = response.output_parsed
                    result.update(actual_model=response.model,response_id=response.id,
                        usage=response.usage.model_dump() if response.usage else None,
                        status='success' if judgment is not None else 'unparsed',
                        judgment=judgment.model_dump() if judgment is not None else None,
                        raw_response_sha256=digest(raw))
                    models[response.model] += 1
                    if response.usage:
                        usage['input_tokens'] += response.usage.input_tokens
                        usage['output_tokens'] += response.usage.output_tokens
                    if response.model != plan['model']:
                        result['status'] = 'unexpected_model'
                except Exception as error:
                    code = getattr(error,'code',None);body = getattr(error,'body',None)
                    if not code and isinstance(body,dict):
                        code = body.get('code') or (body.get('error') or {}).get('code')
                    result.update(status='policy_blocked' if harmful.is_policy_code(code) else 'error',
                                  error_type=type(error).__name__,error_code=code)
                result['request_finished_at'] = time.time()
                stream.write(json.dumps(result,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
                statuses[result['status']] += 1
                state.update(completed=state['completed']+1,statuses=dict(statuses),usage=dict(usage),actual_models=dict(models))
                atomic(RUN/'job.json',state)
                if result['status'] not in ('success','policy_blocked'):
                    state['status'] = 'STOPPED_ON_ERROR_NO_RETRY'
                    break
            else:
                state['status'] = 'JUDGE_COMPLETE_PENDING_HUMAN_REVIEW_AND_CLUSTER_GATE'
    finally:
        state['finished_at'] = time.time();atomic(RUN/'job.json',state)
        await client.close()
        atomic(RUN/'public-summary.json',state)
    print(json.dumps(dict(status=state['status'],attempted=state['attempted'],completed=state['completed'])))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run','source-sha256','packet-sha256','plan-sha256'):
        p.add_argument('--'+name,required=True)
    asyncio.run(execute(p.parse_args()))


if __name__ == '__main__':
    main()
