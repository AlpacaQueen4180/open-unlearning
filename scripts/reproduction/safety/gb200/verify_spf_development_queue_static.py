"""New synthetic result/provenance guards only; never run adapters or import Torch."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import run_spf_development_queue as queue


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adapter-audit', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    adapter = queue.read_bound(a.adapter_audit, queue.ADAPTER_SHA)
    assert adapter['checkpoint_identity_sha256'] == queue.IDENTITY
    commands = queue.plan(queue.TASK/'cuda-development-r1')
    assert tuple(commands) == queue.STAGES
    assert commands['tofu'][-4:] == ['--batch-size','8','--max-new-tokens','200']
    assert commands['ifbench-score'][0] == queue.CPU_PYTHON
    for stage in ('ifbench','safety','conversation'):
        assert commands[stage][-2:] == ['--max-new-tokens', str(queue.CAPS[stage])]
    assert all('judge' not in part and 'train.py' not in part for command in commands.values() for part in command)
    assert queue.environment(cpu=True)['CUDA_VISIBLE_DEVICES'] == ''
    rejected = []

    def rejects(name, function):
        try:
            function()
        except (ValueError, KeyError):
            rejected.append(name)
        else:
            raise AssertionError('Invalid synthetic result accepted: '+name)

    def metadata(stage):
        result = dict(status='pass', checkpoint_identity_sha256=queue.IDENTITY, world_size=1,
                      tokenizer_revision=queue.META_REVISION)
        if stage=='tofu':
            result.update(teacher_forcing_max_length=512, batch_size_per_rank=8,
                          aggregates=dict(forget05=dict(count=200),retain95=dict(count=3800)),
                          manifest_sha256=queue.FREEZE_SHA)
        else:
            result.update(count=queue.DATA[stage][2], dataset_sha256=queue.DATA[stage][1])
        if stage=='knowledge':
            result.update(protocol='auxiliary-zero-shot-ABCD-summed-continuation-logprob-v1', num_fewshot=0, chat_template=False)
        else:
            result['generation'] = dict(max_new_tokens=queue.CAPS[stage],do_sample=False,use_cache=True)
        if stage=='conversation':
            result['turns_per_conversation'] = 2
        return result

    positive = {stage:metadata(stage) for stage in queue.STAGES[:-1]}
    for stage, value in positive.items():
        queue.validate_metadata(stage,value)
    cases = [
        ('wrong_spf_identity','tofu',lambda r:r.update(checkpoint_identity_sha256='0'*64)),
        ('wrong_tokenizer','safety',lambda r:r.update(tokenizer_revision='0'*40)),
        ('wrong_world','ifbench',lambda r:r.update(world_size=2)),
        ('partial_tofu','tofu',lambda r:r['aggregates']['retain95'].update(count=3799)),
        ('wrong_teacher','tofu',lambda r:r.update(teacher_forcing_max_length=256)),
        ('wrong_batch','tofu',lambda r:r.update(batch_size_per_rank=4)),
        ('wrong_freeze','tofu',lambda r:r.update(manifest_sha256='0'*64)),
        ('changed_cap','ifbench',lambda r:r['generation'].update(max_new_tokens=512)),
        ('sampling','safety',lambda r:r['generation'].update(do_sample=True)),
        ('cache_changed','conversation',lambda r:r['generation'].update(use_cache=False)),
        ('missing_second_turn','conversation',lambda r:r.update(turns_per_conversation=1)),
        ('partial_knowledge','knowledge',lambda r:r.update(count=1023)),
        ('changed_dataset','knowledge',lambda r:r.update(dataset_sha256='0'*64)),
        ('five_shot','knowledge',lambda r:r.update(num_fewshot=5)),
        ('chat_knowledge','knowledge',lambda r:r.update(chat_template=True)),
        ('changed_protocol','knowledge',lambda r:r.update(protocol='formal-mmlu')),
    ]
    for name,stage,mutate in cases:
        value=deepcopy(positive[stage]);mutate(value)
        rejects(name, lambda stage=stage,value=value:queue.validate_metadata(stage,value))
    proof=dict(status='CUDA_EXECUTOR_RETURNED',pid=123,source_sha256='1'*64,
               cuda_initialized=True,gpu_count=1,peak_allocated_bytes=1,
               checkpoint_identity_sha256=queue.IDENTITY)
    queue.validate_cuda(proof,123,'1'*64)
    for name,mutate in [
        ('wrong_child_pid',lambda r:r.update(pid=124)),
        ('changed_adapter',lambda r:r.update(source_sha256='2'*64)),
        ('no_actual_cuda',lambda r:r.update(cuda_initialized=False)),
        ('multiple_gpus',lambda r:r.update(gpu_count=2)),
        ('no_allocation',lambda r:r.update(peak_allocated_bytes=0)),
        ('wrong_cuda_identity',lambda r:r.update(checkpoint_identity_sha256='0'*64)),
    ]:
        value=deepcopy(proof);mutate(value)
        rejects(name,lambda value=value:queue.validate_cuda(value,123,'1'*64))
    sources=[dict(id=str(i),prompt='synthetic '+str(i)) for i in range(300)]
    rows=[dict(id=r['id'],prompt=r['prompt'],response='',generated_tokens=0) for r in sources]
    queue.validate_rows('ifbench',rows,sources)
    rejects('partial_generations',lambda:queue.validate_rows('ifbench',rows[:-1],sources))
    changed=deepcopy(rows);changed[0]['prompt']='changed'
    rejects('changed_prompt',lambda:queue.validate_rows('ifbench',changed,sources))
    changed=deepcopy(rows);changed[0]['generated_tokens']=2049
    rejects('generation_above_cap',lambda:queue.validate_rows('ifbench',changed,sources))
    knowledge_sources=[dict(id=str(i)) for i in range(1024)]
    knowledge=[dict(id=r['id'],scores=[0.,1.,2.,3.]) for r in knowledge_sources]
    queue.validate_rows('knowledge',knowledge,knowledge_sources)
    knowledge[0]['scores'][0]=float('nan')
    rejects('nonfinite_knowledge',lambda:queue.validate_rows('knowledge',knowledge,knowledge_sources))
    score=dict(status='pass',count=300,dataset_sha256=queue.DATA['ifbench'][1])
    queue.validate_metadata('ifbench-score',score)
    score['count']=299
    rejects('partial_ifbench_scores',lambda:queue.validate_metadata('ifbench-score',score))
    # This checks producer/consumer schema compatibility, not actual GPU evidence.
    state=dict(completed=list(queue.STAGES[:-1]),stages=[])
    for stage in queue.STAGES[:-1]:
        state['stages'].append(dict(name=stage,returncode=0,gpu_runtime_validated=True,
            process_pid=123,examples=4000 if stage=='tofu' else queue.DATA[stage][2],
            source_sha256=adapter['sources'][queue.NAMES[stage]]['adapter_sha256'],raw_log_sha256='3'*64))
    accepted=queue.acceptance(state,adapter)
    assert accepted['status']=='DEVELOPMENT_CUDA_EXECUTORS_PASS'
    state['completed'].pop()
    rejects('incomplete_cuda_stages',lambda:queue.acceptance(state,adapter))
    result=dict(status='PASS_SYNTHETIC_DEVELOPMENT_RESULTS_COMMANDS_AND_CUDA_METADATA_ONLY',
        rejected_cases=rejected, rejected_count=len(rejected),commands_checked=len(commands),
        source_sha256=queue.digest(Path(queue.__file__).read_bytes()),
        verifier_sha256=queue.digest(Path(__file__).read_bytes()),
        adapter_audit_sha256=queue.ADAPTER_SHA,synthetic_producer_consumer_schema_compatible=True,
        subprocesses_launched=0,torch_imported=False,gpu_runtime_validated=False,
        target_evaluation_executed=False,target_gate_evaluated=False,
        external_judge_executed=False,full_score_equivalence_verified=False,
        limitations=['Positive result/CUDA fixtures are synthetic; no actual executor acceptance',
                     'No Linux lock, imports, CUDA, model loading, generation or scoring was executed'])
    with Path(a.output).open('x',encoding='utf8') as handle:
        handle.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],rejected_count=len(rejected),subprocesses_launched=0)))


if __name__=='__main__':
    main()
