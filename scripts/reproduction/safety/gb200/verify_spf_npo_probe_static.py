"""CPU static/stub checks for the new smoke observer; never import Torch or train."""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Sequence

from spf_npo_native_probe import NATIVE_SOURCE_FILES
from spf_npo_runtime import audit_trace, source_mask_collator


class TensorStub:
    def __init__(self, values, dtype='int'):
        self.values, self.dtype = values, dtype
    def __len__(self):
        return len(self.values)
    def tolist(self):
        return deepcopy(self.values)
    def ne(self, value):
        return TensorStub([[x != value for x in row] for row in self.values], 'bool')
    def to(self, *, dtype):
        converter = bool if dtype == 'bool' else int
        return TensorStub([[converter(x) for x in row] for row in self.values], dtype)
    def flip(self, dims):
        return TensorStub(list(reversed(self.values)) if dims == [0]
                          else [list(reversed(row)) for row in self.values], self.dtype)


def pad_sequence(items, batch_first, padding_value):
    assert batch_first
    maximum = max(map(len, items))
    return TensorStub([item.tolist() + [padding_value] * (maximum-len(item)) for item in items])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('repository','frozen','upstream-data-utils','upstream-unlearn-base','output'):
        p.add_argument('--'+name,required=True)
    a = p.parse_args()
    repo = Path(a.repository)
    frozen_raw = Path(a.frozen).read_bytes()
    frozen = json.loads(frozen_raw)
    sha = lambda raw: hashlib.sha256(raw).hexdigest()
    sources = []
    for name in NATIVE_SOURCE_FILES:
        pinned = {'src/data/utils.py':Path(a.upstream_data_utils),
                  'src/trainer/unlearn/base.py':Path(a.upstream_unlearn_base)}
        local = pinned.get(name, repo/name)
        content = local.read_bytes()
        normalized = content.replace(b'\r\n',b'\n')
        assert sha(normalized) == frozen['construction_code'][name], name
        sources.append(dict(path=name,expected_sha256=sha(normalized),
                            local_bytes_sha256=sha(content),LF_normalization_only=content!=normalized,
                            source='pinned upstream GET, exact frozen bytes' if name in pinned else 'existing local file'))
    parent_source = (repo/'src/data/collators.py').read_bytes()
    tree = ast.parse(parent_source)
    node = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DataCollatorForSupervisedDataset')
    fake_torch = SimpleNamespace(Tensor=TensorStub,tensor=TensorStub,flip=lambda t,dims:t.flip(dims),
        nn=SimpleNamespace(utils=SimpleNamespace(rnn=SimpleNamespace(pad_sequence=pad_sequence))))
    namespace = dict(torch=fake_torch,transformers=SimpleNamespace(PreTrainedTokenizer=object),
                     Dict=Dict,Sequence=Sequence,IGNORE_INDEX=-100)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'native-collator-ast','exec'),namespace)
    base = namespace['DataCollatorForSupervisedDataset']
    adapter = source_mask_collator(base)
    rows = [dict(input_ids=TensorStub([9,2,7,2]),labels=TensorStub([-100,-100,7,2]),
                 attention_mask=TensorStub([1,1,1,1]),index=0),
            dict(input_ids=TensorStub([9,2]),labels=TensorStub([-100,2]),
                 attention_mask=TensorStub([1,1]),index=1)]
    mask_cases = []
    for padding_side, expected in [('right',[[1,1,1,1],[1,1,0,0]]),
                                   ('left',[[1,1,1,1],[0,0,1,1]])]:
        kwargs = dict(tokenizer=SimpleNamespace(pad_token_id=2),index='index',padding_side=padding_side)
        old, new = base(**kwargs)(rows), adapter(**kwargs)(rows)
        assert new['attention_mask'].tolist() == expected
        assert old['attention_mask'].tolist() != new['attention_mask'].tolist()
        assert all(old[key].tolist() == new[key].tolist() for key in ('input_ids','labels','index'))
        nested = adapter(**kwargs)([dict(forget=row,retain=row) for row in rows])
        assert nested['forget']['attention_mask'].tolist() == expected
        mask_cases.append(dict(padding_side=padding_side,valid_EOS_retained=True,
            native_input_label_index_padding_unchanged=True,nested_pair_supported=True))
    rejected_masks = []
    for name, mutate in [('above512',lambda r:r[0].update(input_ids=TensorStub([1]*513))),
                         ('wrong_source_mask_length',lambda r:r[0].update(attention_mask=TensorStub([1]))),
                         ('masked_unpadded_source_token',lambda r:r[0].update(attention_mask=TensorStub([1,0,1,1])) )]:
        changed = deepcopy(rows);mutate(changed)
        try:
            adapter(tokenizer=SimpleNamespace(pad_token_id=2))(changed)
        except ValueError:
            rejected_masks.append(name)
        else:
            raise AssertionError(name+' accepted')
    positive = dict(trainer_global_step=2,engine_global_steps=2,microsteps=10,epoch=1.0,
        world_size=1,micro_batch=4,configured_gas=8,alpha=0.0,gamma=1.0,beta=0.1,
        retain_loss_type='NLL',learning_rate=1e-7,dataset_examples=40,retain_pool_examples=3800,
        forget_exposures=40,retain_exposures=40,unique_forget_examples=40,
        attention='flash_attention_2',optimizer='paged_adamw_32bit',zero_stage=3,
        reference_fingerprint_unchanged=True,initial_target_reference_equal=True,
        loss_scaling_trace=[dict(raw_npo_loss=13.0,loss_entering_ds_backward=13.0/window,
            actual_window_microbatches=window,boundary=i in (7,9),
            kwargs=dict(scale_wrt_gas=False),mask_valid=True) for i,window in enumerate([8]*8+[2]*2)])
    assert audit_trace(positive)['fresh_process_reload_verified'] is False
    negatives = []
    cases = [('wrong_actual_updates',lambda r:r.update(trainer_global_step=1)),
             ('wrong_epoch',lambda r:r.update(epoch=2)),
             ('alpha1',lambda r:r.update(alpha=1)),
             ('reference_mutated',lambda r:r.update(reference_fingerprint_unchanged=False)),
             ('wrong_initial_reference',lambda r:r.update(initial_target_reference_equal=False)),
             ('duplicate_forget_exposure',lambda r:r.update(unique_forget_examples=39)),
             ('double_full_window_division',lambda r:r['loss_scaling_trace'][0].update(loss_entering_ds_backward=13/64)),
             ('tail_divided_by8',lambda r:r['loss_scaling_trace'][8].update(loss_entering_ds_backward=13/8)),
             ('ds_second_division_enabled',lambda r:r['loss_scaling_trace'][0]['kwargs'].update(scale_wrt_gas=True)),
             ('wrong_boundary',lambda r:r['loss_scaling_trace'][7].update(boundary=False)),
             ('masked_assistant_token',lambda r:r['loss_scaling_trace'][0].update(mask_valid=False)),
             ('nonfinite_loss',lambda r:r['loss_scaling_trace'][0].update(raw_npo_loss=float('nan')))]
    for name, mutate in cases:
        changed = deepcopy(positive);mutate(changed)
        try:
            audit_trace(changed)
        except ValueError:
            negatives.append(name)
        else:
            raise AssertionError(name+' accepted')
    directory = Path(__file__).resolve().parent
    result = dict(status='PASS_STATIC_SOURCE_STUB_MASK_AND_SYNTHETIC_TRACE_ONLY',
        frozen_sha256=sha(frozen_raw),native_source_files=sources,mask_cases=mask_cases,
        invalid_masks_rejected=rejected_masks,synthetic_trace_cases_rejected=negatives,
        synthetic_reference_flags_are_actual_model_evidence=False,
        source_sha256={name:sha((directory/name).read_bytes()) for name in
                       ('spf_npo_runtime.py','spf_npo_native_probe.py','verify_spf_npo_probe_static.py')},
        torch_runtime_validation='not_run',gpu_runtime_validation='not_run',
        launch_contract_prepared=False,npo_smoke_executed=False,fresh_process_reload_verified=False,
        pilot_candidates_frozen=False,external_judge_calls=0,
        limits=['Native collator class was AST-extracted and exercised with stdlib tensor stubs; no real Torch tensors or model.',
                'Counter/reference/scaling vectors are synthetic, not actual NPO or gradient evidence.',
                'Launch/config integration, real Torch/DS, all-parameter reference checks and fresh reload remain required.'])
    output = Path(a.output)
    if output.exists():
        raise FileExistsError('Refuse to replace existing evidence')
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status=result['status'],source_files=len(sources),mask_cases=len(mask_cases),
                         invalid_masks=len(rejected_masks),synthetic_trace_rejections=len(negatives))))


if __name__ == '__main__':
    main()
