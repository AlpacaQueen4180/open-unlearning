"""New paired 32-update observations; delegate the frozen SPF loop and optimizer.

This prefix remains within the original 125-update warmup. It diagnoses update
geometry, not late-training harmfulness or a qualified safety starting model.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

CONSTRUCTION = Path('/data/spf-npo-20261006-r4')
REPO = CONSTRUCTION / 'repo'
PYTHON = '/data/npo-gb200-20261004/validation-20261006/venv/bin/python'
FREEZE = '122edbe2c09fcd2903e5b8949657f3dcaf3bf5cbc527e05b4cce91a29fb04d9e'
FULL = '7a7036b903ced64e18307e90523cc7ada73699ddb3d79498bfd4063f86ed8cf9'
def sha(raw): return hashlib.sha256(raw).hexdigest()
def bound(path, digest):
    raw = Path(path).read_bytes()
    if sha(raw) != digest: raise ValueError('Bound source or input changed: '+Path(path).name)
    return json.loads(raw)
def save(path, data):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(data, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())

class GeometryObserver:
    """Read existing gradient/parameter snapshots without replacing their values."""
    def __init__(self, torch, trainer, output):
        self.torch, self.trainer, self.output = torch, trainer, Path(output)
        self.clear_count = 0
        self.safety = None
        self.previous = None
        self.row = {}
        self.after_step = False
        self.records = []
    def dot(self, left, right):
        return sum(float((left[n] * right[n]).sum(dtype=self.torch.float64))
                   for n in left.keys() & right.keys())
    def norm(self, values):
        return math.sqrt(sum(float(g.square().sum(dtype=self.torch.float64)) for g in values.values()))
    def attach_backend(self):
        b = self.trainer.backend
        old_clear, old_combine, old_weights, old_step = b.clear, b.combine, b.weights, b.step
        def clear():
            value = old_clear()
            self.clear_count += 1
            return value
        def combine(task, safety, **kwargs):
            value = old_combine(task, safety, **kwargs)
            self.safety = safety
            self.row.update(raw_task_safety_dot=self.dot(task, safety), safety_gradient_norm=self.norm(safety))
            # Read one block at a time; do not create another 32GB gradient snapshot.
            projected_dot = projected_sq = 0.0
            for name, p in b.params.items():
                g = b.get_local(p) if b.engine is not None else p.grad
                if g is None: continue
                g = g.detach().float().cpu()
                projected_sq += float(g.square().sum(dtype=self.torch.float64))
                if name in safety:
                    projected_dot += float((g * safety[name]).sum(dtype=self.torch.float64))
            self.row.update(projected_gradient_safety_dot=projected_dot,
                            projected_gradient_norm=math.sqrt(projected_sq))
            return value
        def step():
            value = old_step()
            self.after_step = True
            return value
        def weights():
            values = old_weights()
            if not self.after_step:
                self.previous = values
            else:
                dot = squared = 0.0
                for name, current in values.items():
                    delta = current - self.previous[name]
                    squared += float(delta.square().sum(dtype=self.torch.float64))
                    if name in self.safety:
                        dot += float((delta * self.safety[name]).sum(dtype=self.torch.float64))
                norm = math.sqrt(squared)
                denom = norm * self.row['safety_gradient_norm']
                self.row.update(actual_update_safety_dot=dot, actual_update_norm=norm,
                                actual_update_safety_cosine=dot/denom if denom else None,
                                positive_dot_predicts_anchor_loss_increase=dot > 0)
                self.previous = self.safety = None
                self.after_step = False
                self.row['anchor_loss_after'] = self.observe_anchor()
            return values
        b.clear, b.combine, b.weights, b.step = clear, combine, weights, step
    def observe_anchor(self):
        t = self.torch
        model = self.trainer.forward_model
        flags = [(m, m.training) for m in model.modules()]
        cpu = t.random.get_rng_state()
        cuda = t.cuda.get_rng_state_all() if t.cuda.is_initialized() else None
        precision = (t.get_float32_matmul_precision(), t.backends.cuda.matmul.allow_tf32)
        buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
        engine = self.trainer.backend.engine
        counters = tuple(getattr(engine, name, None) for name in ('global_steps','micro_steps','skipped_steps'))
        try:
            model.eval()
            with t.no_grad():
                value = float(model(**self.trainer._move(self.trainer.anchor)).loss.detach())
        finally:
            for m, training in flags: m.training = training
            t.random.set_rng_state(cpu)
            if cuda is not None: t.cuda.set_rng_state_all(cuda)
        if precision != (t.get_float32_matmul_precision(), t.backends.cuda.matmul.allow_tf32):
            raise ValueError('Observer changed numerical context')
        if counters != tuple(getattr(engine, name, None) for name in ('global_steps','micro_steps','skipped_steps')):
            raise ValueError('Observer changed engine counters')
        current = dict(model.named_buffers())
        if current.keys() != buffers.keys() or any(not t.equal(current[name], value) for name,value in buffers.items()):
            raise ValueError('Observer changed model buffers; do not repair values')
        if not math.isfinite(value): raise ValueError('Nonfinite observed anchor loss')
        return value
    def attach(self):
        trainer = self.trainer
        old_setup, old_loss, old_record = trainer._setup, trainer._loss, trainer._record
        def setup(steps, warmup):
            value = old_setup(steps, warmup)
            if steps != 32 or warmup != 125 or trainer.backend.world != 1:
                raise ValueError('Original 125-warmup, world1, 32-update prefix required')
            self.attach_backend()
            return value
        def loss(batch):
            value = old_loss(batch)
            if self.clear_count % 2 == 0:
                self.row['anchor_loss_before'] = float(value.detach())
            return value
        def record(out, row):
            old_record(out, row)
            if row['step']:
                expected = {'raw_task_safety_dot','safety_gradient_norm','projected_gradient_safety_dot',
                            'projected_gradient_norm','actual_update_safety_dot','actual_update_norm',
                            'actual_update_safety_cosine','anchor_loss_before','anchor_loss_after'}
                if not expected.issubset(self.row): raise ValueError('Incomplete actual geometry observation')
                self.row.update(step=row['step'], lr_before=row['lr_before'],
                                anchor_loss_change=self.row['anchor_loss_after']-self.row['anchor_loss_before'])
                if any(isinstance(v,float) and not math.isfinite(v) for v in self.row.values()):
                    raise ValueError('Nonfinite geometry observation')
                with (self.output/'geometry.private.jsonl').open('a', encoding='utf8') as f:
                    f.write(json.dumps(self.row)+'\n'); f.flush(); os.fsync(f.fileno())
                self.records.append(dict(self.row)); self.row = {}
        trainer._setup, trainer._loss, trainer._record = setup, loss, record

def cpu_selftest():
    """One new real-AdamW delegation/RNG test, with no model weights or CUDA."""
    import tempfile
    import torch
    from types import SimpleNamespace
    if torch.cuda.is_initialized(): raise ValueError('CPU selftest must not initialize CUDA')
    all_records = []
    for beta in (0.9, 0.5):
        initial = torch.tensor([0.3,-0.7], dtype=torch.float32)
        def run(observed):
            p = torch.nn.Parameter(initial.clone())
            opt = torch.optim.AdamW([p], lr=1e-3, betas=(beta,0.999), weight_decay=0.01)
            class Model(torch.nn.Module):
                def forward(self, **unused): return SimpleNamespace(loss=(p.square()).sum())
            model = Model()
            def combine(task, safety, **unused):
                gt, gs = task['p'], safety['p']
                dot = (gt*gs).sum()
                p.grad = gt - (dot/gs.square().sum())*gs if dot < 0 else gt.clone()
                return {}
            b = SimpleNamespace(params={'p':p}, engine=None, world=1,
                                clear=lambda: opt.zero_grad(set_to_none=True), combine=combine,
                                weights=lambda: {'p':p.detach().clone()}, step=opt.step)
            tr = SimpleNamespace(backend=b,forward_model=model,anchor={},_move=lambda x:x)
            with tempfile.TemporaryDirectory() as directory:
                ob = GeometryObserver(torch,tr,directory) if observed else None
                if ob: ob.attach_backend()
                for i in range(4):
                    b.clear(); task={'p':torch.tensor([-0.8,0.5+i*0.1])}; safety={'p':p.detach().clone()*2}
                    b.combine(task,safety); previous=b.weights(); b.step(); current=b.weights()
                    if ob:
                        assert torch.equal(p.detach(),current['p'])
                        assert ob.row['actual_update_safety_dot'] == ob.dot({'p':current['p']-previous['p']},safety)
                return p.detach().clone()
        before = torch.random.get_rng_state()
        original, observed = run(False), run(True)
        assert torch.equal(original,observed) and torch.equal(before,torch.random.get_rng_state())
        all_records.append(dict(beta1=beta,bitwise_equal=True,steps=4))
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(status='NEW_CPU_ADAMW_OBSERVER_DELEGATION_PASS',cases=all_records,
                          cuda_initialized=False,weights_read=False,model_generation=False)))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--beta1',type=float,choices=(0.9,0.5)); p.add_argument('--output')
    p.add_argument('--self-test-cpu',action='store_true')
    a = p.parse_args()
    if a.self_test_cpu: cpu_selftest(); return
    if not a.output or a.beta1 is None or sys.executable != PYTHON or os.name != 'posix':
        raise ValueError('Exact original Linux runtime and explicit new diagnostic required')
    out = Path(a.output)
    if out.exists(): raise FileExistsError('Preserve existing diagnostic; no automatic rerun')
    manifest = bound(CONSTRUCTION/'frozen.json',FREEZE)
    bound(CONSTRUCTION/'spf-full-s0-audit.json',FULL)
    for name,digest in manifest['construction_code'].items():
        if sha((REPO/name).read_bytes()) != digest: raise ValueError('Frozen constructor changed: '+name)
    sys.path.insert(0,str(REPO/'src'))
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from construction.cli import load_recipe
    from construction.data import ConstructionDataset, ConstructionCollator
    from construction.protocol import read_rows
    from trainer import load_trainer
    if torch.cuda.device_count() != 1: raise ValueError('Single GPU required')
    recipe = load_recipe('spf')
    original_recipe = copy.deepcopy(recipe)
    recipe['args'].update(output_dir=str(out),per_device_train_batch_size=4,gradient_accumulation_steps=8,
                          seed=0,max_steps=32,adam_beta1=a.beta1,adam_beta2=0.999)
    recipe['method_args'].update(anchor_path=manifest['anchor_path'],anchor_sha256=manifest['anchor_sha256'],
                                save_intermediate=False,save_runtime=False)
    if recipe['args']['learning_rate'] != 1e-5 or recipe['args']['weight_decay'] != 0.01:
        raise ValueError('Original common optimizer profile required')
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(manifest['model_id'],revision=manifest['model_revision'],
        torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2',local_files_only=True)
    if model.config.attention_dropout != 0.0: raise ValueError('Observed anchor losses require original zero dropout')
    tokenizer = AutoTokenizer.from_pretrained(manifest['model_id'],revision=manifest['tokenizer_revision'],local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    template = OmegaConf.load(REPO/'configs/model/Llama-3.1-8B-Instruct.yaml').template_args
    data = manifest['datasets']['full']
    rows = bound(data['path'],data['sha256'])
    if len(rows) != 4000: raise ValueError('Original full4000 order required')
    dataset = ConstructionDataset(rows,tokenizer,template)
    trainer,_ = load_trainer(OmegaConf.create(recipe),model,train_dataset=dataset,
        processing_class=tokenizer,data_collator=ConstructionCollator(tokenizer),template_args=template)
    out.mkdir()
    save(out/'diagnostic-plan.private.json',dict(beta1=a.beta1,updates=32,warmup=125,original_recipe=original_recipe,
        actual_recipe=recipe,source_sha256=sha(Path(__file__).read_bytes()),same_original_m0=True,
        original_dataset_sha256=data['sha256'],original_loop_delegated=True,checkpoint_export=False,
        formal_gate=False,late_training_harmfulness_evaluated=False))
    observer = GeometryObserver(torch,trainer,out); observer.attach()
    result = trainer.train()
    if trainer.state.global_step != 32 or result.metrics['examples'] != 1024 or len(observer.records) != 32:
        raise ValueError('Actual prefix counters incomplete')
    if tuple(trainer.optimizer_metadata['defaults']['betas']) != (a.beta1,0.999):
        raise ValueError('Actual optimizer beta mismatch')
    save(out/'geometry-summary.private.json',dict(status='BOUNDED_PREFIX_GEOMETRY_COMPLETE',beta1=a.beta1,
        actual_updates=32,actual_examples=1024,warmup_steps=125,all_updates_within_warmup=True,
        positive_actual_safety_dot_steps=sum(r['actual_update_safety_dot']>0 for r in observer.records),
        increased_anchor_loss_steps=sum(r['anchor_loss_change']>0 for r in observer.records),
        geometry_sha256=sha((out/'geometry.private.jsonl').read_bytes()),optimizer=trainer.optimizer_metadata,
        harmfulness_labels_produced=False,causal_safety_root_cause_proven=False,
        source_sha256=sha(Path(__file__).read_bytes()),pid=os.getpid(),finished_at=time.time()))

if __name__ == '__main__': main()
