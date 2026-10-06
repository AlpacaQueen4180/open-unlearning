import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, TrainingArguments

from construction.backend import GradientBackend
from construction.data import ConstructionCollator
from construction.projection import ProjectionConfig
from construction.runner import ConstructionTrainer, SafetyMixingTrainer, SPFTrainer, task_windows


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
        ids = [2]
        for message in messages:
            ids += [3 if message["role"] == "user" else 4, 5, 6, 1]
        if add_generation_prompt:
            ids += [4]
        return ids if tokenize else "placeholder"

    def save_pretrained(self, path):
        pass


def tiny_model():
    torch.manual_seed(12)
    return GPT2LMHeadModel(GPT2Config(vocab_size=16, n_positions=32, n_embd=8, n_layer=1, n_head=2,
                                     resid_pdrop=0, embd_pdrop=0, attn_pdrop=0, use_cache=False))


def rows(length=11):
    return [{"input_ids": torch.tensor([2, 3, 4, 5 + i % 6, 1][:5 - i % 2]),
             "labels": torch.tensor([-100, -100, 4, 5 + i % 6, 1][:5 - i % 2]),
             "attention_mask": torch.ones(5 - i % 2, dtype=torch.long)} for i in range(length)]


def train_case(tmp_path, method="standard", micro=2, backend="single", coefficient=1.0, global_batch=8, tail_examples=3):
    from construction.protocol import sha256
    output = tmp_path / f"{method}-{micro}-{backend}-{coefficient}"
    anchor = tmp_path / "anchor.jsonl"
    anchor.parent.mkdir(parents=True, exist_ok=True)
    if not anchor.exists():
        anchor.write_text(json.dumps({"messages": [{"role": "user", "content": "test"},
                                                   {"role": "assistant", "content": "answer"}]}))
    args = TrainingArguments(str(output), use_cpu=backend == "single", bf16=False,
                             per_device_train_batch_size=micro,
                             gradient_accumulation_steps=global_batch // micro // (int(os.environ.get('WORLD_SIZE', '1')) if backend == "zero3" else 1),
                             learning_rate=.001, weight_decay=.01, optim="adamw_torch",
                             num_train_epochs=2, report_to=[], max_steps=3)
    cls = {"standard": ConstructionTrainer, "mixing": SafetyMixingTrainer, "spf": SPFTrainer}[method]
    trainer = cls(tiny_model(), args, rows(global_batch + tail_examples), ConstructionCollator(Tokenizer()), processing_class=Tokenizer(),
                  template_args={"apply_chat_template": True}, backend=backend, mixing_coefficient=coefficient,
                  anchor_path=str(anchor), anchor_sha256=sha256(anchor), warmup_epochs=0,
                  rank=2, svd_method="exact", save_intermediate=False, save_runtime=False)
    trainer.train()
    return trainer


@pytest.mark.parametrize("method", ["standard", "mixing", "spf"])
def test_effective_batch_accumulation_and_optimizer_steps(tmp_path, method):
    full = train_case(tmp_path / "full", method, micro=8)
    accumulated = train_case(tmp_path / "micro", method, micro=2)
    assert full.state.global_step == accumulated.state.global_step == 3
    assert full.lr_scheduler.last_epoch == accumulated.lr_scheduler.last_epoch == 3
    for p, q in zip(full.model.parameters(), accumulated.model.parameters()):
        torch.testing.assert_close(p, q, atol=3e-6, rtol=3e-5)


def test_mixing_zero_matches_standard(tmp_path):
    standard = train_case(tmp_path / "standard")
    mixing = train_case(tmp_path / "mixing", "mixing", coefficient=0)
    for p, q in zip(standard.model.parameters(), mixing.model.parameters()):
        assert torch.equal(p, q)


def test_standard_matches_manual_ce_adamw(tmp_path):
    trainer = train_case(tmp_path)
    model = tiny_model()
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if n in __import__('transformers').Trainer.get_decay_parameter_names(trainer, model)], "weight_decay": .01},
        {"params": [p for n, p in model.named_parameters() if n not in __import__('transformers').Trainer.get_decay_parameter_names(trainer, model)], "weight_decay": 0},
    ], lr=.001)
    from transformers import get_scheduler
    scheduler = get_scheduler("linear", optimizer, 0, 3)
    collator = ConstructionCollator(Tokenizer())
    count = 0
    for epoch in range(2):
        for ids in task_windows(11, 8, 42, epoch):
            optimizer.zero_grad()
            model(**collator([rows()[i] for i in ids])).loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            count += 1
            if count == 3:
                break
        if count == 3:
            break
    for p, q in zip(trainer.model.parameters(), model.parameters()):
        torch.testing.assert_close(p, q, atol=3e-6, rtol=3e-5)


def test_save_reload(tmp_path):
    trainer = train_case(tmp_path)
    trainer.save_model()
    loaded = GPT2LMHeadModel.from_pretrained(trainer.args.output_dir).eval()
    trainer.model.eval()
    batch = ConstructionCollator(Tokenizer())(rows()[:1])
    with torch.no_grad():
        torch.testing.assert_close(loaded(**batch).logits, trainer.model(**batch).logits)


def test_eos_attention_is_not_masked():
    batch = ConstructionCollator(Tokenizer())([
        {"input_ids": torch.tensor([2, 0]), "labels": torch.tensor([-100, 0]), "attention_mask": torch.ones(2)},
        {"input_ids": torch.tensor([2]), "labels": torch.tensor([2]), "attention_mask": torch.ones(1)},
    ])
    assert batch["attention_mask"].tolist() == [[1, 1], [1, 0]]


@pytest.mark.parametrize("method", ["standard", "mixing", "spf"])
def test_recipe_compatible_with_pinned_training_arguments(tmp_path, method):
    from construction.cli import load_recipe
    recipe = load_recipe(method)
    arguments = TrainingArguments(**{**recipe["args"], "bf16": False, "use_cpu": True,
                                    "output_dir": str(tmp_path)})
    assert arguments.seed == 0
    assert arguments.per_device_train_batch_size * arguments.gradient_accumulation_steps * 2 == 32


def test_mixing_preserves_safety_only_gradient_but_spf_does_not():
    model = torch.nn.ParameterDict({"task": torch.nn.Parameter(torch.ones(1)),
                                   "safety": torch.nn.Parameter(torch.ones(1))})
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    backend = GradientBackend(model, optimizer, scheduler)
    backend.combine({"task": torch.ones(1)}, {"safety": torch.tensor([2.])},
                    method="mixing", coefficient=.5, projection=ProjectionConfig(), step=0)
    assert model["safety"].grad.item() == 1.
    backend.combine({"task": torch.ones(1)}, {"safety": torch.tensor([2.])},
                    method="spf", coefficient=1., projection=ProjectionConfig(), step=0)
    assert model["safety"].grad is None
