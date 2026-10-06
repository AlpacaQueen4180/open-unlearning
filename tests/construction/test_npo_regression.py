import torch
import pytest


def test_npo_objective_and_frozen_reference_unchanged(tmp_path):
    pytest.importorskip("deepspeed", reason="Legacy trainer registry requires DeepSpeed; run on Linux environment")
    from transformers import GPT2Config, GPT2LMHeadModel, TrainingArguments
    from trainer.unlearn.npo import NPO
    from trainer.utils import compute_dpo_loss
    model = GPT2LMHeadModel(GPT2Config(vocab_size=16, n_embd=8, n_layer=1, n_head=2,
                                     resid_pdrop=0, embd_pdrop=0, attn_pdrop=0))
    trainer = NPO(model=model, args=TrainingArguments(str(tmp_path), use_cpu=True, report_to=[]),
                  beta=.1, alpha=1., gamma=1., retain_loss_type="NLL")
    batch = {"input_ids": torch.tensor([[2, 3, 4, 5]]), "attention_mask": torch.ones(1, 4),
             "labels": torch.tensor([[-100, -100, 4, 5]])}
    expected, _ = compute_dpo_loss(model, trainer.ref_model, None, batch, beta=.1)
    expected = expected + model(**batch).loss
    actual = trainer.compute_loss(model, {"forget": batch, "retain": batch})
    torch.testing.assert_close(actual, expected)
    assert trainer.ref_model is not trainer.model
