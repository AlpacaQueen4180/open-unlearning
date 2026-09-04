# Experiment 2: Llama-3.1-8B checkpoint comparison

## Research question

Does benign TOFU unlearning damage safety alignment, and is any observed safety
degradation attributable to forgetting itself or to unconstrained parameter
movement and collateral utility loss?

The first-stage experiment uses TOFU `forget05` and compares five checkpoint
conditions. Only two conditions require new training.

| Condition | Model/checkpoint | Training required |
|---|---|---:|
| Original instruct | `meta-llama/Llama-3.1-8B-Instruct` | No |
| TOFU target/full | `open-unlearning/tofu_Llama-3.1-8B-Instruct_full` | No |
| NPO without retain | Local output `tofu_safety_exp2_Llama-3.1-8B-Instruct_forget05_NPO_no_retain_2xH100_seed0_20260904` | Yes |
| NPO + retain95 | Local output `tofu_safety_exp2_Llama-3.1-8B-Instruct_forget05_NPO_retain95_2xH100_seed0_20260904` | Yes |
| Retain-only/retrain oracle | `open-unlearning/tofu_Llama-3.1-8B-Instruct_retain95` | No |

## Controlled training settings

- Model: Llama-3.1-8B-Instruct TOFU full checkpoint
- Forget split: `forget05`
- Retain split: `retain95`
- NPO: `beta=0.1`, `gamma=1`
- NPO without retain: `alpha=0`
- NPO + retain95: `alpha=1`, retain loss `NLL`
- Learning rate: `1e-5`
- Weight decay: `0.01`
- Epochs: `10`
- Seed: `0`
- Topology: two H100 processes, DeepSpeed ZeRO-3
- Effective batch size: 32 (microbatch 4 × accumulation 4 × world size 2)
- Attention: FlashAttention 2, bfloat16
- Gradient checkpointing: enabled

`alpha=0` is the pure-NPO ablation. The current trainer still loads and forwards
the retain batch, but multiplying its NLL by zero gives it zero gradient
contribution. This preserves the same data loader and execution structure across
the two trained conditions.

## Evaluation contract

TOFU and safety results must be joined by immutable checkpoint ID and training
seed. For every condition, report:

- Forget Quality p-value and KS statistic
- Forget Truth Ratio
- Forget QA probability and ROUGE
- Retain Truth Ratio and Model Utility
- Harmful assistance rate
- Refusal rate
- Safe non-refusal rate
- Degeneration rate
- Harmful-among-nondegenerate rate

Safety generation uses both the legacy 300-prompt sampling protocol and a
300-prompt deterministic protocol for every checkpoint. Generation and judging
run on the H100 host before trained checkpoints are archived to the GB10 host.
The formal GPT judge is `gpt-5.6-terra` with medium reasoning and prompt version
`hexphi-two-dimensional-v1`. It must return two orthogonal labels per response:

1. safety: `refusal`, `safe_non_refusal`, or `harmful_assistance`;
2. output quality: `normal` or `degenerate`.

Raw model responses, judge labels, judge rationale, prompt/template version,
model version, decoding parameters, and retry/error state should be retained.

## Repository handoff

Push code, manifests, and compact aggregate metrics to GitHub. Do not push
HEx-PHI prompt text, raw model responses, judge rationales, or secrets.
Do not push multi-gigabyte model weights to ordinary Git history. The trained
weights should remain on the H100 host or be uploaded to a model/artifact store;
GitHub should carry stable checkpoint identifiers and checksums.

After the seed-0 pipeline is validated end to end, expand both trained conditions
to seeds 1–4 and perform paired query-level safety comparisons.

## Safety evaluation CLI

Install optional dependencies with `pip install '.[safety-eval]'`. Raw prompts,
responses, and judge rationales must remain under `safety_artifacts/`, which is
ignored by Git.

```bash
PYTHONPATH=src python -m evals.safety generate \
  --model MODEL_OR_CHECKPOINT --input HEXPHI.json --output RAW.jsonl \
  --run-id RUN_ID --protocol legacy-300 --checkpoint-id CHECKPOINT_ID

PYTHONPATH=src python -m evals.safety judge \
  --input RAW.jsonl --output JUDGED.jsonl \
  --model gpt-5.6-terra --reasoning-effort medium \
  --concurrency 8 --retries 3 --timeout 60 \
  --env-file /home/ai/.config/open-unlearning/safety.env

PYTHONPATH=src python -m evals.safety summarize \
  --input JUDGED.jsonl --output SUMMARY.json
```

The secret environment file must be mode `0600` and define
`OPENAI_API_KEY`. The CLI records both the requested judge alias and the actual
model ID returned by the API. Successful rows are resumed by stable ID; failed
rows remain visible and retry on the next run.

The legacy prompt file has SHA-256
`f72785518afa1dde3c1324987e123ef307a6e2ee2b69a8646c738c06e051db2e`.
The formal dataset metadata is pinned to Hugging Face revision
`83128b46334b80cc567bd7a2caf7af11c5b0bab7`. The authors removed all 30
category-2 prompts from the repository, so the currently obtainable benchmark
contains 300 prompts. Category 2 must be reported as unavailable, not recreated
or substituted. HEx-PHI prompt text and raw generations must not be committed or
redistributed.
