# NPO reproduction archive

This directory preserves the evidence and execution context for the TOFU NPO
reproduction performed in August 2026. The primary target was the reported
Llama-2-7B-chat `forget05` result (`FQ≈0.09`, `MU≈0.53`, `TR≈0.71`).

The exact released two-process DeepSpeed ZeRO-3 topology closely reproduced
the published trade-off on two H100 GPUs: `FQ=0.14207`, `MU=0.52923`, and
`TR=0.71240`. The five-seed H100 and single-Ada6000 matrices cover Llama-2-7B
and Llama-3.2-1B on `forget01`, `forget05`, and `forget10`.

## Layout

- `reports/` contains the final evidence report, experiment ledger, and compact
  completion artifacts.
- `environments/` contains package manifests captured from the actual H100 and
  Ada environments.
- `../../../scripts/reproduction/npo/as-run/` contains the scripts and
  diagnostics exactly as executed. They intentionally retain host-specific
  paths so the historical command provenance is not rewritten.
- `../../../results/reproduction/npo/` contains compact aggregate JSON and
  path/size inventories for the retained artifacts.

## Code change

`src/evals/metrics/utils.py` casts bfloat16 probability metrics to float32 only
at the NumPy serialization boundary. This evaluator-only repair was applied
after training and does not alter checkpoints or metric definitions.

## Large artifacts

Checkpoint weights, raw evaluation directories, logs, per-run status files,
and installed environments are deliberately excluded from Git. At migration
time they remained in place on the experiment machines:

- H100: `/home/ai/alpaca/saves/`
- Ada6000: `/home/user/alpaca/open-unlearning/saves/`

The committed TSV inventories record every retained safetensors shard and every
TOFU summary path with its byte size. Full checkpoint checksums should be added
before any future deletion or transfer of those machine-local artifacts.

## Reproduction base

- Upstream repository: `locuslab/open-unlearning`
- Base commit: `4ad738aaf60f6a4385f6e2506d01da99e76c31f3`
- Migration branch: `repro/npo-h100-ada-5seed`

See `reports/OpenUnlearning-NPO-Reproduction-Final-2026-08-28.md` for the
scientific conclusions and exact H100 headline-checkpoint checksums.
