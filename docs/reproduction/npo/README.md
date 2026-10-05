# NPO reproduction archive

This directory preserves the evidence and execution context for the TOFU NPO
reproduction performed in August and October 2026. The primary target was the reported
Llama-2-7B-chat `forget05` result (`FQ≈0.09`, `MU≈0.53`, `TR≈0.71`).

The exact released two-process DeepSpeed ZeRO-3 topology closely reproduced
the published trade-off on two H100 GPUs: `FQ=0.14207`, `MU=0.52923`, and
`TR=0.71240`. The five-seed H100 and single-Ada6000 matrices cover Llama-2-7B
and Llama-3.2-1B on `forget01`, `forget05`, and `forget10`.

## October 2026 update

A single GB200 with ZeRO-3 closely matched the historical two-H100 Llama-2
forget05 seed0 result; two physical GPUs are not necessary for that measured
result. This narrows the August topology conclusion without isolating hardware,
numerical, sampling or accumulation effects.

New instrumentation measured Trainer/DeepSpeed update mismatches. Six opt-in
corrected seed0 runs completed ten actual epochs with synchronized updates.
Earlier H100/Ada runs and the safety reports' Llama-3.1 NPO checkpoints did not
use this training correction. Safety `aggregate.corrected.json` is an evaluation
repair, not corrected training.

- [GB200 results and comparison with August H100/Ada](reports/GB200-NPO-Comparison-2026-10-06.md)
- [Original six-cell seed0 matrix](reports/GB200-NPO-Matrix-2026-10-05.md)
- [Corrected cells and original forget01 three-seed results](reports/GB200-NPO-Followup-2026-10-05.md)
- [Update-count mechanism and NPO/safety report correction inventory](update-count-mismatch-20261006.md)
- [Execution history, controls and environment](gb200-20261004.md)

Corrected update counts do not imply reproducing every published metric.
Corrected runs have one seed per cell; original forget01 seeds0/1/2 are analyzed
separately. All source training files remain unchanged; the opt-in correction
is archived under `scripts/reproduction/npo/gb200/`.

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

## Historical evaluator change

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
