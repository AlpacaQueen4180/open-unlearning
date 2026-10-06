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
- [Completed H100/GB200 gradient controls and new-stack three-split seed0 validation](environment-gradient-validation-20261006.md)
- [Loss-scaling mechanism, measured evidence and unresolved comparisons](loss-scaling-20261006.md)
- [Current upstream fixes, original-paper evidence limits, related papers and upgrade validation](upstream-status-20261006.md)
- [Execution history, controls and environment](gb200-20261004.md)

Corrected update counts do not imply reproducing every published metric.
The six corrected runs validate update counts and boundaries and retain the old
loss scaling. The October 6 controlled audits measured missing accumulation
normalization in the old H100/GB200 stack. Three separate native new-stack 1B
runs completed with aligned updates and ten actual epochs. They do not replace
the six update-only corrected runs or historical checkpoints.
Corrected runs have one seed per cell; original forget01 seeds0/1/2 are analyzed
separately. All source training files remain unchanged; the opt-in correction
is archived under `scripts/reproduction/npo/gb200/`.

As checked on 2026-10-06, upstream `17cbbc8` defaults to Transformers 5.5.4
and Accelerate 1.13.0. Their source paths provide ceil update planning,
Trainer/DeepSpeed boundary synchronization and accumulation normalization for
the standard nested NPO inputs. Subsequent runtime controls measured the old
full/tail gradient ratios of 8/2 and new ratios of 1/1 against a matched
DeepSpeed explicit-mean reference. All three native 1B splits independently
passed counts, checkpoint and metric audits; their scores do not reproduce all
`docs/repro.md` values. Real-1B bf16 autograd comparisons remain outside the 5%
gate and unexplained. Container Torch/bitsandbytes and two evaluator serialization
casts differ from upstream defaults. These observations do not establish the
same defect in the original-paper environment or upgrade historical safety runs.

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

At the August migration, checkpoint weights, raw evaluation directories, logs,
per-run status files and installed environments remained on the experiment
machines. The October 6 archive additionally publishes curated original small
evaluation/log/status evidence and verified hashes; large weights, gradient
vectors and frozen token tensors remain remote.

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
