# OpenUnlearning NPO reproduction: final evidence report

> **Superseded on 2026-08-28.** A subsequently acquired two-H100 machine enabled the missing exact ZeRO-3 control, which closely reproduced the published trade-off. Use `OpenUnlearning-NPO-Reproduction-Final-2026-08-28.md` as the current conclusion. The material below is retained only as the pre-H100 evidence snapshot.

Date: 2026-08-27 (America/Chicago)

## Scope and conclusion

Target: TOFU / Llama-2-7B-chat / forget05 / NPO, documented as approximately:

| Source | Forget Quality | Model Utility | Forget Truth Ratio |
|---|---:|---:|---:|
| Published OpenUnlearning row | `0.09` | `0.53` | `0.71` |

The released implementation is mathematically and operationally sound, but the published row was not reproduced from the released code/settings on any available execution path. The strongest remaining explanation is an undocumented or unarchived run provenance difference—checkpoint, resolved configuration, hyperparameter/early-stop choice, or training trajectory—rather than a sign error, evaluator defect, hardware family, Transformers version, attention implementation, effective batch size, alternate trainer, or the acknowledged duplicate-retain sampling bug.

A true 2-GPU ZeRO-3 run remains the only unexecuted released-topology control. It is unavailable because the Run:ai project quota is exactly one GPU; the valid 2-GPU workspace remains pending at zero allocation. This is recorded as residual uncertainty, not silently treated as tested.

## Key reproduction results

| Run/control | Relevant difference | FQ | MU | TR | Conclusion |
|---|---|---:|---:|---:|---|
| Published | Claimed released result | `0.09` | `0.53` | `0.71` | Target |
| Current OpenUnlearning, GB200 | Transformers 4.51.3; 1 GPU; eager; effective batch 32 | `1.4276e-12` | `0.57552` | `0.53672` | Does not reproduce |
| Initial source/version, Ada6000 | Commit `9be06255`; Transformers 4.45.1; 1 GPU; micro 8 × accum 4 | `6.5690e-12` | `0.57198` | `0.53699` | Code/package version does not explain gap |
| Alternate Llama-2 NPO+retain | Independent implementation; same mathematical objective; micro 8 × accum 4 | `1.2128e-10` | `0.63287` | `0.53456` | Alternate trainer does not explain gap |
| Released-style duplicate-retain diagnostic | 8 distinct forget rows; 4 retain rows duplicated twice, emulating synchronized two-rank retain gradients | `1.4276e-12` | `0.57473` | `0.53669` | Acknowledged distributed sampling bug does not explain gap |
| Alternate Llama-3.1 pure NPO | Pure NPO, no retain term | `0.08784` | `0.17456` | `0.64248` | Matches FQ alone only by collapsing utility |

The duplicate-retain diagnostic's secondary metrics were QA probability `0.37261`, QA ROUGE `0.60698`, extraction `0.41022`, privleak `-97.84955`, and 200-vs-200 KS `D=0.37`, `p=1.4276e-12`. It is nearly identical to the ordinary released implementation.

## A. Is the released NPO implementation correct?

Yes, within the documented objective.

- `compute_batch_nll` sums response-masked token NLL for each sequence.
- The code forms `log πθ(y|x) - log πref(y|x)` and minimizes `-(2/β) log σ(-β log-ratio)`, matching Equation 3 of the original NPO paper.
- Reference forwards use `torch.no_grad()` and the copied reference model is in evaluation mode.
- Released `trainer=NPO` adds retain NLL with `beta=0.1`, `alpha=1`, and `gamma=1`.
- The core NPO loss is unchanged between initial commit `9be06255` and current commit `4ad738aa`; later edits are device/API compatibility changes.
- Stable training across GB10, GB200, and Ada6000 plus agreement with an independently written trainer provide operational evidence.

No sign, label-mask, reference-gradient, or reduction error was found that could generate the published/reproduced discrepancy.

## B. Can the published row be reproduced from released settings?

No, on every available reproduction path. Results repeatedly cluster around FQ `10^-10` to `10^-12` and TR `0.535–0.537`, not FQ `0.09` and TR `0.71`.

The remaining qualification is that the exact 2-GPU/ZeRO-3 topology could not be executed. However:

- A maintainer clarified in [issue #164](https://github.com/locuslab/open-unlearning/issues/164) that the correct setting is per-device 4, two GPUs, accumulation 4; the documentation's per-device 8 was a mistake.
- Effective-batch-equivalent single-GPU runs do not reproduce the row.
- The specific known two-rank retain-sampling effect was isolated and emulated, and produced an unchanged result.
- Hardware family, current versus historical Transformers, and eager/default attention controls are consistent.

Thus the released settings do not currently constitute a complete, auditable reproduction recipe for the published row.

## C. Most likely mismatch

Most likely: run/checkpoint provenance or an undocumented resolved configuration/hyperparameter choice.

Evidence:

1. The target row appeared in the initial OpenUnlearning commit, but the repository contains no matching raw `TOFU_EVAL.json`, resolved Hydra config, trainer state, loss trajectory, or immutable checkpoint.
2. The public evaluation collection exposes full/retain baselines but no auditable Llama-2 NPO artifact corresponding to the row.
3. Full and retain controls reproduce their documented values, localizing the mismatch to NPO run provenance rather than the evaluator or dataset.
4. A maintainer described the reproducibility table as outdated and potentially requiring hyperparameter tuning in [issue #164](https://github.com/locuslab/open-unlearning/issues/164).
5. [Issue #199](https://github.com/locuslab/open-unlearning/issues/199) independently reports the same method-specific pattern for a released Llama-3.2 NPO checkpoint: full/retain controls reproduce while the NPO checkpoint does not match its documented row.
6. The acknowledged synchronized-retain bug from [issue #139](https://github.com/locuslab/open-unlearning/issues/139) was a plausible topology mechanism, but its diagnostic result is indistinguishable from the ordinary run.

Topology remains a lower-probability residual because true ZeRO-3 execution was unavailable. A simple code-version or package mismatch is unlikely.

## D. What does extremely low FQ mean?

It does not mean stronger forgetting by itself. OpenUnlearning FQ is the two-sample KS p-value comparing the unlearned model's per-example truth-ratio distribution with the retain95 reference distribution. A tiny value means the distributions are detectably different; it does not specify a desirable direction.

- Retain95 truth-ratio scores: median `0.8014`, symmetric mean `0.66956`, 27.5% above 1.
- Released-style NPO runs: medians approximately `0.532–0.554`, symmetric means `0.535–0.539`, only 9–11% above 1, KS `D=0.34–0.37`.
- Full target baseline: FQ `5.8673e-14`, TR `0.51087`, QA probability `0.98949`, QA ROUGE `0.96536`.
- Released NPO lowers QA probability/ROUGE and shifts TR toward retain95, so it performs some forgetting, but remains far from the retain distribution.
- The alternate control makes the point especially clearly: FQ `1.21e-10` coexists with QA probability `0.80441`, showing substantial retained answer confidence.

The published `TR≈0.71` describes a fundamentally different truth-ratio/utility trade-off from the released runs. FQ alone cannot establish successful forgetting.

## Residual experiment if resources change

If two physical GPUs become available, run exactly:

1. Current released commit, two CUDA devices, DeepSpeed ZeRO-3, per-device batch 4, accumulation 4, 10 epochs, seed 42, LR `1e-5`, weight decay `0.01`, `beta/alpha/gamma=0.1/1/1`, retaining the released synchronized-sampling behavior for the first audit.
2. If still discrepant, repeat the same topology with commit `9be06255` and Transformers `4.45.1`.

Do not use fractional GPU allocations as evidence. Do not stop the existing Run:ai workspace until its non-PVC repo, environments, checkpoints, and logs are backed up to `/data`.
