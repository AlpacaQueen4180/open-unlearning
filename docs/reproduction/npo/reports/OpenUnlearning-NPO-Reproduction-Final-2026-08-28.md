# OpenUnlearning NPO reproduction: final two-H100 evidence report

Date: 2026-08-28 (America/Chicago)

## Scope and conclusion

Target: TOFU / Llama-2-7B-chat / forget05 / NPO, documented approximately as:

| Result | Forget Quality | Model Utility | Forget Truth Ratio |
|---|---:|---:|---:|
| Published OpenUnlearning row | `0.09` | `0.53` | `0.71` |
| Exact two-H100 released topology | `0.1420746515` | `0.5292339411` | `0.7124040519` |

The published trade-off is closely reproduced when the released two-process DeepSpeed ZeRO-3 topology is executed. Model utility and truth ratio match the published row at its displayed precision (`0.53`, `0.71`). Forget quality is `0.142`, rather than `0.09`, but this is a small empirical-distribution difference: for 200 examples on each side, the reproduced KS distance is `D=0.115`; `FQ≈0.09` corresponds to `D=0.125`, only two `1/200` empirical-CDF increments away.

This reverses the earlier one-GPU-only conclusion. Effective-batch-equivalent one-GPU runs, historical/current package controls, an alternate trainer, and a duplicate-retain gradient diagnostic all missed the published regime. The exact released distributed topology is therefore operationally decisive. The evidence does not isolate whether the trajectory change comes from ZeRO-3 numerical behavior, distributed sampling interactions beyond the duplicate-retain diagnostic, or their combination.

## Headline comparison

| Run/control | Topology | FQ | MU | TR | Conclusion |
|---|---|---:|---:|---:|---|
| Published | Reported released result | `0.09` | `0.53` | `0.71` | Target |
| Two H100s, current release | 2 processes, DeepSpeed ZeRO-3, FlashAttention 2, micro 4 × accum 4 | `0.14207` | `0.52923` | `0.71240` | Closely reproduces the published trade-off |
| Current OpenUnlearning, GB200 | 1 GPU, eager attention, effective batch 32 | `1.4276e-12` | `0.57552` | `0.53672` | Does not reproduce |
| Initial source/version, Ada6000 | 1 GPU, commit `9be06255`, Transformers 4.45.1 | `6.5690e-12` | `0.57198` | `0.53699` | Does not reproduce |
| Alternate Llama-2 NPO+retain | 1 GPU, independent trainer, effective batch 32 | `1.2128e-10` | `0.63287` | `0.53456` | Does not reproduce |
| Duplicate-retain diagnostic | 1 GPU, synchronized-retain gradient emulation | `1.4276e-12` | `0.57473` | `0.53669` | Does not reproduce |

## Exact run provenance

- Host working directory: `/home/ai/alpaca`
- Repository commit: `4ad738aaf60f6a4385f6e2506d01da99e76c31f3`
- Hardware: 2 × NVIDIA H100 PCIe, 80 GB each
- Runtime: PyTorch `2.4.1+cu121`, Transformers `4.51.3`, Accelerate `0.34.2`, DeepSpeed `0.15.4`, bitsandbytes `0.44.1`, FlashAttention `2.6.3`
- Distributed evidence from the training log: NCCL backend, `world_size=2`, `zero_optimization_stage=3`, `train_batch_size=32`
- Model: `open-unlearning/tofu_Llama-2-7b-chat-hf_full`, bf16, `flash_attention_2`
- Data: `forget05` plus `retain95`
- NPO: `beta=0.1`, `alpha=1`, `gamma=1`, retain loss `NLL`
- Training: per-device batch `4`, gradient accumulation `4`, LR `1e-5`, weight decay `0.01`, `paged_adamw_32bit`, 10 configured epochs, seed `0`
- Trainer state: global step `60`, max steps `60`, recorded epoch `8.64`
- Training runtime: `1142.37` seconds (about 19 minutes)
- Task: `tofu_Llama-2-7b-chat-hf_forget05_NPO_2xH100_zero3_flash2_micro4_acc4_e10_20260827`

The first evaluation attempt failed after training because H100/FlashAttention preserved bf16 through probability calculation and PyTorch cannot convert bf16 directly to NumPy. The evaluation-only repair casts `avg_losses` and `normalized_probs` to float32 immediately before NumPy serialization. It was applied after the checkpoint was complete; it changes neither training nor metric definitions. Evaluation then succeeded from the unchanged checkpoint.

## Independent metric audit

The headline metrics were independently recomputed from the raw `TOFU_EVAL.json`, the official retain95 reference log, and the nine utility components. The recomputed values matched `TOFU_SUMMARY.json` exactly (absolute delta `0` for all three).

### Forget quality

- Model examples: `200`
- Retain-reference examples: `200`
- Two-sample KS statistic: `D=0.115`
- Exact KS p-value: `0.1420746514551761`
- Neighboring attainable distances: `D=0.120 → p=0.11228`; `D=0.125 → p=0.08784`

Thus the published `FQ≈0.09` and reproduced `FQ≈0.14` differ by only `0.010` in KS distance, or two empirical steps at this sample size. The p-value itself is nonlinear and should not be compared as if it were a linear score.

### Forget truth ratio

- Raw truth-ratio median: `0.7739484564`
- Raw truth-ratio mean: `0.7996479656`
- Fraction above 1: `0.21`
- Symmetric reported mean: `0.7124040519`

This is the same truth-ratio regime as the published `0.71`, and sharply different from the one-GPU released-style runs near `0.537`.

### Model utility

The independently recomputed harmonic mean of the nine released components is `0.5292339411`:

| Component | Value |
|---|---:|
| retain QA probability | `0.5749499512` |
| retain QA ROUGE | `0.5288615256` |
| retain truth ratio | `0.4310961100` |
| real-author normalized QA probability | `0.4116496000` |
| real-author QA ROUGE | `0.7518333333` |
| real-author truth ratio | `0.5208579500` |
| world-facts normalized QA probability | `0.4214098978` |
| world-facts QA ROUGE | `0.8945868946` |
| world-facts truth ratio | `0.5302662597` |

Other final metrics were forget QA probability `0.1127766800`, forget QA ROUGE `0.3476362711`, extraction strength `0.0831315382`, and privleak `20.3214120517`.

## Revised answers to the investigation questions

### A. Is the released NPO implementation correct?

Yes, within the documented objective. The code implements the original sequence-level NPO loss, freezes/evaluates the reference model correctly, and combines NPO with retain NLL using the documented weights. The successful exact-topology reproduction is additional operational evidence against a sign, mask, reference-gradient, or basic reduction error.

### B. Can the published row be reproduced from released settings?

Yes, closely, using the released two-process ZeRO-3 topology and corrected per-device batch size from maintainer issue #164. MU and TR match at published precision. FQ is nearby in the underlying KS statistic but not numerically identical.

### C. What explains the earlier mismatch?

Distributed execution topology is the decisive observed factor. Single-GPU “effective batch 32” is not equivalent for this experiment. The duplicate-retain diagnostic alone was insufficient, so it would be an overclaim to attribute the effect solely to the acknowledged synchronized-retain sampling bug. ZeRO-3 sharding/numerics, distributed data order and sampling, and their interaction remain candidate mechanisms.

The lack of a public matching checkpoint and resolved configuration remains a provenance limitation, but it is no longer necessary to posit an undocumented hyperparameter choice to explain the published row.

### D. What does extremely low FQ mean?

FQ is a KS p-value, not a monotonic “forgetting strength” score. The one-GPU runs' `10^-10`–`10^-12` values say their truth-ratio distributions are detectably different from retain95; they do not prove stronger forgetting. The two-H100 run moves into the published distributional regime (`D=0.115`, `p=0.142`) while preserving `MU≈0.53` and producing `TR≈0.71`.

## Immutable artifact evidence

Remote completion artifact:

```text
/home/ai/alpaca/tofu_Llama-2-7b-chat-hf_forget05_NPO_2xH100_zero3_flash2_micro4_acc4_e10_20260827_eval_retry2.status.json
```

Key SHA-256 checksums:

```text
904922d2c5f007d9d3952330ba10aed9ef3d4dc1a9ffb5f8a3f5591c4a0f32e4  model-00001-of-00003.safetensors
47805c14ffe24117108ee73eeaf492e206b63548d79556f5b9ce2362a49509c3  model-00002-of-00003.safetensors
ea3d4dd96bc2d0c92cef8ef4329fd372a40520e61158dd68a4f9c3ece30eb3f6  model-00003-of-00003.safetensors
e5500ffdc310def9ee2e264cb2f26ab78ce23214dddef68958882118c9efeada  trainer_state.json
66b84c5e519e552a38255e3cc41cdab38dd4fd1dcd5153e8bf71df34d71b70de  TOFU_SUMMARY.json
06f708e569af6610d39e906c78ac171a12b5baecef7a0fcb8e3b7b57912aa973  TOFU_EVAL.json
```

## Final assessment

The requested two-H100 control resolves the central uncertainty. The released NPO code/settings can recreate the published Llama-2 forget05 trade-off when run in its actual two-process DeepSpeed ZeRO-3 topology. The result is a close scientific reproduction rather than an exact checkpoint match: MU/TR match the displayed row, while FQ differs by two empirical KS steps.
