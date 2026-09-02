# OpenUnlearning NPO reproduction ledger

Updated: 2026-08-28 (America/Chicago)

| Hypothesis | Configuration difference | Run identifier | Status | FQ / MU / TR | Key secondary metrics | Conclusion |
|---|---|---|---|---|---|---|
| True released topology reproduces the table | 2 × H100; world size 2; DeepSpeed ZeRO-3; FlashAttention 2.6.3; per-device 4; accumulation 4; global batch 32; 10 configured epochs; seed 0 | `tofu_Llama-2-7b-chat-hf_forget05_NPO_2xH100_zero3_flash2_micro4_acc4_e10_20260827` | Complete and independently verified | `0.14207 / 0.52923 / 0.71240` | QA prob `0.11278`; QA ROUGE `0.34764`; extraction `0.08313`; privleak `20.32141`; KS `D=0.115`, `p=0.14207` | Closely reproduces official `0.09 / 0.53 / 0.71`. MU/TR match displayed precision; FQ is two empirical KS steps from the implied published distance. |
| Current released NPO reproduces the table on one GPU | Transformers 4.51.3; 1 GPU; microbatch 4; accumulation 8; eager attention | `tofu_Llama-2-7b-chat-hf_forget05_NPO_GB200_eager_retry_20260826` | Complete | `1.4276e-12 / 0.57552 / 0.53672` | QA prob `0.37500`; QA ROUGE `0.60490`; extraction `0.40730`; privleak `-97.616` | Does not reproduce official FQ/TR. |
| Initial source/version reproduces the table on one GPU | Commit `9be06255`; Transformers 4.45.1; 1 GPU; microbatch 8; accumulation 4 | `tofu_Llama-2-7b-chat-hf_forget05_NPO_official2025_micro8_accum4_retry3_eval8_20260827` | Complete | `6.5690e-12 / 0.57198 / 0.53699` | QA prob `0.32442`; QA ROUGE `0.59206`; KS `D=0.36`, `p=6.5690e-12` | Initial code/version and effective batch 32 still do not reproduce official FQ/TR. |
| Alternate mathematical implementation plus retain control matches the table | Alternate trainer; Llama-2-7B; 1 GPU; microbatch 8; accumulation 4; 10 epochs; NPO + retain weight 1 | `tofu_Llama-2-7b-chat-hf_forget05_NPO_alternate_retain_micro8_acc4_e10_retry2_20260827` | Complete | `1.2128e-10 / 0.63287 / 0.53456` | QA prob `0.80441`; QA ROUGE `0.62181`; extraction `0.54490`; privleak `-99.34484`; KS `D=0.34`, `p=1.2128e-10` | Contradicts official FQ/TR while preserving higher utility. Alternate implementation does not explain the reported result. |
| Alternate pure NPO's FQ match is a full reproduction | Alternate trainer; Llama-3.1-8B; pure NPO; batch 32; 10 epochs | existing GB10 control | Complete | `0.08784 / 0.17456 / 0.64248` | — | FQ alone matches; utility collapses, so this is not the published trade-off. |
| Hardware family explains discrepancy | Same current stack on GB10, GB200, Ada6000 | completed cross-hardware controls | Complete | Consistently extremely low FQ and TR near `0.537` | — | Hardware model is not the primary explanation. |
| True released topology is decisive | 2 physical H100 GPUs; DeepSpeed ZeRO-3; per-device 4; accumulation 4; 10 configured epochs | same H100 run above | Complete | `0.14207 / 0.52923 / 0.71240` | DeepSpeed log proves NCCL `world_size=2`, stage 3, train batch 32 | Yes. Exact topology changes the result from the one-GPU regime and closely recovers the published trade-off. |
| Acknowledged synchronized-retain bug explains topology sensitivity | One-GPU gradient-equivalent diagnostic: microbatch 8 with 8 distinct forget rows but first 4 retain rows duplicated as rows 5–8; accumulation 4; current stack; eager attention | `tofu_Llama-2-7b-chat-hf_forget05_NPO_rankdup_diag_micro8_acc4_e10_20260827` | Complete | `1.4276e-12 / 0.57473 / 0.53669` | QA prob `0.37261`; QA ROUGE `0.60698`; extraction `0.41022`; privleak `-97.84955`; KS `D=0.37`, `p=1.4276e-12`; step `60`, epoch `8.64` | Nearly identical to ordinary released implementation/GB200. The duplicate-retain mechanism alone does not explain the official row and failed to predict the true two-GPU result. |

## Provenance/configuration findings

- Target table row `FQ=0.09 / MU=0.53 / TR=0.71` exists from initial commit `9be06255` (2025-02-25).
- Initial and current `scripts/tofu_unlearn.sh` specify two GPUs, per-device batch 4, accumulation 4: effective batch 32.
- `docs/repro.md` was changed in commit `2e16546d` (2025-04-06) to say effective batch 32 was “8 per device, 4 grad accum steps.” On two GPUs that is effective batch 64, so the documentation conflicts mathematically with itself and with the launcher.
- Released Accelerate config specifies two processes and DeepSpeed ZeRO stage 3. The released NPO config is `beta=0.1`, `alpha=1`, `gamma=1`, retain loss `NLL`; TOFU config uses LR `1e-5`, weight decay `0.01`, warmup 1 epoch, and 10 epochs.
- The public Hugging Face eval dataset exposes full/retain baseline logs, but no discoverable Llama-2 NPO evaluation artifact; the public model collection likewise does not expose the target Llama-2 NPO checkpoint. The table row therefore lacks an auditable checkpoint/trainer-state pairing.
- The completed two-H100 run shows that exact distributed topology, not an undocumented hyperparameter, is sufficient to recover the published MU/TR regime and a nearby FQ.
- OpenUnlearning issue #164 received an explicit maintainer clarification: `8 per device` is a documentation mistake; the correct released setting is per-device 4 on two GPUs with accumulation 4. The maintainer also described the published reproducibility numbers as outdated and potentially requiring hyperparameter tuning.
- Issue #139 identifies a released distributed-data bug: synchronized RNG streams make all ranks sample the same retain examples while their forget examples differ. A maintainer acknowledged the bug; the proposed correction was reported to change experimental results and require retuning. PR/issue #145 remained open when checked.
- This bug gives true topology a concrete operational effect beyond effective batch size: with two ranks at batch 4, each optimizer microstep has 8 distinct forget examples but only 4 unique retain examples duplicated across ranks. Single-GPU microbatch 8 controls do not reproduce that retain-gradient sampling unless duplication is explicitly emulated.
- The one-GPU duplicate-retain diagnostic still did not predict the two-H100 outcome. Therefore the acknowledged sampling bug alone is not an adequate causal explanation; ZeRO-3/distributed numerical trajectory or a broader interaction remains involved.
- The superseded `locuslab/tofu` repository used a different target checkpoint (`locuslab/tofu_ft_llama2-7b`) and warned of Llama-2 FlashAttention reproducibility. Its shipped legacy baseline is close to, but not identical with, the recreated OpenUnlearning baseline; the released OpenUnlearning baseline exactly matches the documented table. Wholesale reuse of old baseline outputs is therefore unlikely, though the NPO run still lacks provenance artifacts.
- OpenUnlearning issue #199 independently reports the same method-specific provenance pattern for a released Llama-3.2 NPO checkpoint: full and retain controls reproduce, but the NPO checkpoint does not match its documented row. No maintainer response was present when checked.

## Released implementation audit

- With only a forget/loser sample, `compute_dpo_loss` evaluates sequence log-ratio `log πθ(y|x) - log πref(y|x)` from summed, response-masked token NLL and minimizes `-(2/β) log σ(-β log-ratio)`, exactly matching Equation 3 of the original NPO paper.
- Reference-model forwards are under `torch.no_grad()`, the copied reference is placed in evaluation mode, and under ZeRO-3 both active and reference models are initialized for sharding. The per-example forget loss is averaged across the local batch; equal distributed batches plus DDP gradient averaging preserve the intended global mean.
- Released `trainer=NPO` adds `gamma × NPO + alpha × retain NLL` with `beta=0.1`, `alpha=1`, `gamma=1`. This is the documented NPO+retain objective; sequence-summed forget log-probability and token-mean retain NLL are deliberate but imply their relative scale depends on completion length.
- The core loss code is unchanged between initial commit `9be06255` and current commit `4ad738aa`; intervening changes only add the modern Trainer `num_items_in_batch` argument and replace hard-coded CUDA placement with `self.accelerator.device`.
- Successful, stable optimization on multiple devices/stacks and agreement with an independently written alternate trainer provide operational evidence. No sign, masking, reference-gradient, or reduction defect was found that explains the earlier one-GPU gap; the exact distributed topology resolves it empirically.

## Truth-ratio interpretation

- Retain95 reference (200 examples): raw score median `0.8014`, mean `0.9040`, 27.5% above 1; symmetric truth-ratio mean `0.66956`.
- Released-style NPO controls on GB10 (three configurations): raw medians `0.5319–0.5420`, symmetric means `0.53586–0.53867`, and KS distance from retain95 `D=0.35–0.37` (`p=2.89e-11` to `1.43e-12`).
- Alternate Llama-2 NPO+retain: raw median `0.5540`, symmetric mean `0.53456`, 9% above 1, and KS `D=0.34`, `p=1.2128e-10`. Its truth-ratio distribution is therefore essentially the same shifted regime as the released implementation, not the official `TR≈0.71` regime.
- Exact two-H100 run: raw median `0.77395`, raw mean `0.79965`, 21% above 1, symmetric mean `0.71240`, and KS against retain95 `D=0.115`, `p=0.14207`. This closely matches the published regime.
- Full target baseline: FQ `5.8673e-14`, TR `0.51087`, QA probability `0.98949`, QA ROUGE `0.96536`. Retain95: QA probability `0.14910`, QA ROUGE `0.40115`, and symmetric TR `0.66956`.
- Released NPO does move QA probability/ROUGE and TR away from the full target toward retain95, so it performs some forgetting. But its extremely small FQ is only the KS p-value showing that its truth-ratio distribution remains detectably different from retain95; it is not evidence of stronger forgetting. The alternate control is especially clear: despite tiny FQ, QA probability remains high at `0.80441`.

## Completed H100 artifact

```text
/home/ai/alpaca/tofu_Llama-2-7b-chat-hf_forget05_NPO_2xH100_zero3_flash2_micro4_acc4_e10_20260827_eval_retry2.status.json
```

It contains `status: DONE`, all requested metrics, trainer state, and the independently confirmed 200-vs-200 KS result. The evaluator-only bf16-to-float32 serialization patch was applied after training and does not alter the checkpoint or metric definitions.
