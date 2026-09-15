# OLMo fully-open benign unlearning proposal

Date: 2026-09-15. Status: proposed protocol; no OLMo runs reported here.

## Motivation and checkpoint selection

**Question:** Does removing known benign training examples damage an already aligned model, after accounting for utility loss and degeneration? The [Llama five-seed report](llama31_npo_5seed_report_20260911.md) motivates avoiding another custom target/full fine-tuning stage.

Use official OLMo 2 7B checkpoints directly. Stage 0 evaluates `allenai/OLMo-2-1124-7B-SFT`, `allenai/OLMo-2-1124-7B-DPO`, and `allenai/OLMo-2-1124-7B-Instruct` on baseline safety, instruction following, and generation quality. Choose the earliest sufficiently aligned stage, preferring SFT for direct data attribution; use DPO or final Instruct if earlier stages fail. Proposed preregistered gate: HEx-PHI full-set harmfulness upper bound <=15%, judge coverage >=95%, and degeneration <=1%. If none passes, report that limitation and revise the protocol before unlearning; do not assume the Instruct label guarantees safety.

Pin model/tokenizer revisions, official chat template, dataset revision, and training recipe. Avoid mixing the old preview checkpoints with the corrected releases. The [official model card](https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct) documents the SFT → DPO → RLVR lineage. Keep the corresponding pre-SFT base checkpoint as a diagnostic reference only. Confirm examples survive the official preprocessing and assistant-token masking before calling them known SFT members; lineage membership does not prove memorization or absence from pretraining.

## Data: nested requests and fixed retain pools

Extract `source=ai2-adapt-dev/flan_v2_converted` from [allenai/tulu-3-sft-olmo-2-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-olmo-2-mixture), not the generic Tülu mixture. The card lists 89,982 FLAN examples within 939,344 SFT examples:

| Forget request | Fraction of original FLAN component | Fraction of full SFT mixture |
|---|---:|---:|
| 1K | 1.11% | 0.11% |
| 5K | 5.56% | 0.53% |
| 10K | 11.11% | 1.06% |

“~1/5/10%” refers only to FLAN. Recount the pinned revision and separately report the denominator after filtering.

- Remove harmful, refusal-training, and safety-related examples using a versioned classifier plus 200–500 manual spot checks. Audit duplicate/near-duplicate prompts and responses across the full mixture and later post-training stages; exclude ambiguous overlap from this clean study.
- Fix a task/length-balanced permutation (split seed 42): `F_1K ⊂ F_5K ⊂ F_10K`. Preserve original IDs, hashes, source/task, length, filtering decisions, and preprocessing coverage.
- From the remaining eligible FLAN pool reserve `R_reg=10K` for retain regularization and `R_eval=10K` for final retain evaluation. Require **pairwise** disjointness among `F_10K`, `R_reg`, and `R_eval`, including duplicate groups. All sizes/methods share these pools.
- Reserve a separate `R_dev=2K` from the remainder for pilot selection; do not tune on `R_eval`. Retain-eval/dev are held out from unlearning, but are still original SFT members.
- Save fixed 500-example generation subsets per request and retain pool; include a common subset from F_1K across all sizes for paired comparisons, alongside representative per-size subsets. Score NLL over each full requested set.

## Methods, pilot, and frozen runs

Use full-parameter NPO with the selected starting checkpoint M0 as its frozen reference. Match the [current trainer](../../../src/trainer/unlearn/npo.py): `loss = gamma * NPO + alpha * retain_NLL`, with `gamma=1`; pure NPO uses `alpha=0`, NPO+retain uses `alpha=1`. Retain batches come only from R_reg, sampled 1:1 with forget examples. The current alpha=0 path still forwards retain batches; record this compute overhead.

1. **5K pilot, training seed 0:** sweep LR `{1e-7, 3e-7, 1e-6}` for both methods, fixed `beta=0.1`, effective forget batch 32, one epoch, gradient clipping 1.0. Use AdamW, weight decay 0.01, constant LR/no warmup, bf16 and the existing H100/ZeRO-3 setup where compatible. Validate OLMo loading, loss normalization, masking, length cap, and actual example coverage before the sweep. Extend to 3e-6 only if no setting produces a forgetting signal.
2. Save 25%, 50%, and 100% epoch checkpoints. Select using forget NLL plus generation metrics and R_dev/utility development checks, **not post-unlearning HEx-PHI outcomes**. Proposed feasibility limits: R_dev NLL increase <=0.1 nats/token, instruction-following development drop <=5 percentage points, degeneration <=1%. Among feasible final-epoch settings choose the lowest LR with positive paired forget-NLL change and decreased extraction; publish all pilot results and freeze the decision. Prefer a shared LR for direct method comparison; if only method-specific LRs are feasible, label that comparison and retain shared-LR pilot results.
3. **Formal scalability:** freeze checkpoint, LR(s), beta, alpha, batch, optimizer, masking, decoding, and split manifests; run 1K/5K/10K × two methods. One complete epoch means approximately 32/157/313 updates with batch 32 and final partial batches retained; log actual updates, tokens, and coverage.
4. Validate the six seed-0 runs end to end, then expand to training seeds 1–4 (30 main training runs total). Keep data splits fixed: this measures optimization-seed variability, not split variability.
5. One-epoch results combine request size and optimization budget. Add a secondary matched-update comparison (e.g. 32 updates for every size), reporting unique-example coverage and tokens; it does not establish completion of the larger deletion requests. Do not pool it with the one-epoch analysis.

## Evaluation: efficacy, utility, safety, degeneration

For a dataset D, let L(M,D) be the mean per-example assistant-token NLL, length-normalized within each response. Use identical serialization and masks for every compared model.

| Axis | Required measurements and interpretation |
|---|---|
| Forgetting efficacy | Report ΔL_f = L(M_UL,F) − L(M0,F), normalized target likelihood, prompt→target ROUGE-L, and prefix→suffix extraction (ROUGE-L and exact match). For extraction, fix the first 50% of response tokens as prefix and score only the held-out suffix; report short-response eligibility and fixed decoding. Higher NLL/lower overlap indicate suppression, not by themselves successful unlearning. |
| Selectivity | Report ΔL_r on R_eval and ΔL_f − ΔL_r; compare generation changes on matched retain examples. FLAN task accuracy/EM where applicable helps distinguish general task damage from reduced verbatim recall. |
| Base diagnostic | Optional return-to-base ratio: [L(M_UL,F)−L(M0,F)]/[L(M_base,F)−L(M0,F)]. Report only with a positive, non-negligible denominator (predeclare 0.01 nats/token); otherwise N/A. Values near 1 indicate matching this likelihood level only. Base is neither a retain-only oracle nor a guaranteed nonmember. |
| Local/general utility | R_eval NLL and generation/task scores; final IFEval instruction/prompt strict accuracy and MT-Bench with fixed judge/version/settings. Optional MMLU and GSM8K. Report raw scores and changes from M0 separately, with fixed evaluation manifests and contamination audit. Reserve separate development prompts for pilot selection. |
| Safety | Reuse the repository’s formal-300 HEx-PHI manifest/checksum and two-dimensional judge contract: safety = refusal / safe_non_refusal / harmful_assistance; quality = normal / degenerate. Pin judge model, actual returned version, prompt version, decoding, and retry policy. Report harmfulness, refusal, safe non-refusal, category breakdown, ΔH from M0, and paired safe→harmful/harmful→safe transitions. |
| Degeneration | Independently report repetitive/gibberish, empty/off-topic outputs, output length and truncation, plus harmful-among-nondegenerate with its denominator. Show safety × quality cross-tabs. Reduced harmfulness accompanied by collapse is not improved alignment. |

Do **not** rename these metrics TOFU FQ or copy TOFU MU. [TOFU evaluation](../../evaluation.md) uses an oracle-referenced truth-ratio KS test and a benchmark-specific utility aggregate; FLAN lacks those constructions. A nonsignificant KS p-value does not establish equivalence. Report efficacy and utility as separate dimensions rather than an invented universal scalar.

Follow the [existing safety contract](experiment2_llama31_8b.md) and five-seed report: judge failures/policy-blocked rows remain unknown, never model refusals. Report coverage, successful-row rates, and full-300 harmfulness bounds [h/300, (h+unknown)/300]; these are missing-label bounds, not confidence intervals. Audit a blinded stratified sample manually. Stage-0 checkpoint selection on HEx-PHI makes subsequent safety results conditional on that selection; confirm key findings on an independent safety set if available.

Report every seed, mean ± sample SD, paired prompt-level bootstrap intervals, and seed-level paired differences. Plot forgetting versus utility, safety, and degeneration across size/checkpoints; do not treat all seed-prompt pairs as independent.

## Member versus nonmember control

Construct nested N_1K ⊂ N_5K ⊂ N_10K from newly authored benign instruction-response pairs created after the pinned model release. Audit exact/near overlap with available training corpora; record provenance and qualify unverifiable membership. A held-out FLAN row alone is not a confirmed nonmember.

Match task, language, prompt/response length, difficulty, and baseline NLL as closely as feasible; report residual mismatches. Apply the same frozen NPO and NPO+retain settings, R_reg, budgets, and seeds. Start at 5K, then extend to all sizes for the main causal comparison (30 additional runs at five seeds). Keep an untouched nonmember evaluation pool for optional MIA calibration.

Compare paired safety/utility drift and suppression between member and nonmember training. Similar drift supports a generic negative-training explanation; differences are suggestive, not definitive causal proof because matching may be imperfect. MIA (e.g. loss/Min-K AUROC and TPR at fixed FPR with confidence intervals) is a secondary privacy diagnostic, calibrated on separate matched pools; never use it to establish membership or as the sole success criterion.

## Optional oracle validation and deliverables

If main results warrant the cost, first retrain **one 10K oracle** from the corresponding pre-SFT base on the full SFT mixture excluding F_10K, matching the official recipe. Reproduce an all-data SFT baseline with that recipe to measure reproduction mismatch. Compare unlearned and oracle NLL/extraction distributions, retained/general utility, and safety; report effect sizes/distribution distances and uncertainty, not merely a KS p-value.

For a DPO/Instruct target, an SFT-only oracle is stage-mismatched: replay the corresponding downstream DPO/RLVR stages with audited data (and regenerate model-dependent data where the recipe requires it), or explicitly label the result an approximate SFT-stage diagnostic. The 10K oracle cannot stand in for 1K/5K or prove universal removal of facts shared with pretraining.

Execution order: baseline/provenance audit → 5K pilot and frozen manifest → six seed-0 conditions + 5K nonmember control → five-seed scalability/control matrix → optional 10K oracle. Commit configs, ID/hash manifests, selection decisions, checkpoint checksums, and aggregate tables; keep weights and raw safety generations in the existing artifact storage. This proposal requires new OLMo data/model/evaluation configuration before execution; it does not claim the TOFU pipeline already implements this protocol.

