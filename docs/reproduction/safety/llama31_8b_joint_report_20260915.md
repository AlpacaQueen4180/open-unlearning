# Llama-3.1-8B TOFU NPO：TOFU、Safety 與一般能力聯合分析

> **2026-10-06 訓練來源註記：** 本報告的 Llama-3.1 NPO checkpoints 使用原版訓練，未套用 October GB200 更新次數／DS 邊界修正。「10 epochs」為配置，不能直接視為實際完成值。`seed0 修正版`／`aggregate.corrected.json` 只指既有 checkpoint 的 TOFU 評估與 retain reference 修復，沒有重訓。詳見[更新次數問題與逐報告分類](../npo/update-count-mismatch-20261006.md)。

> 產生時間：2026-09-15T19:20:59.147157+08:00<br>
> 分支：`repro/npo-h100-ada-5seed`<br>
> 本報告只包含 aggregate statistics；不包含 HEx-PHI 原文、raw model responses、Judge rationale、API key 或 checkpoint 權重。

## 一、結論摘要

1. **Retain loss 最一致的作用是避免模型 collapse，但不保證恢復 safety。** 公開 full 的 f05／f10 no-retain 均在五個 seeds 全面 degeneration；加入 retain 後，TOFU utility、MMLU 與 MT-Bench 明顯恢復。
2. **一般能力評估確認：低 harmfulness 若伴隨低 MT-Bench 或 degeneration，不能解讀為安全改善。** Collapse 模型的 HEx-PHI harmfulness 接近 0%，但開放式對話能力同時落至接近量表底部。
3. **Safety outcome 具有強烈的 checkpoint、forget ratio 與 seed 依賴性。** 自製 f05 +retain95 的 harmfulness 高於其 full baseline；公開 f01 的 retain 效果接近零；公開 f05／f10 +retain 則平均低於公開 full，但跨 seed 波動明顯。
4. **MMLU 與 MT-Bench 提供互補資訊。** MMLU 測知識／選擇題能力，MT-Bench 對 collapse、重複生成與多輪品質更敏感；因此 TOFU MU 正常並不足以單獨證明模型整體可用。
5. **MT-Bench 分數是 Terra-judged internal comparison。** Judge 為 `gpt-5.6-terra` medium，不能直接當作官方 GPT-4 MT-Bench leaderboard 分數。

## 二、Baseline 與 oracle

| Checkpoint | TOFU MU | MMLU | MT-Bench | Turn 2 | Harmfulness | Degeneration | Blocked/300 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original instruct | — | 0.6830 | 5.77 | 4.67 | 9.70% | 0.33% | 1 |
| 公開 TOFU full | 0.6276 | 0.6684 | 5.47 | 4.47 | 70.89% | 0.00% | 8 |
| 自製 TOFU full | 0.6191 | 0.6646 | 5.24 | 4.29 | 74.90% | 0.00% | 37 |
| 公開 retain95 oracle | 0.6323 | 0.6698 | 5.54 | 4.55 | 75.66% | 0.00% | 33 |
| 自製 retain95 oracle | 0.6390 | 0.6680 | 5.60 | 4.80 | 80.45% | 0.00% | 34 |
| 公開 retain99 oracle | 0.6177 | 0.6656 | 5.21 | 4.28 | 68.52% | 0.00% | 30 |
| 公開 retain90 oracle | 0.6461 | 0.6710 | 4.97 | 4.29 | 68.95% | 0.00% | 23 |

Original instruct 是 safety 參考點；TOFU full 與 retain oracles 已接觸 TOFU fine-tuning data，因此 harmfulness 明顯高於 original。Oracle 的角色是對應 forget split 的 retrain reference，不代表 safety oracle。

## 三、八個 NPO 設定

TOFU／HEx-PHI 使用五個 seeds；MMLU／MT-Bench 使用事前固定的 seeds 0、2、4。`ΔH` 是相對同一起始 full checkpoint 的 successful-row harmfulness 百分點差；`ΔMMLU`、`ΔMT` 同理。

| 設定 | FQ>.05 | MU (5 seeds) | MMLU (3 seeds) | MT-Bench (3 seeds) | Harmfulness (5 seeds) | Degeneration | ΔH | ΔMMLU | ΔMT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 自製 full, f05, no retain | 5/5 | 0.0629±0.0337 | 0.6206±0.0161 | 1.76±0.30 | 45.14±22.01% | 35.92±41.09% | -29.77pp | -0.0440 | -3.48 |
| 自製 full, f05, +retain95 | 5/5 | 0.5935±0.0053 | 0.6497±0.0040 | 3.43±0.49 | 84.95±5.88% | 6.02±13.27% | +10.05pp | -0.0149 | -1.81 |
| 公開 full, f01, no retain | 5/5 | 0.6469±0.0047 | 0.6682±0.0004 | 5.13±0.13 | 65.00±1.04% | 0.00±0.00% | -5.89pp | -0.0002 | -0.35 |
| 公開 full, f01, +retain99 | 5/5 | 0.6461±0.0052 | 0.6687±0.0009 | 4.97±0.07 | 64.08±1.85% | 0.00±0.00% | -6.81pp | +0.0003 | -0.50 |
| 公開 full, f05, no retain | 2/5 | 0.0000±0.0000 | 0.5790±0.0356 | 1.01±0.01 | 0.07±0.15% | 100.00±0.00% | -70.82pp | -0.0894 | -4.47 |
| 公開 full, f05, +retain95 | 5/5 | 0.5859±0.0248 | 0.6409±0.0072 | 3.25±1.16 | 48.68±14.35% | 0.42±0.45% | -22.21pp | -0.0276 | -2.23 |
| 公開 full, f10, no retain | 2/5 | 0.0000±0.0000 | 0.4756±0.0333 | 1.00±0.01 | 0.00±0.00% | 100.00±0.00% | -70.89pp | -0.1928 | -4.47 |
| 公開 full, f10, +retain90 | 3/5 | 0.6158±0.0248 | 0.6319±0.0024 | 2.61±0.47 | 34.56±18.71% | 0.49±0.69% | -36.33pp | -0.0365 | -2.87 |

### 3.1 TOFU forgetting diagnostics

FQ p-value 與 `log10(p)` 依序列出 seeds 0–4；p-value 不做平均。KS 與 retain truth ratio 則報五個 seeds 的 mean±sample SD。

| 設定 | FQ p-value (s0…s4) | log10(p) (s0…s4) | KS | Retain truth ratio |
|---|---|---|---:|---:|
| 自製 full, f05, no retain | 8.78e-02 / 7.13e-01 / 6.28e-01 / 6.28e-01 / 7.93e-01 | -1.06 / -0.15 / -0.20 / -0.20 / -0.10 | 0.082±0.024 | 0.4597±0.0196 |
| 自製 full, f05, +retain95 | 2.21e-01 / 1.42e-01 / 3.28e-01 / 9.24e-01 / 5.45e-01 | -0.66 / -0.85 / -0.48 / -0.03 / -0.26 | 0.090±0.023 | 0.5002±0.0129 |
| 公開 full, f01, no retain | 2.66e-01 / 7.66e-01 / 7.66e-01 / 7.66e-01 / 2.66e-01 | -0.58 / -0.12 / -0.12 / -0.12 / -0.58 | 0.180±0.041 | 0.5190±0.0044 |
| 公開 full, f01, +retain99 | 1.65e-01 / 7.66e-01 / 7.66e-01 / 7.66e-01 / 1.65e-01 | -0.78 / -0.12 / -0.12 / -0.12 / -0.78 | 0.190±0.055 | 0.5190±0.0039 |
| 公開 full, f05, no retain | 1.78e-01 / 4.91e-19 / 2.97e-02 / 3.01e-03 / 6.80e-02 | -0.75 / -18.31 / -1.53 / -2.52 / -1.17 | 0.204±0.143 | 0.4843±0.0817 |
| 公開 full, f05, +retain95 | 6.80e-02 / 7.93e-01 / 3.28e-01 / 5.45e-01 / 8.66e-01 | -1.17 / -0.10 / -0.48 / -0.26 / -0.06 | 0.086±0.028 | 0.4722±0.0144 |
| 公開 full, f10, no retain | 2.42e-02 / 2.81e-01 / 5.41e-02 / 2.42e-02 / 7.95e-03 | -1.62 / -0.55 / -1.27 / -1.62 / -2.10 | 0.099±0.018 | 0.3514±0.0285 |
| 公開 full, f10, +retain90 | 2.99e-02 / 4.16e-01 / 4.37e-04 / 3.22e-01 / 9.35e-02 | -1.53 / -0.38 / -3.36 / -0.49 / -1.03 | 0.093±0.033 | 0.5158±0.0083 |

### 3.2 公開 full：forget-ratio 趨勢

- `forget01` with／without retain 都保有一般能力，兩者差異很小；retain99 並未帶來一致的額外 safety benefit。
- `forget05` no-retain 在 5/5 seeds 全面 degeneration。retain95 能恢復 capability，但 safety 改善幅度隨 seed 變化。
- `forget10` no-retain 同樣在 5/5 seeds collapse。retain90 避免全面崩潰，但既有結果顯示 FQ、harmfulness 與 capability 都有較大 seed variance。

### 3.3 起始 checkpoint 敏感性

自製 full 與公開 full 的 baseline TOFU metrics 接近，但 f05 NPO 動態不同：公開 no-retain 是一致全面 collapse，自製 no-retain 則是嚴重但程度不一的能力損失；with-retain 的 safety 方向也不同。這表示 baseline MU 接近不足以保證相同的 unlearning trajectory。

## 四、跨指標關係

以下 Pearson correlation 是探索性描述，不代表因果；NPO 部分使用固定 seeds 0、2、4。

| 指標組合 | n | Pearson r | 解讀 |
|---|---:|---:|---|
| `model_utility_vs_mmlu` | 24 | 0.773 | TOFU retain utility 與知識能力的一致程度 |
| `model_utility_vs_mtbench` | 24 | 0.839 | TOFU utility 與開放式對話品質的一致程度 |
| `mmlu_vs_mtbench` | 24 | 0.747 | 兩種一般能力 benchmark 的一致程度 |
| `degeneration_vs_mtbench` | 24 | -0.722 | 生成崩潰是否對應低對話品質 |
| `harmful_vs_mtbench` | 24 | 0.736 | 全部 NPO reps 中 harmfulness 與能力的混合關係 |
| `harmful_vs_mtbench_nondegenerate` | 16 | 0.412 | 排除 degeneration≥5% 後的 safety／能力關係 |
| `mmlu_vs_mtbench_all31` | 31 | 0.779 | 全部 baseline、oracle 與 NPO checkpoint |

能力與 harmfulness 的正相關可能主要反映 collapse：失去正常回答能力的模型既無法完成 MT-Bench，也無法提供可判定的 harmful assistance。只有在 degeneration 很低且 capability 可接受的子集中，safety 差異才較適合解讀為 alignment 變化。

## 五、完整性與限制

- 八個 NPO conditions 的 TOFU 與 HEx-PHI 均有 seeds 0–4；一般能力使用固定 seeds 0、2、4，而不是事後挑選結果。
- Formal-300 為 deterministic generation；Judge 點估計只以成功判定列為分母，policy-blocked 必須搭配 coverage／bounds 解讀。
- Baseline／oracle 多數只有單一 checkpoint；尚未量測多個 independently fine-tuned full checkpoints 的 starting-checkpoint variance。
- MT-Bench 使用 Terra structured-output Judge，適合本研究內部比較，但沒有官方 GPT-4 Judge 的 leaderboard 可比性。
- Correlation 樣本包含同一訓練設定的多個 seeds，不應視為彼此完全獨立的 24 個研究條件。

## 六、建議下一步

1. 對 non-degenerate checkpoints 做同 prompt 的 full→NPO label transition 與 paired bootstrap，主攻自製 f05 +retain95 和公開 f05／f10 +retain。
2. 以人工分層抽查校準 Terra HEx-PHI labels：全收 policy-blocked，另抽 harmful、safe、degenerate 與 safe→harmful transitions。
3. 若需要外部 MT-Bench 可比性，使用官方 GPT-4 Judge 對七個 baselines／oracles與固定代表 NPO checkpoints 做校準；不要直接把 Terra 分數對照 leaderboard。
4. 若論文要主張 checkpoint-level robustness，至少再建立數個 independent full fine-tuning seeds，再各自執行固定的 f05 +retain／no-retain。

## 七、Artifacts

- Machine-readable joint analysis：`results/reproduction/safety/llama31_8b_completion_20260915/joint_analysis.json`
- MMLU aggregate：`results/reproduction/capability/llama31_8b_completion_20260915/mmlu_and_mtbench_generation.json`
- MT-Bench aggregate：`results/reproduction/capability/llama31_8b_completion_20260915/mtbench_terra.json`
- 新增 HEx-PHI aggregate：`results/reproduction/safety/llama31_8b_completion_20260915/pending_hexphi_judged.json`
