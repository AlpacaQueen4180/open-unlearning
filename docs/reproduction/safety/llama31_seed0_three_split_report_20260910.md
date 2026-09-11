# Llama-3.1-8B：TOFU NPO 單一 Seed、三個 Forget Split 階段報告

資料核對日期：2026-09-10。訓練 seed：0。研究階段：描述性初步結果。

## 一、摘要

本實驗研究 benign TOFU fine-tuning 與 NPO unlearning 對模型知識保留、遺忘及安全性的影響。主實驗以同一個 OpenUnlearning 公開 TOFU full checkpoint 為起點，比較 forget01、forget05、forget10，各自執行 NPO without retain 與 NPO + 對應 retain split，共六個 unlearning 設定。另列自製 full／retain95 的重建結果，以及自製 full 上的兩個 forget05 實驗，不與公開模型混合計算。

目前主要觀察：

1. TOFU fine-tuning 後，harmful assistance 從 original instruct 的 9.70% 上升至公開 full 的 70.89%；自製 full 也達到 74.90%，呈現相同方向。
2. 公開 full 的 forget05／forget10 without-retain 均發生嚴重 collapse：Model Utility 為 0，且成功取得 Judge 標籤的回答全部 degenerate。其低 harmfulness 不能視為安全改善。
3. 公開 full 的 forget10 + retain90 是目前最明顯的非全面崩潰改善案例：harmfulness 19.53%、degeneration 0%、Model Utility 0.6186。但 refusal 僅 4.04%，改善主要反映在 safe non-refusal，而非恢復原始拒答行為。
4. 自製與公開 full 的核心 TOFU baseline metrics 雖接近，forget05 unlearning 的結果卻差異很大，顯示結果可能對起始 checkpoint 敏感。
5. 所有結論均限於 seed-0；不能推論跨 seed 穩定性，也不能單憑低 harmfulness 宣稱成功 unlearning 或 safety alignment 恢復。

## 二、實驗設計與比較範圍

### 2.1 模型與資料

- 模型家族：Llama-3.1-8B-Instruct。
- 主實驗起点：OpenUnlearning 公開 TOFU full。
- Forget split：forget01、forget05、forget10，分別對應 retain99、retain95、retain90。
- 方法：NPO without retain（alpha=0）；NPO + retain NLL（alpha=1）。
- 補充實驗：由 original instruct 自行 fine-tune full 與 retain95，再於自製 full 執行 forget05 NPO。
- 自製 forget05 的 TOFU reference 使用自製 retain95；公開模型使用對應公開 retain reference，兩者不混用。
- 本報告僅整理 seed-0，不納入其他 seeds，也不混入 Llama-2-7B 或 Llama-3.2-1B 的舊結果。

### 2.2 訓練與評估協定

本輪設定：2×H100、ZeRO-3、global batch 32、bf16、FlashAttention 2、gradient checkpointing；NPO 為 10 epochs、LR 1e-5、weight decay 0.01、beta=0.1、gamma=1。自製 fine-tuning 為 5 epochs、LR 1e-5、weight decay 0.01、warmup 1 epoch。

Safety 使用 formal-300 HEx-PHI，不與 legacy-300 混合。Judge 設定為 gpt-5.6-terra、medium、hexphi-two-dimensional-v1；concurrency 8、retries 3、timeout 60 秒。此處是配置名稱，不宣稱各次請求的後端版本完全相同。

TOFU 指標包括 Forget Quality（FQ）p-value、KS statistic、Forget Truth Ratio、Forget QA probability／ROUGE、Retain Truth Ratio 與 Model Utility（MU）。不同 split 的樣本數不同，FQ p-value 不適合直接用來排列跨 split 優劣。

## 三、Safety 指標與分母

- Harmful assistance（H）：協助有害要求。
- Refusal（R）：拒絕要求。
- Safe non-refusal（S）：沒有拒答，但未被判定為有害協助。
- Indeterminate（I）：安全行為無法確定。
- Degenerate（D）：獨立的生成品質標籤，與 H／R／S／I 可以重疊。
- Harmful-among-normal：僅在 normal 回答中計算有害協助比例；normal=0 時不可定義。

除特別註明外，H／R／S／I／D 百分比的分母都是「成功取得 Judge 標籤的回答數」，不是固定 300。Policy-blocked 是 Judge API 未能判定，不能當作目標模型 refusal 或 safe response。

全體 300 題的 harmfulness bounds 為：下界 H/300，上界 (H+未判定題數)/300。本表的未判定題數由彙總檔中的 policy-blocked 補足。這是未知標籤的極端界限，不是統計信賴區間，也不涵蓋 Judge 誤判。

## 四、主實驗：公開 Full、三個 Split

### 4.1 TOFU 與 Safety 核心結果

| Split | 方法 | FQ p-value | MU | H：筆數／成功數（%） | D（%） | Harmful-among-normal（%） | 全體 H bounds（%） |
|---|---|---:|---:|---|---:|---:|---|
| — | 公開 full baseline | 6.54e-13* | 0.6276* | 207/292（70.89） | 0.00 | 70.89 | 69.00–71.67 |
| forget01 | NPO without retain | 0.2657 | 0.6424 | 177/275（64.36） | 0.00 | 64.36 | 59.00–67.33 |
| forget01 | NPO + retain99 | 0.1650 | 0.6404 | 168/272（61.76） | 0.00 | 61.76 | 56.00–65.33 |
| forget05 | NPO without retain | 0.1779 | 0.0000 | 1/295（0.34） | 100.00 | 不可定義 | 0.33–2.00 |
| forget05 | NPO + retain95 | 0.0680 | 0.5709 | 177/268（66.04） | 0.75 | 65.79 | 59.00–69.67 |
| forget10 | NPO without retain | 0.02425 | 0.0000 | 0/295（0.00） | 100.00 | 不可定義 | 0.00–1.67 |
| forget10 | NPO + retain90 | 0.02985 | 0.6186 | 58/297（19.53） | 0.00 | 19.53 | 19.33–20.33 |

*公開 full 的 TOFU baseline 為既有 forget05 評估設定，非針對三個 split 各自重算的 baseline；跨 split 的 MU 比較應留意評估資料差異。Safety 則使用同一 formal-300。

### 4.2 三種主要 Safety Tags 與 Judge Coverage

| 模型／設定 | H（%） | R（%） | S（%） | I（%） | 成功／300 | Policy-blocked |
|---|---:|---:|---:|---:|---:|---:|
| Original instruct | 9.70 | 88.29 | 2.01 | 0.00 | 299 | 1 |
| 公開 TOFU full | 70.89 | 3.42 | 25.68 | 0.00 | 292 | 8 |
| forget01 without retain | 64.36 | 7.64 | 28.00 | 0.00 | 275 | 25 |
| forget01 + retain99 | 61.76 | 5.51 | 32.72 | 0.00 | 272 | 28 |
| forget05 without retain | 0.34 | 0.00 | 97.29 | 2.37 | 295 | 5 |
| forget05 + retain95 | 66.04 | 1.49 | 32.46 | 0.00 | 268 | 32 |
| forget10 without retain | 0.00 | 0.00 | 98.64 | 1.36 | 295 | 5 |
| forget10 + retain90 | 19.53 | 4.04 | 76.43 | 0.00 | 297 | 3 |

注意：forget05／10 without-retain 的 S 高達 97–99%，但成功判定的回答同時全部 degenerate。因此 S 不能單獨解讀為「提供了有用的安全回答」。Original instruct 的 degeneration 為 1/299（0.33%）。四捨五入後各項可能不恰好加總 100%。

### 4.3 TOFU 底層指標

| Split／方法 | Forget TR | Forget QA probability | Forget QA ROUGE | Retain TR | KS statistic |
|---|---:|---:|---:|---:|---:|
| forget01 without retain | 0.5720 | 0.20974 | 0.4594 | 0.5200 | 0.2250 |
| forget01 + retain99 | 0.5703 | 0.20909 | 0.4712 | 0.5196 | 0.2500 |
| forget05 without retain | 0.5631 | 9.96e-7 | 0.0658 | 未摘錄 | 未摘錄 |
| forget05 + retain95 | 0.6849 | 0.06927 | 0.2800 | 未摘錄 | 未摘錄 |
| forget10 without retain | 0.5928 | 9.01e-6 | 0.0677 | 0.3929 | 0.1050 |
| forget10 + retain90 | 0.6258 | 0.06907 | 0.2767 | 0.5122 | 0.1025 |

公開 forget05 的兩項「未摘錄」是本次讀取的 TOFU_SUMMARY 未直接包含的欄位，不代表數值為零或整體評估失敗。Forget TR 不宜簡化為越低越好，應與對應 retrain reference 的分布共同判讀。

### 4.4 各 Split 判讀

**forget01：沒有觀察到全面 collapse，有害協助小幅下降。**

兩種方法的 MU 約 0.64，D 均為 0。H 相對公開 full 分別下降 6.53 與 9.13 個百分點；全體 bounds 也低於 full baseline。但兩個 NPO 方法的 bounds 互相重疊，尚不能斷定 retain objective 在此設定有穩定 safety 優勢。

**forget05：retain objective 避免全面崩潰，但 safety 改善有限。**

Without-retain 的 MU=0、D=100%，不是成功安全修復。加入 retain95 後 MU=0.5709，仍低於 full 的 0.6276；H=66.04%，比 full 低 4.85 個百分點，但 bounds 與 baseline 重疊，因此對未知樣本的判定仍可能改變改善方向。

**forget10：retain objective 下出現最明顯的非全面崩潰 safety 改善。**

Without-retain 再度全面 collapse；加入 retain90 後 MU=0.6186、D=0，H 降至 19.53%，較 full 下降 51.36 個百分點。其全體 bounds 與 full 明顯分離，不能僅以 policy-blocked 分母偏差解釋此幅度。然而 H 仍高於 original instruct 的 9.70%，refusal 也沒有回到 original 的 88.29%。FQ p=0.02985 表示在傳統 0.05 門檻下仍可檢出與 reference 分布的差異，不能宣稱已完全達成 retrain-equivalent forgetting。

## 五、補充：自製 Full／Retain95 與 Forget05

### 5.1 重建 Baseline

| 模型 | MU | Retain TR | Forget QA probability | Forget QA ROUGE | H（%） |
|---|---:|---:|---:|---:|---:|
| 公開 full | 0.6276 | 0.5298 | 0.98984 | 0.9860 | 70.89 |
| 自製 full | 0.6191 | 0.5280 | 0.99264 | 0.9966 | 74.90 |
| 公開 retain95 | 0.6323 | 0.5279 | 0.10756 | 0.3917 | 75.66 |
| 自製 retain95 | 0.6390 | 0.5275 | 0.10947 | 0.3986 | 80.45 |

兩組自製 baseline 的四項核心 TOFU metrics 絕對差皆小於事前設定的 0.03。H 差異分別為 +4.01 與 +4.80 個百分點，亦符合先前設定的 5 個百分點描述性門檻；但這不是統計等價性檢定，更不代表參數或下游 unlearning 行為等價。

### 5.2 自製 Full 上的 Forget05

| 設定 | FQ p | KS | MU | H（%） | D（%） | H among normal（%） | 成功／300 | H bounds（%） |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 自製 full | 3.08e-12 | 0.365 | 0.6191 | 74.90 | 0.00 | 74.90 | 263 | 65.67–78.00 |
| forget05 without retain | 0.08784 | 0.125 | 0.1138 | 56.79 | 4.64 | 57.30 | 280 | 53.00–59.67 |
| forget05 + retain95 | 0.22054 | 0.105 | 0.5910 | 87.83 | 0.00 | 87.83 | 263 | 77.00–89.33 |

| 設定 | Forget TR | Forget QA probability | Forget QA ROUGE | Retain TR | R（%） | S（%） |
|---|---:|---:|---:|---:|---:|---:|
| 自製 full | 0.4732 | 0.99264 | 0.9966 | 0.5280 | 3.04 | 22.05 |
| forget05 without retain | 0.6516 | 0.00571 | 0.3148 | 0.4269 | 0.00 | 43.21 |
| forget05 + retain95 | 0.6592 | 0.08946 | 0.2831 | 0.4881 | 1.52 | 10.65 |

這三個模型的 I 均為 0；policy-blocked 分別為 37、20、37。自製 retain95 的成功數為 266、blocked=34、H bounds=71.33–82.67%，D=0。

自製 without-retain 的 H 雖下降 18.12 個百分點，但 MU 從 0.6191 降至 0.1138，約損失 81.6%。這顯示 degeneration classifier 沒有捕捉到的能力損失仍可能很嚴重。

自製 +retain95 的 MU 相對保留，但 H 點估計反而上升 12.93 個百分點。其 bounds 與自製 full 略有重疊，因此在所有未知標籤的極端配置下，仍不能斷言上升方向必然成立。

公開與自製 forget05 的 without-retain 分別是全面 collapse 與嚴重 utility 損失；with-retain 的 harmfulness 則為 66.04% 與 87.83%。因此，baseline metrics 接近不足以保證相同的 unlearning 動態。

## 六、目前能下與不能下的結論

可以支持：

- 在目前評估協定中，TOFU fine-tuning 與明顯的 safety degradation 同時出現。
- NPO 的安全表現高度依赖 split、retain objective 及起始 checkpoint。
- Retain objective 在公開 forget05／10 可避免觀察到的全面 collapse，但不保證降低 harmfulness。
- 公開 forget10 + retain90 的 harmful assistance 大幅下降，且未被 Judge 標成 degenerate，是最值得進一步驗證的案例。

不能支持：

- 「forget ratio 越大，safety 越好」：ratio 與 optimizer steps 可能混淆，且 without-retain 在大 split 已崩潰。
- 「FQ p>0.05 就代表成功遺忘」：未拒絕分布相同假設，不等於證明等價；本實驗甚至有 p>0.05 但 MU=0 的案例。
- 「低 harmfulness 就代表 safety alignment 恢復」：需排除 collapse、離題、安全但無用的回答及 Judge 偏差。
- 「retain95 oracle 是安全 oracle」：它是 TOFU 遺忘參照，不是 safety gold standard。
- 「結果在多 seeds 上成立」：本報告只包含 seed-0。

## 七、建議下一步

1. 優先逐題盲審公開 full → forget10 + retain90 的 harmful-to-safe transitions，區分拒答、有用安全替代回答、離題與能力不足。
2. 配對分析相同 HEx-PHI prompts，報告差異及不確定性；對 policy-blocked 保留全體 bounds。
3. 使用對應 retain90 oracle 比較 forget10；不得拿 retain95 oracle 代替。
4. 補一般 benign instruction utility 與獨立 safety benchmark，確認改善是否可泛化。
5. 多 seed 結果另行彙整；控制實際 optimizer steps 後再解釋 split effect。

## 八、資料來源與版本說明

本次以 SSH 唯讀核對 H100 現有彙總檔，未啟動訓練、generation 或 Judge，也未持續輪詢工作。

- 新 queue 與自製模型：`/home/ai/alpaca/results/reproduction/safety/llama31_8b_tofu_rebuild_npo_seed0_20260907/aggregate.corrected.json`。
- 公開 forget05 TOFU：`/home/ai/alpaca/saves/eval/tofu_safety_exp2_Llama-3.1-8B-Instruct_forget05_NPO_no_retain_2xH100_seed0_20260904/TOFU_SUMMARY.json` 與相同前綴的 `NPO_retain95` 目錄。
- Original／公開 full／公開 retain95／公開 forget05 safety：`/home/ai/alpaca/safety_artifacts/summaries/*_formal-300_seed0.json`。
- 修正後 aggregate 已補齊自製 full 與兩個自製 forget05 的 TOFU metrics，取代之前只取得部分指標的版本。
- 本報告只含 aggregate results，不含 API key、HEx-PHI 原文、raw responses 或 Judge rationale。

## 九、可向教授口頭報告的結論

目前完成 Llama-3.1-8B 在三個 forget splits 的 seed-0 NPO 比較。TOFU fine-tuning 後 harmfulness 大幅上升；unlearning 後的方向並不一致。Without-retain 在公開 forget05／10 發生全面崩潰，因此極低 harmfulness 不具安全改善意義。加入 retain 後，forget10 的 harmfulness 降到 19.53%、沒有觀察到 degeneration，且保有較高 TOFU utility，是目前最有潛力的案例。然而其拒答率只有 4.04%，尚不能稱為恢復原始 alignment。自製模型的 forget05 呈現不同結果，也提醒我們需要確認 checkpoint 敏感性與多 seed 穩定性。
