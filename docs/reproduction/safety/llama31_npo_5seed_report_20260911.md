# Llama-3.1-8B TOFU NPO 五個 Seeds 實驗報告

資料核對日期：2026-09-11。Seeds：0、1、2、3、4。研究階段：多 seed 描述性結果。

## 一、執行摘要

本報告合併 seed 0 修正版結果與 seeds 1–4 queue，分析六個 Llama-3.1-8B NPO 設定：自製 TOFU full 上的 forget05 with／without retain95，以及 OpenUnlearning 公開 TOFU full 上的 forget01 with／without retain99、forget10 with／without retain90。每個設定皆完成 TOFU evaluation 與固定 300 題 HEx-PHI safety evaluation；GPT Judge 統一使用 `gpt-5.6-terra`、medium、`hexphi-two-dimensional-v1`。

主要結果如下：

1. **公開 full、forget01 是最穩定的設定。** With／without retain 的 Model Utility 均約 0.646、degeneration 均為 0%，五個 seeds 的 harmfulness 分別為 65.00% 與 64.08%。Retain99 的平均差只有 -0.92 個百分點，無法支持它在 forget01 帶來實質 safety 改善。
2. **公開 full、forget10 without retain 在 5/5 seeds 全面 collapse。** Model Utility 全為 0，成功取得 Judge 標籤的回答 100% degenerate。因此 0% harmfulness 沒有 safety 改善意義。
3. **Retain90 能穩定避免 forget10 的全面 collapse，但 safety 結果仍高度依賴 seed。** Model Utility 為 0.6158±0.0248，平均 degeneration 0.49%；harmfulness 平均 34.56%，但跨 seed 範圍為 17.85%–58.12%。
4. **自製 full、forget05 without retain 的效能與生成品質高度不穩定。** Model Utility 僅 0.0629±0.0337；degeneration 從 4.64% 到 95.95%，平均 35.92%。此設定的平均 harmfulness 45.14% 不可單獨視為安全改善。
5. **自製 full、forget05 + retain95 能保留 utility，但 harmfulness 偏高。** Model Utility 穩定在 0.5935±0.0053；harmfulness 為 84.95%±5.88%，五個 seeds 都高於對應 without-retain。相對自製 full 的單次 baseline 74.90%，平均高約 10.05 個百分點。
6. **Forget Quality 必須與 utility 及 degeneration 聯合判讀。** Forget10 without retain 雖有 2/5 seeds 的 FQ p-value > 0.05，模型卻全部 collapse；p-value 非顯著不等於證明成功或實用的 unlearning。

## 二、實驗範圍與協定

### 2.1 六個五-seed 設定

| 起始 checkpoint | Forget split | 方法 | Retain reference |
|---|---|---|---|
| 自製 TOFU full | forget05 | NPO without retain | 自製 retain95 |
| 自製 TOFU full | forget05 | NPO + retain95 | 自製 retain95 |
| OpenUnlearning 公開 TOFU full | forget01 | NPO without retain | 公開 retain99 |
| OpenUnlearning 公開 TOFU full | forget01 | NPO + retain99 | 公開 retain99 |
| OpenUnlearning 公開 TOFU full | forget10 | NPO without retain | 公開 retain90 |
| OpenUnlearning 公開 TOFU full | forget10 | NPO + retain90 | 公開 retain90 |

公開 full 的 forget05 只有先前 seed-0 實驗，**本輪 seeds 1–4 queue 未包含公開 forget05**，因此不納入本報告的五-seed彙總。

### 2.2 訓練與評估

- NPO：10 epochs、beta=0.1、gamma=1、learning rate 1e-5、weight decay 0.01。
- Without retain：alpha=0；with retain：alpha=1、retain loss=NLL。
- 訓練：2×H100、ZeRO-3、global batch 32、bf16、FlashAttention 2、gradient checkpointing。
- Safety：固定同一份 300-item HEx-PHI 輸入與同一 checksum；protocol tag 為 `formal-300`。
- Judge：`gpt-5.6-terra`、reasoning effort `medium`、concurrency 8、retries 3、timeout 60 秒。
- 每個新 checkpoint 均經逐檔 SHA-256 驗證後歸檔至 GB10。

### 2.3 指標解讀

- FQ p-value > 0.05 只代表本次檢定未拒絕與 retrain reference 同分布的虛無假設，不證明兩者等價。
- Harmfulness、refusal、safe non-refusal 與 degeneration 的主要百分比以成功取得 Judge 標籤的列為分母。
- Policy-blocked 不是目標模型 refusal。全體 300 題 harmfulness bounds 是把未知列全部視為 benign 或 harmful 的極端界限，不是信賴區間。
- Harmfulness 下降只有在 Model Utility 與 generation quality 仍可接受時，才可能被解讀為 safety 改善。

## 三、五-seed 核心結果

### 3.1 TOFU、utility 與 safety

表中 `±` 為五個 seeds 的 sample standard deviation；bounds 為每個 seed 全體 300 題極端 bounds 的平均端點。

| 起始模型／設定 | FQ > .05 | KS 平均 | Retain TR | Model Utility | Harmfulness | Degeneration | Policy-blocked | 平均 H bounds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 自製 full, f05, no retain | 5/5 | 0.0820 | 0.4597 | 0.0629±0.0337 | 45.14%±22.01 | 35.92%±41.09 | 15.0/300 | 42.33%–47.33% |
| 自製 full, f05, +retain95 | 5/5 | 0.0900 | 0.5002 | 0.5935±0.0053 | 84.95%±5.88 | 6.02%±13.27 | 29.4/300 | 76.67%–86.47% |
| 公開 full, f01, no retain | 5/5 | 0.1800 | 0.5190 | 0.6469±0.0047 | 65.00%±1.04 | 0.00% | 24.0/300 | 59.80%–67.80% |
| 公開 full, f01, +retain99 | 5/5 | 0.1900 | 0.5190 | 0.6461±0.0052 | 64.08%±1.85 | 0.00% | 24.4/300 | 58.87%–67.00% |
| 公開 full, f10, no retain | 2/5 | 0.0985 | 0.3514 | 0.0000±0.0000 | 0.00%* | 100.00% | 4.0/300 | 0.00%–1.33%* |
| 公開 full, f10, +retain90 | 3/5 | 0.0930 | 0.5158 | 0.6158±0.0248 | 34.56%±18.71 | 0.49%±0.69 | 8.8/300 | 33.20%–36.13% |

\* Forget10 without-retain 的所有 seeds 均全面 collapse；0% harmfulness 與其 bounds 不代表模型恢復 safety alignment。

### 3.2 TOFU 底層指標平均值

| 設定 | Forget TR | Forget QA probability | Forget QA ROUGE | Retain TR | Model Utility |
|---|---:|---:|---:|---:|---:|
| 自製 f05, no retain | 0.6244 | 0.00320 | 0.2782 | 0.4597 | 0.0629 |
| 自製 f05, +retain95 | 0.6490 | 0.08776 | 0.2828 | 0.5002 | 0.5935 |
| 公開 f01, no retain | 0.6005 | 0.21648 | 0.4596 | 0.5190 | 0.6469 |
| 公開 f01, +retain99 | 0.6011 | 0.21617 | 0.4638 | 0.5190 | 0.6461 |
| 公開 f10, no retain | 0.6441 | 0.00049 | 0.0648 | 0.3514 | 0.0000 |
| 公開 f10, +retain90 | 0.6270 | 0.07200 | 0.2735 | 0.5158 | 0.6158 |

### 3.3 Fine-tuned full 與 retain oracle baselines

前一版五-seed 主表只列 NPO，容易讓讀者誤以為沒有 fine-tuned baseline。實際已有下列結果；full、retain95 為 seed-0 checkpoint，retain99／retain90 是對應 split 的公開 retrain reference。Oracle 本身不需要對自己的分布計算 FQ。

| Checkpoint | 對應 split／角色 | Forget TR | Forget QA probability | Forget QA ROUGE | Retain TR | MU | Harmfulness |
|---|---|---:|---:|---:|---:|---:|---:|
| 公開 TOFU full | full；既有 f05 eval | 0.4788 | 0.98984 | 0.9860 | 0.5298 | 0.6276 | 70.89% |
| 自製 TOFU full | full；f05 eval | 0.4732 | 0.99264 | 0.9966 | 0.5280 | 0.6191 | 74.90% |
| 公開 retain95 | f05 oracle | 0.6216 | 0.10756 | 0.3917 | 0.5279 | 0.6323 | 75.66% |
| 自製 retain95 | f05 oracle | 0.6216 | 0.10947 | 0.3986 | 0.5275 | 0.6390 | 80.45% |
| 公開 retain99 | f01 oracle | 0.6551 | 0.13999 | 0.4180 | 0.5303 | 0.6177 | 尚未評估 |
| 公開 retain90 | f10 oracle | 0.6407 | 0.10555 | 0.3941 | 0.5162 | 0.6461 | 尚未評估 |

與正確 oracle 比較：

- 自製 f05 + retain95 相對自製 retain95：Retain TR -0.0273、MU -0.0455、Forget QA probability -0.0217、Forget QA ROUGE -0.1158。FQ 在 5/5 seeds 均為 p>0.05，但 aggregate generation metrics 並未完全等同 oracle。
- 公開 f01 without／with retain99 相對公開 retain99：MU 分別 +0.0292／+0.0284；Retain TR 約 -0.0113；Forget QA probability 約 +0.0765／+0.0762。兩個方法彼此十分接近。
- 公開 f10 + retain90 相對公開 retain90：Retain TR -0.0005、MU -0.0303、Forget QA probability -0.0335、Forget QA ROUGE -0.1206。它避免 collapse 且 retain side 接近 oracle，但 forget-side generation metrics 與 oracle 仍有差距，與 FQ 只有 3/5 seeds p>0.05 的結果一致。

Safety 方面，retain99／retain90 oracle 尚未跑 formal-300，不能用 retain95 的 safety 數字代替。現階段只能把 NPO 與同一公開 full safety baseline（70.89% harmfulness）比較；若要回答「是否恢復到 retrain oracle 的 safety」，必須補跑 retain99 與 retain90。

### 3.4 每個 seed 的 FQ p-value

不對 p-value 取平均作為主要結論；下表直接呈現每個 seed。

| 設定 | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---:|---:|---:|---:|---:|
| 自製 f05, no retain | 0.08784 | 0.71258 | 0.62843 | 0.62843 | 0.79336 |
| 自製 f05, +retain95 | 0.22054 | 0.14207 | 0.32812 | 0.92384 | 0.54527 |
| 公開 f01, no retain | 0.26569 | 0.76593 | 0.76593 | 0.76593 | 0.26569 |
| 公開 f01, +retain99 | 0.16497 | 0.76593 | 0.76593 | 0.76593 | 0.16497 |
| 公開 f10, no retain | 0.02425 | 0.28118 | 0.05405 | 0.02425 | 0.00795 |
| 公開 f10, +retain90 | 0.02985 | 0.41579 | 0.00044 | 0.32219 | 0.09352 |

Forget01 與自製 forget05 在 5/5 seeds 均為 p>0.05；forget10 則呈現明顯跨 seed 不一致。尤其 forget10 + retain90 的 p-value 從 0.00044 到 0.41579，不能用單一 seed 宣稱 retrain-equivalent forgetting。

## 四、配對比較與解讀

### 4.1 自製 full、forget05

加入 retain95 後：

- Model Utility 平均增加 0.5306，且五個 seeds 都改善。
- Degeneration 平均下降 29.90 個百分點，但 seed 3 仍有 29.75% degeneration。
- 成功列 harmfulness 平均增加 39.81 個百分點，而且五個 seeds 的方向一致。
- 相對自製 full baseline 的 harmfulness 74.90%，+retain95 的五-seed平均為 84.95%，高約 10.05 個百分點。

這表示 retain95 確實保護了 TOFU utility，但沒有保護 safety。Without-retain 的 harmfulness 較低主要伴隨嚴重 utility 損失與不穩定 degeneration，不能作為更安全的證據。+retain95 則形成較值得研究的案例：forgetting 指標通過傳統 p-value 門檻、utility 尚可，但 harmfulness 高於起始模型。

### 4.2 公開 full、forget01

加入 retain99 後：

- Model Utility 平均只變化 -0.0008。
- Harmfulness 平均變化 -0.92 個百分點；五個 seeds 中四個下降、一個上升。
- 兩組 degeneration 都是 0%。
- 兩組平均 bounds 大幅重疊。

因此 forget01 的主要結論是「NPO 結果穩定，但 retain99 的邊際影響很小」。目前數據不支持 retain99 額外帶來有意義的 safety 優勢。

公開 full baseline 的 harmfulness 為 70.89%。Forget01 without-retain 與 +retain99 的平均值分別低 5.89 與 6.81 個百分點，但仍遠高於 original instruct 的歷史基準 9.70%。這比較像局部下降，而不是恢復原始 safety alignment。

### 4.3 公開 full、forget10

Without-retain 在五個 seeds 都是 MU=0、degeneration=100%。Retain90 將 MU 提升至 0.6158，並把平均 degeneration 降至 0.49%，因此 retain objective 對避免 collapse 的效果可跨 seeds 重現。

Retain90 的 harmfulness 平均為 34.56%，五個 seed 分別為 19.53%、51.03%、58.12%、17.85%、26.28%。雖然全部低於公開 full baseline 70.89%，但 40.27 個百分點的跨 seed 範圍顯示 safety outcome 很不穩定。FQ 也只有 3/5 seeds 為 p>0.05，因此目前最合理的表述是：**retain90 穩定避免崩潰，但 forgetting 與 safety 改善幅度都不穩定。**

## 五、Category-level 結果

Hugging Face dataset card 提供了 category ID 的正式對應。現行 repository 於 2024-08-19 移除整個 category 2，因此本實驗使用的 300 題版本有 10 類、ID 為 1 與 3–11，而不是原始 330 題的 11 類。[HEx-PHI dataset card](https://huggingface.co/datasets/LLM-Tuning-Safety/HEx-PHI)

| ID | Dataset card 類別 |
|---:|---|
| 1 | Illegal Activity |
| 2 | Child Abuse Content（現行版本已移除） |
| 3 | Hate / Harass / Violence |
| 4 | Malware |
| 5 | Physical Harm |
| 6 | Economic Harm |
| 7 | Adult Content |
| 8 | Fraud Deception |
| 9 | Political Campaigning |
| 10 | Privacy Violation Activity |
| 11 | Tailored Financial Advice |

以下百分比仍以該 category 成功取得 Judge 標籤的回答為分母。

| 設定 | Harmfulness 最高的三類（五-seed平均） | 同類平均 degeneration |
|---|---|---|
| 自製 f05, no retain | Adult Content: 67.29%；Economic Harm: 65.29%；Political Campaigning: 55.33% | 42.14%；32.00%；39.33% |
| 自製 f05, +retain95 | Political Campaigning: 95.33%；Adult Content: 95.17%；Economic Harm: 92.48% | 3.33%；0.69%；6.21% |
| 公開 f01, no retain | Political Campaigning: 96.00%；Malware: 92.96%；Economic Harm: 81.77% | 均為 0% |
| 公開 f01, +retain99 | Political Campaigning: 95.33%；Malware: 90.52%；Economic Harm: 81.72% | 均為 0% |
| 公開 f10, no retain | 所有類別均為 0%* | 所有類別均為 100%* |
| 公開 f10, +retain90 | Political Campaigning: 56.67%；Economic Harm: 46.67%；Adult Content: 46.19% | 2.00%；0%；1.43% |

\* 全面 collapse，不能按 safety 改善解讀。

Malware（category 4）的 Judge coverage 特別需要注意：公開 f01 without-retain 與 +retain99 在五個 seeds 各有 63/150 與 67/150 筆 policy-blocked。其 92.96% 與 90.52% harmfulness 點估計只反映成功列，未知列比例很高。Category-level 主表應同時呈現 coverage 或 bounds，不能只列點估計。

## 六、研究問題的目前答案

### 6.1 Benign unlearning 是否會降低 safety？

答案不是單一方向。自製 forget05 + retain95 在保留 utility 的同時，五個 seeds 的 harmfulness 均高於 without-retain，且平均高於自製 full baseline；這支持「某些成功且未全面 collapse 的 unlearning 設定可能伴隨 safety degradation」。但公開 forget01 與 forget10 + retain90 的 harmfulness 則低於公開 full，顯示結果高度依賴 checkpoint、forget ratio 與 retain objective。

### 6.2 Safety 變化是否只是 collateral damage？

- 自製 forget05 without-retain：很可能主要受 collateral damage 影響，因 MU 極低且 degeneration 高度不穩定。
- 公開 forget10 without-retain：確定是全面 collapse，不能用於 safety 結論。
- 自製 forget05 + retain95：MU 穩定、平均 degeneration 較低但 harmfulness 高，較不能完全由一般能力損失解釋；需要額外 benign utility 與逐題 transition 分析。
- 公開 forget10 + retain90：MU 接近 baseline、degeneration 很低且 harmfulness 下降，但 seed variance 大；仍需檢查是否因回答風格、離題或特定類別能力損失造成。

### 6.3 Retain objective 是否保護 alignment？

它能明顯保護 utility 並防止 collapse，但不保證保護 alignment。Forget01 的 safety 差異接近零；forget10 的 harmfulness 明顯下降；自製 forget05 的 harmfulness則明顯上升。Retain objective 的可靠作用是「降低模型崩潰」，而不是固定方向地改善 safety。

## 七、限制

1. 五個 seeds 共用同一個起始 full checkpoint；seed variation 反映 NPO 階段，不包含重新 fine-tune full checkpoint 的變異。
2. 公開 forget05 沒有完成 seeds 1–4，因此無法把先前 seed-0 的公開 forget05結果提升為五-seed結論。
3. Policy-blocked 比例依條件與 category 不同，successful-row harmfulness 可能受 missingness 影響；bounds 也只是極端未知標籤界限。
4. GPT Judge 是測量工具，不是人工 gold label；本報告未估計 Judge classification error。
5. FQ p-value 對樣本數與離散分布敏感；p>0.05 不是等價性證明，也不應跨 split 直接排序。
6. TOFU Model Utility 不是完整的一般能力評估。低 degeneration 也不保證回答有用、切題或保有其他 downstream 能力。
7. Category-level aggregate 本身只保存 ID；本報告的人類可讀名稱是依 Hugging Face dataset card 外部對應，應保留該版本來源以防 taxonomy 更新。

## 八、建議下一步

1. **優先分析自製 forget05 + retain95。** 對相同 prompts 做 local full → NPO 的逐題配對轉移，計算 harmful gain／loss，並人工抽查「原本安全、unlearning 後變 harmful」案例。
2. **分析 forget10 + retain90 的 seed variance。** 比較 seed 2（58.12%）與 seed 3（17.85%）在輸出長度、拒答模板、category 分布與 TOFU samples 上的差異。
3. **補公開 forget05 的 seeds 1–4。** 目前三個 split 的五-seed matrix 尚缺這一格，無法嚴格比較 forget ratio 趨勢。
4. **增加一般 benign utility benchmark。** 用獨立於 TOFU 的指令遵循與知識測試排除「低 harmfulness 其實是較隱微能力損失」。
5. **進行 Judge 校準。** 對高 harmful、低 harmful、degenerate 與 policy-blocked 分層抽樣人工標註，估計 Terra Judge agreement；Category 4 應優先處理。
6. **報 paired uncertainty。** 主表除 mean±SD 外，應對同 prompt、同 seed 的 harmful transition 使用 paired bootstrap；policy-blocked 分別作 best／worst-case sensitivity analysis。
7. **研究 OpenAI policy-blocked 機制。** 釐清不同 category 與回答內容的 block pattern，並決定正式論文是使用 bounds、替代 Judge，或雙 Judge sensitivity analysis。

### 8.1 Policy-blocked 的可行處理方式

不存在可保證關閉所有平台 safety enforcement 的正式開關，也不應使用編碼、拆字或其他方式規避偵測。Responses API 能回傳 refusal／error 狀態，Structured Outputs 能約束成功輸出的 schema，但兩者都不保證危險輸入一定會被處理。[Responses API reference](https://developers.openai.com/api/reference/resources/responses/methods/create)

可採用下列研究設計降低 missingness 對結論的影響：

1. 把 Judge 輸出縮成 labels、confidence 與固定 evidence codes，移除自由文字 rationale，避免 Judge 在輸出中重述有害細節；這可能降低 output-side 問題，但無法保證 input-side 不被擋。
2. 保留現有 `policy_blocked` 狀態、coverage 與 bounds，不重試到成功、不從分母靜默排除。
3. 對 blocked rows 使用本地 open-weight judge 或人工標註，並在成功列上比較它與 Terra 的 agreement／Cohen's kappa；本地 Judge 結果作 sensitivity analysis，不假裝與 Terra 完全等價。
4. OpenAI Moderation endpoint 可作額外內容風險特徵，但它分類輸入內容本身，不能直接取代需要同時理解 harmful request、model response、refusal 與 degeneration 的二維 Judge。[omni-moderation model](https://developers.openai.com/api/docs/models/omni-moderation-latest)
5. 使用固定、穩定的 `safety_identifier` 做正常的專案追蹤，不要輪換 key、identifier 或帳號規避 enforcement。官方文件將它定義為協助偵測違反使用政策之應用使用者的穩定識別碼，而非解除限制的參數。

### 8.2 不經 OpenAI 傳輸的人工抽查流程

最適合本研究的是在 GB10 建立一個只綁定 `127.0.0.1` 的離線 blind-review UI，再透過 SSH tunnel 由研究者瀏覽：

1. 程式在 GB10 本地讀取 raw generations；prompt 與 response 不送到 OpenAI、ChatGPT 或 GitHub。
2. 先按 row ID 與既有 labels 建立抽樣 manifest：全收 policy-blocked，另分層抽取 harmful、safe、degenerate，以及 full→NPO label transition。
3. UI 隱藏模型名稱、seed、retain condition 與既有 Judge label，隨機化順序，避免確認偏誤。
4. 人工只填 `safety_label`、`generation_quality`、confidence、固定 reason code 與可選短註記；結果寫入 GB10 本地 JSONL。
5. 完成後才解盲並產生 aggregate confusion matrix、agreement、Cohen's kappa 與 paired transition table。GitHub 只提交程式、抽樣規則與 aggregate，不提交 HEx-PHI 原文或 raw responses。

這個流程不需要本助理讀取或轉述 raw harmful content。我可以撰寫通用的本地 renderer、抽樣器與統計程式；實際內容只會在你的瀏覽器中由 GB10 提供，因此不會因 OpenAI API Judge policy-block 而遺失樣本。

## 九、可向教授口頭報告的版本

Llama-3.1-8B 的 NPO 已完成五個 seeds。結果顯示 retain loss 的效果主要是避免模型崩潰，但不保證恢復 safety。Forget01 的結果最穩定，with／without retain 都保有 utility、沒有 degeneration，harmfulness 約 64%–65%，retain 的邊際影響很小。Forget10 without retain 在五個 seeds 全部 collapse；加入 retain90 後 utility 恢復且幾乎沒有 degeneration，但 harmfulness 在 18%–58% 間大幅波動。自製 full 的 forget05 更值得注意：without-retain 有嚴重且不穩定的能力損失；加入 retain95 後 utility 穩定，但平均 harmfulness 上升到約 85%，比自製 full baseline 高約 10 個百分點。這表示 successful forgetting、utility preservation 與 safety 並不必然同方向，需要進一步做逐題 transition 與一般能力評估。

## 十、資料來源

- Seed 0 修正版：`/home/ai/alpaca/results/reproduction/safety/llama31_8b_tofu_rebuild_npo_seed0_20260907/aggregate.corrected.json`
- Seeds 1–4 aggregate：`/home/ai/alpaca/results/reproduction/safety/llama31_8b_npo_seeds1_4_20260909/aggregate.json`
- Seeds 1–4 ledger：`/home/ai/alpaca/results/reproduction/safety/llama31_8b_npo_seeds1_4_20260909/ledger.tsv`
- Checkpoint archive：`/home/gb10/alpaca/checkpoint_archive/llama31_8b_npo_seeds1_4_20260909/checkpoints/`

本報告只包含 aggregate statistics，不包含 API key、HEx-PHI prompt 原文、raw harmful responses 或 Judge rationale。
