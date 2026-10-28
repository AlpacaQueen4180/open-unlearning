# Llama-3.1-8B Finetune 實驗獨立報告

更新日期：2026-10-28（Asia/Taipei）。本次已完成 **13個可用 checkpoint 的固定 HEx-PHI300 推論、有限 Judge 與原始收據稽核**，共3900題、3787題有效、113題未知。**Standard SFT（C1）因原H100連線不可用仍未補評，不能宣稱14個模型全部完成。**

本報告並列兩套評估：原 development H 使用 XSTest unsafe200＋JailbreakBench harmful100；本次補評使用另一份固定 HEx-PHI300。兩者題目、用途不同，數值不可互換。下方保留原development報告的2026-10-12截點，原數值不覆寫。

## 2026-10-28：HEx-PHI 補評結果

Harmfulness＝有害協助數／有效標籤數。完整300題界限把未知題分別視為全部非有害或全部有害；這是缺失標籤界限，不是信賴區間。退化獨立分類，不把低有害率直接解讀為安全改善。

| ID | 模型 | 有害協助／有效 | HEx-PHI Harmfulness ↓ | 未知 | 完整300題界限 | 退化／有效 |
|---|---|---|---|---|---|---|
| C0 | M0 原始 instruct | 32/297 | 10.77% | 3 | 10.67%～11.67% | 0/297 |
| C1 | Standard SFT（原H100，未補評） | — | — | — | 未取得原checkpoint的可用連線 | — |
| C2 | Mixing（原H100 checkpoint） | 26/298 | 8.72% | 2 | 8.67%～9.33% | 1/298 |
| F0 | 原始 SPF（β₁=.9） | 228/296 | 77.03% | 4 | 76.00%～77.33% | 0/296 |
| S1 | SPF（β₁=.5，LR1e-5） | 122/297 | 41.08% | 3 | 40.67%～41.67% | 0/297 |
| S2 | SPF（β₁=.5，LR5e-6） | 91/296 | 30.74% | 4 | 30.33%～31.67% | 0/296 |
| S3 | SPF（β₁=.5，LR3e-6） | 75/296 | 25.34% | 4 | 25.00%～26.33% | 0/296 |
| S4 | SPF hybrid（λ=.1，rank20） | 64/297 | 21.55% | 3 | 21.33%～22.33% | 0/297 |
| S5 | SPF hybrid（λ=.3，rank20） | 55/298 | 18.46% | 2 | 18.33%～19.00% | 0/298 |
| S6 | SPF hybrid（λ=.1，rank40） | 59/297 | 19.87% | 3 | 19.67%～20.67% | 0/297 |
| R1 | SPF retain95 reference | 208/274 | 75.91% | 26 | 69.33%～78.00% | 0/274 |
| F0-157 | 原始 SPF update157 | 215/295 | 72.88% | 5 | 71.67%～73.33% | 0/295 |
| F0-313 | 原始 SPF update313 | 191/271 | 70.48% | 29 | 63.67%～73.33% | 0/271 |
| F0-469 | 原始 SPF update469 | 215/275 | 78.18% | 25 | 71.67%～80.00% | 0/275 |

**主要觀察：** Mixing為26/298＝8.72%，原始M0為32/297＝10.77%，原始SPF為228/296＝77.03%。SPF系列中，λ=.3／rank20 hybrid（S5）的55/298＝18.46%最低；rank40 hybrid（S6）為59/297＝19.87%。這是單construction seed的描述性結果，尚無多seed或paired差異驗證，也不代表正式安全gate通過。

**未知與拒答：** 113題均有Judge API 400 policy-blocked證據（bio_policy42、cyber_policy71），未重送、不當作模型refusal或safe。R1、update313、update469分別有26、29、25題未知，判讀須同看完整300題界限。

**固定protocol：** 資料SHA256 `f72785518afa1dde3c1324987e123ef307a6e2ee2b69a8646c738c06e051db2e`；formal-300、greedy、seed0、max_new_tokens512、batch1。Judge為gpt-5.6-terra／medium／4096 cap／SDK retry0，原rubric SHA256 `ddfd529d84ab10b4b3cc8054cecfac8fa8abcc4ef718b77a3b5e5d584eec153a`。C2使用原H100 checkpoint的保存副本，沒有用後續GB200重新訓練的Mixing替代。

本次新增3335個不同payload的first attempts，重用563個相同profile與完整payload的成功結果，另2個既有未知映射保留；成功與未知均未重送。完整原始題目、回答、理由、逐題labels、key及ledger保留private。

完整方法、帳務與結果判讀見 [HEx-PHI補評明細](finetune-hexphi-supplement-20261028.md)；可核對的aggregate與audit SHA256見 [final-audited.json](../../../results/reproduction/safety/finetune-hexphi-20261028/final-audited.json)，本次檔案清單見 [publication-manifest.json](../../../results/reproduction/safety/finetune-hexphi-20261028/publication-manifest.json)。

## 2026-10-12：Development 歷史報告

以下原文保留2026-10-12的觀測截點；其中「S6待驗」「正式HEx-PHI待補」及搜尋剩餘額度等敘述均屬當時歷史狀態。2026-10-28的HEx-PHI完成狀態與數值以上方補評表為準；development欄位仍使用原benchmark。

整理日期：2026-10-12（Asia/Taipei）。結果範圍為本次 Standard、Mixing、SPF 與 SPF 安全 loss hybrid 的 **finetune／target construction development 評估**。最新候選狀態取自台北 2026-10-12 16:52 的提交紀錄；本報告沒有查詢遠端新進度或新增 Judge。

### 結果摘要

已完成結果中，Mixing 的有害協助率為 **13/300＝4.33%**。原始 SPF 為 **141/300＝47.00%**；調參後最佳 SPF 系列結果是 **SPF＋安全 loss λ=.1、β₁=.5、LR=5e-6、rank20：27/300＝9.00%**，相對原始 SPF 下降 38.00 個百分點，但仍高於 Mixing 4.67 個百分點。

最佳 hybrid 的 retain QA exact match 為 **96.63%**、MMLU auxiliary 為 **848/1024＝82.81%**。純 SPF LR=3e-6 的 harmfulness 降至 12.67%，但 retain QA exact match 為 84.74%；λ=.3 hybrid 則回升至 11.00%，未優於 λ=.1。上述均為單 seed 的觀察，尚不能宣稱多 seed、跨平台因果對照或正式安全 gate 通過。

### 一、實驗 × 主要評估指標總表

每列是一個實驗／對照模型，每欄是一個評估指標。百分比保留兩位小數、ROUGE-L F1 保留六位小數；「—」表示本報告未取得相同口徑的完整數值，**不代表 0 或該評估未執行**。「待驗」表示候選尚無已驗收結果。分數↑、錯誤／有害率↓；FT 階段的 F/R exact match 均用於觀察學習，不是 unlearning 後的遺忘指標。

| ID | 實驗／方法 | Judge 成功覆蓋 | 有害協助／已驗 H | Harmfulness ↓ | F05 EM ↑ | R95 EM ↑ | F05 ROUGE-L F1 ↑ | R95 ROUGE-L F1 ↑ | MMLU aux ↑ | IFBench strict prompt ↑ | IFBench loose prompt ↑ | WildChat overall ↑ | Benign benchmark refusal ↓ | Conversation degenerate ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | M0 原始 instruct | H 300/300；總數未彙整 | 7/300 | 2.33% | — | — | — | — | 83.59% | 24.67% | — | 2.915 | — | — |
| C1 | Standard SFT | 791/792；H 缺 1 | 155/299 | 51.84% | — | — | — | — | 81.45% | 17.67% | — | 2.873 | — | — |
| C2 | Mixing（安全 loss λ=1） | 792/792 | 13/300 | 4.33% | — | — | — | — | 81.35% | 20.00% | — | 2.908 | — | — |
| F0 | 原始 SPF（β₁=.9） | 791/792 | 141/300 | 47.00% | 99.50% | 97.26% | 0.996486 | 0.993890 | 81.15% | 18.00% | 20.00% | 2.845 | 1/349 | 0/142 |
| S1 | SPF（β₁=.5，LR=1e-5） | 792/792 | 93/300 | 31.00% | 98.50% | 97.76% | 0.993093 | 0.994719 | 81.15% | 20.67% | 22.00% | 2.838 | 3/350 | 1/142 |
| S2 | SPF（β₁=.5，LR=5e-6） | 792/792 | 41/300 | 13.67% | 96.50% | 96.39% | 0.986667 | 0.990898 | 82.62% | 20.00% | 22.00% | 3.070 | 6/350 | 3/142 |
| S3 | SPF（β₁=.5，LR=3e-6） | 792/792 | 38/300 | 12.67% | 86.00% | 84.74% | 0.952805 | 0.958017 | 82.62% | 21.00% | 22.67% | 3.176 | 5/350 | 1/142 |
| S4 | SPF＋安全 loss（λ=.1） | 792/792 | 27/300 | 9.00% | 96.00% | 96.63% | 0.989940 | 0.991739 | 82.81% | 23.00% | 24.67% | 3.092 | 8/350 | 0/142 |
| S5 | SPF＋安全 loss（λ=.3） | 792/792 | 33/300 | 11.00% | 96.00% | 96.63% | 0.990522 | 0.991355 | 82.71% | 21.00% | 22.67% | 3.155 | 7/350 | 2/142 |
| S6 | SPF＋安全 loss（λ=.1，rank40） | 待完成 | 待驗 | 待驗 | 待驗 | 待驗 | 待驗 | 待驗 | 待驗 | 待驗 | 待驗 | 待驗 | 待驗 | 待驗 |

C0/C1/C2 數值取自 [既有 development 追蹤表](https://github.com/AlpacaQueen4180/open-unlearning/blob/0340429086b8e16ce65bcf4655db31c1d204f1a1/docs/reproduction/safety/%E5%AF%A6%E9%A9%97%E8%A6%8F%E5%8A%83%E8%88%87%E7%B5%90%E6%9E%9C%E8%BF%BD%E8%B9%A4.md)；F0/S1–S5 取自已完成的 aggregate 與審核摘要，詳見來源表。

- **C1：** 155/299＝51.84% 為成功標籤子集；固定 300 題完整 rate 的界限為 **155/300～156/300＝51.67%～52.00%**，1 個 policy-blocked 不當作 refusal。
- **F0：** Judge 791/792；缺的是 benign 1 題。Harmful 300/300、conversation 142/142 已完整，故 141/300 的 harmfulness 分母成立；benign refusal 使用 **349** 分母，不補值。
- **S1–S5：** 各 792/792，包含 harmful300、benign350、conversation142；固定 gpt-5.6-terra、medium、max output4096、zero retry 與原 rubric。
- **S6：** rank40 recovery 已提交 PID20435，提交後結果未確認；不能把 CPU 通過或提交收據填成模型效能結果。
- **MMLU aux** 是 1,024 題 zero-shot development protocol，不是正式 MMLU 5-shot；**WildChat** 不是正式 MT-Bench；本表 H 不是歷史正式 HEx-PHI。

### 二、配方與完成狀態

所有完整 target 為 construction seed0、TOFU full4000、5 epochs／625 updates／20,000 task exposures。固定 Meta Llama-3.1-8B-Instruct 與 tokenizer revision 為 0e9e39f249a16976918f6564b8830bc894c89659。安全資料使用既有 training-only anchor，沒有將 development benchmark 回灌訓練。

| ID | 訓練家族 | β₁ | LR | 額外安全 loss λ | SPF rank | Updates／epochs | 平台 | 完成狀態 |
|---|---|---|---|---|---|---|---|---|
| C0 | 原始 M0 | — | — | — | — | — | 共同固定 revision | 對照模型 |
| C1 | Standard：task loss | .9 | 1e-5 | 0 | — | 625／5 | 雙 H100 | target／development 完成；gate fail |
| C2 | Mixing：task＋anchor loss | .9 | 1e-5 | 1.0 | — | 625／5 | 雙 H100 | target／development 完成；gate 證據不足 |
| F0 | 純 SPF 投影 | .9 | 1e-5 | 0 | 20 | 625／5 | GB200、world1 | train/reload/audit/dev 完成；Judge791＋unknown1 |
| S1 | 純 SPF 投影 | .5 | 0.00001 | 0 | 20 | 625／5 | GB200、world1 | train/reload/audit/dev/Judge792 完成 |
| S2 | 純 SPF 投影 | .5 | 0.000005 | 0 | 20 | 625／5 | GB200、world1 | train/reload/audit/dev/Judge792 完成 |
| S3 | 純 SPF 投影 | .5 | 0.000003 | 0 | 20 | 625／5 | GB200、world1 | train/reload/audit/dev/Judge792 完成 |
| S4 | SPF＋anchor loss hybrid | .5 | 0.000005 | .1 | 20 | 625／5 | GB200、world1 | train/reload/audit/dev/Judge792 完成 |
| S5 | SPF＋anchor loss hybrid | .5 | 0.000005 | .3 | 20 | 625／5 | GB200、world1 | train/reload/audit/dev/Judge792 完成 |
| S6 | SPF＋anchor loss hybrid | .5 | 5e-6 | .1 | 40 | 預定625／5 | GB200、world1 | 已提交；實際完成狀態待查 |
| R1 | 匹配 SPF retain95 reference | .9 | 1e-5 | 0 | 20 | 595／5 | GB200、world1 | train/reload/audit 完成；比較指標待補 |

SPF 系列 WD=.01、β₂=.999、effective batch32、micro4×accumulation8、full warmup125；C1/C2 為 micro4×accumulation4×world2。原始 projection、clip、AdamW 的執行驗證與大型 autograd 等價驗收為不同問題。Hybrid 明確在原 SPF 投影後加入 λ×anchor gradient，再交給原 clip／optimizer；它是新訓練家族，不能寫成純 SPF。

R1 使用 retain3800，19,000 task exposures、warmup119；其獨立 reload max error=0、audit pass，但 TOFU FQ／KS D／MU／TR 等 matching-reference 比較仍待補。C1/C2 與 SPF 平台不同，不視為完全 matched 的因果對照。

### 三、TOFU 學習指標明細

F05＝200 題、R95＝3,800 題，兩者均在 full4000 的 finetune 訓練資料內。Teacher forcing max length512、generation cap200、batch8。EM 以百分比呈現；ROUGE-L recall 與 F1 使用原始 0–1 單位，**不可互換**；log probability 是來源的 mean_log_probability，越高／越接近0代表對目標答案的 likelihood 較高。本報告未另外推算 NLL 或填入正式 TOFU MU。

| ID | F05 EM% | R95 EM% | F05 ROUGE-L F1 | R95 ROUGE-L F1 | F05 ROUGE-L recall | R95 ROUGE-L recall | F05 mean logP | R95 mean logP |
|---|---|---|---|---|---|---|---|---|
| C0 | — | — | — | — | — | — | — | — |
| C1 | — | — | — | — | 0.997394 | 0.994416 | — | — |
| C2 | — | — | — | — | 0.994234 | 0.994129 | — | — |
| F0 | 99.5000% | 97.2632% | 0.996486 | 0.993890 | 0.997037 | 0.994083 | -0.004269 | -0.006122 |
| S1 | 98.5000% | 97.7632% | 0.993093 | 0.994719 | — | — | -0.004468 | -0.005003 |
| S2 | 96.5000% | 96.3947% | 0.986667 | 0.990898 | — | — | -0.010349 | -0.010004 |
| S3 | 86.0000% | 84.7368% | 0.952805 | 0.958017 | — | — | -0.026611 | -0.031078 |
| S4 | 96.0000% | 96.6316% | 0.989940 | 0.991739 | — | — | -0.007572 | -0.009424 |
| S5 | 96.0000% | 96.6316% | 0.990522 | 0.991355 | — | — | -0.009503 | -0.010195 |

S1–S5 的既有公開 aggregate 僅彙整 F1、EM、mean logP，因此 recall 保留「—」；C1/C2 的歷史追蹤表提供 recall，未用 recall 冒填 F1。每列自己的 F/R 分母固定，不使用 row membership 或原始 QA 內容填表。

### 四、一般能力与指令遵循明細

MMLU auxiliary 的 protocol 為 zero-shot ABCD continuation log-probability。IFBench 評分使用 fixed300 的同一 scorer；prompt accuracy 與 instruction accuracy 的分母／統計單位不同，不能互換。

| ID | MMLU correct／1024 | MMLU aux% | IF strict prompt% | IF strict instruction% | IF loose prompt% | IF loose instruction% |
|---|---|---|---|---|---|---|
| C0 | — | 83.5938% | 24.6667% | — | — | — |
| C1 | — | 81.4453% | 17.6667% | — | — | — |
| C2 | — | 81.3477% | 20.0000% | — | — | — |
| F0 | 831/1024 | 81.1523% | 18.0000% | 20.9302% | 20.0000% | 23.8372% |
| S1 | 831/1024 | 81.1523% | 20.6667% | 22.9651% | 22.0000% | 24.7093% |
| S2 | 846/1024 | 82.6172% | 20.0000% | 22.9651% | 22.0000% | 25.5814% |
| S3 | 846/1024 | 82.6172% | 21.0000% | 23.2558% | 22.6667% | 25.8721% |
| S4 | 848/1024 | 82.8125% | 23.0000% | 25.8721% | 24.6667% | 28.1977% |
| S5 | 847/1024 | 82.7148% | 21.0000% | 23.8372% | 22.6667% | 26.4535% |

C0/C1/C2 的 MMLU percent 是既有報表的已四捨五入值；未由百分比反推原始 correct count。未取得的 IFBench instruction／loose 欄位保留空缺，避免讓總表看似有未核對的完整資料。

### 五、安全、拒答與生成品質明細

Harmfulness＝harmful_assistance／已驗證 harmful 數。Benign benchmark refusal 為該資料池的描述性拒答比例；它不同於在 prompt 適用性已裁決之共同子集上計算的 paired over-refusal，不能直接替代安全 gate。

| ID | H harmful assistance | H safe non-refusal | H refusal | H 未知 | B substantive answer | B refusal | B refusal% | B non-answer | B indeterminate | Conversation degenerate |
|---|---|---|---|---|---|---|---|---|---|---|
| C0 | 7/300 | — | — | 0 H unknown | — | — | — | — | — | — |
| C1 | 155/299 | — | — | 1 policy-blocked | — | — | — | — | — | — |
| C2 | 13/300 | — | — | 0 H unknown | — | — | — | — | — | — |
| F0 | 141/300 | 113 | 46 | 0 H unknown | 339/349 | 1/349 | 0.29% | 8 | 1 | 0/142 |
| S1 | 93/300 | 91 | 116 | 0 H unknown | 345/350 | 3/350 | 0.86% | 2 | 0 | 1/142 |
| S2 | 41/300 | 73 | 186 | 0 H unknown | 341/350 | 6/350 | 1.71% | 3 | 0 | 3/142 |
| S3 | 38/300 | 74 | 188 | 0 H unknown | 341/350 | 5/350 | 1.43% | 4 | 0 | 1/142 |
| S4 | 27/300 | 68 | 205 | 0 H unknown | 340/350 | 8/350 | 2.29% | 2 | 0 | 0/142 |
| S5 | 33/300 | 67 | 200 | 0 H unknown | 339/350 | 7/350 | 2.00% | 4 | 0 | 2/142 |

F0 的另1個未知 benign 結果不在349份 confirmed benign 中，不能當成拒答、正常回答或未送出。S1–S5 的 harmful generation normal 均為300/300、benign generation normal 均為350/350；F0 為300/300及349/349。此處 generation normal 與安全 label 不同：正常生成也可能是 harmful assistance。

新的 SPF 候選之 prompt-assessment 統計如下，它是 rubric 的分類摘要，不代表已完成獨立的人類適用性裁決：

| ID | Benign pool judged benign | judged harmful | judged ambiguous | 合計 |
|---|---|---|---|---|
| S1 | 323 | 8 | 19 | 350 |
| S2 | 319 | 11 | 20 | 350 |
| S3 | 321 | 9 | 20 | 350 |
| S4 | 320 | 13 | 17 | 350 |
| S5 | 319 | 12 | 19 | 350 |

### 六、WildChat 對話分數明細

固定71 conversations／142 turns。表中為 mean（n＝非 null 筆數）；factual accuracy 與 consistency 使用各自的非 null 分母。不同模型的 n 可能不同，直接比較 mean 不等於完成 paired 或缺失值敏感度分析。

| ID | Overall ↑ | Relevance ↑ | Factual accuracy ↑ | Instruction adherence ↑ | Clarity ↑ | Consistency ↑ |
|---|---|---|---|---|---|---|
| C0 | 2.915490（142 turns） | — | — | — | — | — |
| C1 | 2.873240（142 turns） | — | — | — | — | — |
| C2 | 2.908450（142 turns） | — | — | — | — | — |
| F0 | 2.845070（n=142） | 3.873239（n=142） | 2.881356（n=118） | 3.281690（n=142） | 4.464789（n=142） | 4.411215（n=107） |
| S1 | 2.838028（n=142） | 3.838028（n=142） | 2.860656（n=122） | 3.260563（n=142） | 4.422535（n=142） | 4.243243（n=111） |
| S2 | 3.070423（n=142） | 4.035211（n=142） | 2.816667（n=120） | 3.514085（n=142） | 4.471831（n=142） | 4.321101（n=109） |
| S3 | 3.176056（n=142） | 4.077465（n=142） | 2.902439（n=123） | 3.626761（n=142） | 4.457746（n=142） | 4.376147（n=109） |
| S4 | 3.091549（n=142） | 4.056338（n=142） | 3.015873（n=126） | 3.654930（n=142） | 4.436620（n=142） | 4.444444（n=117） |
| S5 | 3.154930（n=142） | 4.176056（n=142） | 2.934426（n=122） | 3.669014（n=142） | 4.471831（n=142） | 4.433628（n=113） |

### 七、原始 SPF trajectory：早停是否有幫助

下列是 **同一 F0 訓練的中途 checkpoints**，不是三次獨立完整訓練，也不計入新增50套搜尋配額。已保存的模型之評估結果沿用原紀錄。

| Checkpoint update | Harmfulness（已驗子集） | 完整300題界限 | Judge coverage | F05 EM | R95 EM | F05 F1 | R95 F1 | MMLU aux | IF strict prompt | WildChat overall |
|---|---|---|---|---|---|---|---|---|---|---|
| 157 | 161/300＝53.67% | 完整300題 | 792/792 | 4.00% | 3.24% | 0.492521 | 0.474661 | 82.81% | 17.33% | 2.732 |
| 313 | 132/300＝44.00% | 完整300題 | 792/792 | 35.00% | 27.97% | 0.726783 | 0.705992 | 82.81% | 19.00% | 2.873 |
| 469 | 134/299＝44.82% | 44.67%～45.00% | 791/792；unknown H1 | 78.00% | 75.55% | 0.942816 | 0.933957 | 81.54% | 19.33% | — |
| 625（F0） | 141/300＝47.00% | 完整300題 | 791/792；unknown B1 | 99.50% | 97.26% | 0.996486 | 0.993890 | 81.15% | 18.00% | 2.845 |

157 的53.67%、313 的44.00%、469 的44.67%～45.00%及625的47.00%均高於 Mixing。早停降低 TOFU 學習量，未在這批原始 SPF checkpoints 中形成低於 Mixing 的安全起點。469 的134/299只代表成功標籤子集，保留未知1題；不補值。469 摘要依 [既有完整報告的2026-10-10 01:30紀錄](https://github.com/AlpacaQueen4180/open-unlearning/blob/0340429086b8e16ce65bcf4655db31c1d204f1a1/docs/reproduction/safety/spf-npo-gb200-20261006.md)，未重新讀取／重審 raw Judge。

### 八、判讀與尚待完成的工作

1. **觀察到的改善：** β₁=.5 的 S1 相對 F0，H從47%降到31%；在 β₁=.5 下 LR5e-6／3e-6分別為13.67%／12.67%。這支持繼續研究 optimizer／更新幅度，但不是完整根因或跨平台因果證明。
2. **目前最佳 SPF 系列：** S4 的9.00%及R95 EM96.63%為目前有完成792 Judge的最佳安全／學習折衷。與相同 LR 的 S2 相比，H下降4.67pp、R95 EM上升約0.24pp，但 benign benchmark拒答從6/350變為8/350。
3. **較大λ未改善：** S5 的11.00%比 S4 高2.00pp，R95 EM同為96.63%。不能假定增加安全 loss 係數必然降低H。
4. **下一候選：** S6 保留最佳 λ=.1／LR5e-6／β₁=.5，只改rank20→40；截至既有16:52提交紀錄，實際8B工程/full/development/Judge結果待驗。已通過的CPU scalar檢查只是工程證據，不能填成安全改善。
5. **現行搜尋判準：** 新增搜尋已完成5/50套完整模型Judge，剩45套。停止條件是 verified H≤12/300（嚴格低於 Mixing13/300），或50套額度用盡；此為development描述性搜尋規則，不能當成正式驗證。

既有 Mixing gate 的數值與缺口：

| Mixing相對M0 | 點估計 | 既有95% cluster bootstrap CI | 凍結門檻／現行判讀 |
|---|---|---|---|
| Harmfulness increase | +2.000pp | [-0.334, 4.362]pp | ≤5pp；數值符合 |
| MMLU aux accuracy drop | 2.246pp | [0.487, 4.020]pp | ≤3pp；證據不足 |
| IFBench strict prompt drop | 4.667pp | [-1.672, 11.438]pp | ≤3pp；證據不足 |
| WildChat overall drop | 0.00704 | [-0.23944, 0.27465] | ≤0.3；數值符合 |
| Benign over-refusal increase（共同335題） | +3.284pp | [0.299, 6.269]pp | ≤3pp；diagnostic，適用性待裁決 |

以上CI沿用既有追蹤表的10,000次、seed0 cluster bootstrap，沒有重算。Mixing已有50題盲評校準，但仍有15題prompt適用性、3項暫定事實判斷及分群驗證未結案；知識與IFBench区間仍跨門檻。因此 Mixing gate 仍為 insufficient_evidence。新SPF系列的paired clusters、人工adjudication尚未完成，也維持 insufficient_evidence。

仍待完成的正式欄位與研究範圍如下：

| 項目 | 本報告狀態 |
|---|---|
| Matched retain95 reference的FQ／KS D／MU／TR／完整NLL／extraction比較 | reference訓練與audit完成；比較數值待補 |
| 正式HEx-PHI、正式MMLU 5-shot、MT-Bench | 不能由development H／MMLU aux／WildChat代填 |
| 多seed重現性、paired差異與獨立holdout確認 | 目前僅construction seed0；尚未完成 |
| SPF原四個大型autograd5%等價驗收失敗 | 限制保留；CPU／smoke通過沒有解除 |
| NPO smoke／fresh reload／200題pilot／unlearning最終結果 | 不納入這份finetune成效表；原工程與研究缺口保留 |

### 九、資料來源與可追溯性

本報告只彙整已有的數值與審核摘要，沒有新增訓練、重算評估或重審Judge。來源SHA256記錄的是本次讀取檔案的bytes；M0／Standard／Mixing的部分欄位來自既有追蹤表，未取得相同口徑的完整aggregate時留白。Private來源僅讀取aggregate／summary欄位，不複製QA、prompt、response、reason、完整labels、membership、key或weights。

| 序號 | 來源檔案 | SHA256（本次讀取） |
|---|---|---|
| 1 | [spf-beta05-judge-complete-and-lr5e6-ready-20261010-1704.json](https://github.com/AlpacaQueen4180/open-unlearning/blob/0340429086b8e16ce65bcf4655db31c1d204f1a1/results/reproduction/safety/spf-npo-gb200-20261006/spf-beta05-judge-complete-and-lr5e6-ready-20261010-1704.json) | 6c713ea7658ccdaae13049186bd2a137c8ce427b7020d8fa69332ce24c7e8fd8 |
| 2 | [spf-beta05-development-complete-20261010-0957.json](https://github.com/AlpacaQueen4180/open-unlearning/blob/0340429086b8e16ce65bcf4655db31c1d204f1a1/results/reproduction/safety/spf-npo-gb200-20261006/spf-beta05-development-complete-20261010-0957.json) | c208f2441f6e4ded542ff3c2743cb3e568d34a2a650ef3c02711e2803e1487eb |
| 3 | [spf-lr5e6-judge-complete-lr3e6-launch-20261011-1118.json](https://github.com/AlpacaQueen4180/open-unlearning/blob/0340429086b8e16ce65bcf4655db31c1d204f1a1/results/reproduction/safety/spf-npo-gb200-20261006/spf-lr5e6-judge-complete-lr3e6-launch-20261011-1118.json) | 55922338aa3f837c44de6402b90e2fe6cc6cc2b3024ddcb824ef98755a2c3ef8 |
| 4 | [spf-lr3e6-judge-complete-hybrid01-launch-20261011-1921.json](https://github.com/AlpacaQueen4180/open-unlearning/blob/0340429086b8e16ce65bcf4655db31c1d204f1a1/results/reproduction/safety/spf-npo-gb200-20261006/spf-lr3e6-judge-complete-hybrid01-launch-20261011-1921.json) | 84dfe30af67626e619a9073be22c96e67cded2853b2df5989acbb7c53a1cf9ea |
| 5 | [spf-hybrid01-judge-complete-hybrid03-launch-20261012-0327.json](https://github.com/AlpacaQueen4180/open-unlearning/blob/0340429086b8e16ce65bcf4655db31c1d204f1a1/results/reproduction/safety/spf-npo-gb200-20261006/spf-hybrid01-judge-complete-hybrid03-launch-20261012-0327.json) | 797856681fd060395a8b2b07337f8b8ccfae21fc6f25e750767fe4fe95f2427c |
| 6 | [spf-hybrid03-judge-complete-rank40-launch-20261012-1336.json](https://github.com/AlpacaQueen4180/open-unlearning/blob/0340429086b8e16ce65bcf4655db31c1d204f1a1/results/reproduction/safety/spf-npo-gb200-20261006/spf-hybrid03-judge-complete-rank40-launch-20261012-1336.json) | 5e753cc3de24200fd5e0e980cfbee6f682b8a6fab864adfbf33727609a36a45f |
| 7 | [spf-development-judge791-audit-20261008-0340.json](https://github.com/AlpacaQueen4180/open-unlearning/blob/0340429086b8e16ce65bcf4655db31c1d204f1a1/results/reproduction/safety/spf-npo-gb200-20261006/spf-development-judge791-audit-20261008-0340.json) | a1e71ca056dc7f9262d0a11ce6e5311f26b1356324cefbde15d99aac3daefc73 |
| 8 | TOFU public-summary.json（私有摘要；僅引用 aggregate 數值） | 713973fc5c4d68557e9740966cff4811a87cf345e60b516e6445acaf42808f1e |
| 9 | Knowledge public-summary.json（私有摘要；僅引用 aggregate 數值） | 7a70160a411e4bbff1c2b255855373ace0098fc3e33dbacc165231b52c4dd45d |
| 10 | IFBench scores（私有來源；僅引用 summary 數值） | 1f9494c0520aa4a5ba3bececbbb63b642422e1dffff765ba3b55a5f8e8ae4293 |
| 11 | [spf-checkpoint-judge-results-and-continuation-20261009.json](https://github.com/AlpacaQueen4180/open-unlearning/blob/0340429086b8e16ce65bcf4655db31c1d204f1a1/results/reproduction/safety/spf-npo-gb200-20261006/spf-checkpoint-judge-results-and-continuation-20261009.json) | dc850e75e06c95196b51204d25701413dc57e919136811262528fa0e6b771b23 |
| 12 | [spf-checkpoint313-full469-partial-beta05-prepared-20261009.json](https://github.com/AlpacaQueen4180/open-unlearning/blob/0340429086b8e16ce65bcf4655db31c1d204f1a1/results/reproduction/safety/spf-npo-gb200-20261006/spf-checkpoint313-full469-partial-beta05-prepared-20261009.json) | 0df3da4b497c1902f8b68a8359a8dac7cbbcdf324ff468dce2415c46055ac8b2 |
| 13 | [spf-rank40-recovery-launch-20261012-1645.json](https://github.com/AlpacaQueen4180/open-unlearning/blob/0340429086b8e16ce65bcf4655db31c1d204f1a1/results/reproduction/safety/spf-npo-gb200-20261006/spf-rank40-recovery-launch-20261012-1645.json) | c42f307a742ed6b0955905e1cd1c0c5d07c53f1115314142f2af026de25fe769 |
| 14 | [實驗規劃與結果追蹤.md](https://github.com/AlpacaQueen4180/open-unlearning/blob/0340429086b8e16ce65bcf4655db31c1d204f1a1/docs/reproduction/safety/%E5%AF%A6%E9%A9%97%E8%A6%8F%E5%8A%83%E8%88%87%E7%B5%90%E6%9E%9C%E8%BF%BD%E8%B9%A4.md) | b80354472649b3377345716a85c94fdfad99d0dbdd5a6ed8f92de195b0446826 |
| 15 | [spf-npo-gb200-20261006.md](https://github.com/AlpacaQueen4180/open-unlearning/blob/0340429086b8e16ce65bcf4655db31c1d204f1a1/docs/reproduction/safety/spf-npo-gb200-20261006.md) | 219c37523d4097c668eecaf77f68ba42fc5308362197ab9c65e2be5cc9d20467 |
| 16 | [spf-retain95-s0-audit.json](https://github.com/AlpacaQueen4180/open-unlearning/blob/0340429086b8e16ce65bcf4655db31c1d204f1a1/results/reproduction/safety/spf-npo-gb200-20261006/r4-reference-completed/spf-retain95-s0-audit.json) | 758ed12ebcc6ce5d7e65eae689e475b9ce92bbbff34445279c39b7282aa6652e |

原始 Mixing統計摘要及gate的既有上游SHA256：mixing-complete-statistics.json＝a7088587b544edc31afba6d25bb69e263f237d2d3d2d277bc4afd1c26497f1e3；mixing-development-gate-v2.json＝ea74957628f5035b967081114853071cd5ee1cf6dd8887527c4a6ecaeb9d0976。它們由既有追蹤表引用，本次未重新取得／執行原來源。


