# 安全保留微調與開放模型的機器遺忘實驗規劃

> **2026-10-06 執行狀態註記：** 本文件為研究計畫，沒有已完成 October GB200 更新次數修正的實驗證據。SPF gradient projection 與此更新邊界 policy 是不同修正；完整 epoch／實際 steps要求仍須在執行時驗收。詳見[更新次數問題與逐報告分類](../npo/update-count-mismatch-20261006.md)。

版本 v2　日期 2026 年 9 月 16 日

## 一 研究摘要

本研究評估：在起始模型仍具安全對齊、且確實學過欲移除資料的條件下，benign machine unlearning 是否改變安全行為；這種改變是否伴隨一般能力受損，以及 retain regularization 能否改善三者的取捨。

採取兩條互補路線。**路線 A 以 Llama-3.1-8B 的安全保留微調建立 TOFU target，延續既有 benchmark 並保留可建立 retrain reference 的優勢；路線 B 直接使用官方 OLMo aligned checkpoint，移除已知 SFT members，減少自行建立 full model 的干擾。** 兩條路線分別回答問題，不把不同模型和資料集的分數合併成同一個因果效果。

先做 A 的 target 建立驗證及 B 的 baseline／資料稽核，再在各自符合條件的起點上進行小型 unlearning pilot。確認遺忘評估有辨識力後，才擴展多 seeds、資料規模和機制對照。Safety 變好、變差或沒有可辨識變化，都是可接受結果；不以找到 harmfulness 上升作為選參目標。

### 本版相對原規劃的變更

- 納入實驗二最新 MMLU 與 MT-Bench 結果；一般能力不再只由 TOFU MU 或 degeneration 判斷。
- 新增 Standard SFT、safety-data mixing、SPF 三種 TOFU target 建立方式，並區分建立 target 與 unlearning 兩個階段。
- 將 SPF-specific retrain reference 列為路線 A 的正式驗證項目，避免沿用不匹配的普通 SFT oracle。
- 保留 trajectory、retain coefficient、matched SFT controls；nonmember 先做補充，不立即展開完整矩陣。
- 保留 OLMo nested 1K／5K／10K，但先以 5K pilot 決定是否值得擴展。
- 所有新增門檻、預算與優先順序都是本研究的設計提案；與既有結果分開呈現。

## 二 現有證據與研究缺口

### 實驗二的主要結果

截至 9 月 15 日，八個 NPO conditions 的 TOFU／HEx-PHI 有 seeds 0–4；MMLU／MT-Bench 使用事前固定的 seeds 0、2、4。另有七個 baseline／oracle，共 31 個 checkpoint 的一般能力評估。以下摘錄平均值；完整數據、sample SD 和逐 seed 結果見 [R1]。

| 起點或條件 | TOFU MU | MMLU 百分比 | MT-Bench | Harmfulness 百分比 | HEx-PHI degeneration 百分比 |
|---|---:|---:|---:|---:|---:|
| Original instruct | — | 68.30 | 5.77 | 9.70 | 0.33 |
| 公開 TOFU full | 0.628 | 66.84 | 5.47 | 70.89 | 0.00 |
| 自製 TOFU full | 0.619 | 66.46 | 5.24 | 74.90 | 0.00 |
| 自製 f05 +retain | 0.594 | 64.97 | 3.43 | 84.95 | 6.02 |
| 公開 f01 +retain | 0.646 | 66.87 | 4.97 | 64.08 | 0.00 |
| 公開 f05 +retain | 0.586 | 64.09 | 3.25 | 48.68 | 0.42 |
| 公開 f10 +retain | 0.616 | 63.19 | 2.61 | 34.56 | 0.49 |

安全比例以成功取得 judge 標籤的回答為分母，須搭配 coverage 與全體 bounds。MT-Bench 採 Terra judge，是內部比較分數。不同指標的 seed 數不同，正式跨指標配對只使用共同的 seeds 0、2、4。

**已支持的觀察：** 建立 TOFU full 後，安全行為已大幅改變。Retain 能避免多個條件的全面崩潰，但低 degeneration 和接近 baseline 的 TOFU MU，仍可能伴隨明顯 MT-Bench 損失。

**尚未解決的問題：** 現有結果還不足以確認「在一般能力保留良好時，選擇性遺忘本身破壞安全」。自製 f05 +retain 是值得追蹤的案例，但它同時損失一般對話能力。公開 retain-only oracles 也不是安全的起點，因此 retrain reference 不等於 safety reference。

原 Word 規劃中的早期 9.0／17.66／9.66% 數字只保留為歷史背景；在 checkpoint、judge、prompt 與 decoding 尚未對齊前，不與最新實驗直接合併。[R1–R3]

## 三 研究問題與共同定義

- **RQ1 起點建立：** SPF 能否在實際學會 TOFU 的同時，保留原始模型的安全與一般能力？
- **RQ2 遺忘影響：** 從安全且有學習訊號的起點執行 NPO，安全變化是否出現在明顯一般能力損傷之前？
- **RQ3 正則化：** 在相近遺忘程度下，加入 retain 是否改善安全與一般能力？
- **RQ4 解釋與外推：** 結果是否也出現在官方 OLMo checkpoint；一般正向更新、負向更新和訓練曝光如何影響結果？

F 為 requested forget set；R 為應保留資料。M0 表示路線 A 的 original instruct；M_pre 表示每次 unlearning 真正使用的起始 checkpoint；M_UL 表示 unlearning 後模型。H 為 harmful-assistance rate，U 為各項 utility 分數。

分別報告：

- ΔH_FT = H(M_pre) − H(M0)：建立 target 階段的變化。
- ΔH_UL = H(M_UL) − H(M_pre)：unlearning 階段的變化。
- H(M_UL) − H(M0)：完整流程後與原始模型的差距。

SPF 與 Standard 分支的 ΔH_UL 差異屬於「不同 target 建立流程下的 unlearning 效果比較」。即使學習程度相近，兩組參數結構仍不同；不宣稱已完全隔離「遺忘本身」的因果效果。

## 四 路線 A 以安全保留微調建立 TOFU target

### A1 方法定義與實作核對

先用 meta-llama/Llama-3.1-8B-Instruct；以相同 TOFU full 資料建立 targets，再在固定 forget05／retain95 上執行 unlearning。不要只先訓練 forget05，否則與既有 full-model 設定不同。

| 建立方式 | 簡稱 | 用途 |
|---|---|---|
| 普通 supervised fine-tuning | Standard | 重建可配對的原流程 |
| TOFU loss 加安全示例 loss | Safety mixing | 較簡單的安全保留對照 |
| Safety-Preserving Fine-tuning | SPF | 核心安全起點建立方法 |

此處 SPF 特指 Zhang 等人的方法：每步計算 task 與 safety gradients，偵測衝突時以低秩安全方向修正 task gradient；不是只增加 safety loss，也不是 fine-tuning 完成後才做修補。論文的方法與理論設定見 [R4]。它在 TOFU、目前 optimizer 與分散式環境是否有效，仍需本研究驗證。

作者首頁的 Code 連結目前指向 Safety at One Shot 的儲存庫；另有名為 spf.zip 的作者軟體封存。開始實作前需核對封存內容、版本與 Algorithm 1，不能把另一篇的 post-hoc patching 直接稱為 SPF。[R5–R6]

安全資料採版本固定、來源可追蹤的公開 refusal／安全回答資料；排除與 TOFU、HEx-PHI、utility tests 和 safety development set 的重複。第一輪 SPF anchor 依官方 recipe 固定，記錄 ID 與選取規則；不得依最終 HEx-PHI 分數挑 anchor。Mixing 的安全資料與 SPF anchor 的來源關係須記錄。此比較評估完整 recipe；若要歸因於 projection，另做相同安全資料預算的 ablation。

實作驗收：核對 rank、parameter blocks、衝突判斷、safety gradient 更新頻率、masking、gradient accumulation，以及 projection 與 optimizer／clipping 的先後順序。特別檢查 ZeRO-3 下取到的是完整或分片梯度。先做小步數數值檢查和 GPU memory profile；不直接假設現有兩張 H100 recipe 可原樣支援 SPF。

### A2 公平建立與 target 合格條件

三組從相同 M0 revision 開始，固定 TOFU IDs、資料順序、tokenization、trainable parameters、batch 與 task exposure。Mixing 可採每個 task batch 加一個 safety loss，避免用安全資料取代 TOFU examples；額外計算量與安全 tokens 獨立報告。

先做 construction seed 0。Standard／mixing／SPF 各跑一個固定 recipe；mixing 係數及 SPF rank 的候選值先依官方實作核對後寫入 manifest。若需額外 target 候選，每種方法最多再評估兩個 recipe，選取規則只使用 construction development data。

SPF target 必須同時符合：

1. **有學到資料：** 在 F 與 R 上，相對 M0 有明顯 target-response likelihood／QA ROUGE／extraction 改善。
2. **學習程度可比較：** 與 Standard target 的 QA ROUGE、likelihood、extraction 分布相近，不能只比較 TOFU MU。
3. **安全仍保留：** 獨立 safety development set 的 harmfulness 接近 M0，並檢查 benign over-refusal。
4. **一般能力可用：** MMLU、MT-Bench、IFEval development 評估沒有重大下降；不把正常語句等同任務成功。

操作門檻草案：安全 development 的 ΔH 上界不超過 +5 個百分點；TOFU QA ROUGE 相對 Standard 差距不超過 0.05；主要一般能力 accuracy 下降不超過 3 個百分點；MT-Bench development 下降不超過 0.5 分。這些是待研究者審閱的容忍值，不是文獻公認標準。Likelihood／extraction 的量尺與最低學習訊號需於 baseline audit 後補入預註冊。

門檻與不確定性一起判讀。若資料量不足以確認，標記「證據不足」，不以 p>0.05 當成等價。若所有 SPF targets 都安全但未學會 TOFU，回到 target 建立問題；若學會資料但仍不安全，也不進入「安全起點」主分析。

### A3 Unlearning 核心比較

| 起始 checkpoint | 無更新 baseline | NPO | NPO +retain95 |
|---|---|---|---|
| M_Standard | 必要 | 必要 | 必要 |
| M_SPF | 必要 | 必要 | 必要 |
| M_Mix | 必要 | 若通過 target gate 則加入 | 若通過 target gate 則加入 |

**SPF 在這裡只用於建立 target；主實驗 unlearning 階段不持續套用 SPF。** 否則會把「起點較安全」與「unlearning 當下受到額外保護」混在一起。將 SPF 用於 NPO 的保護屬於後續 mitigation 實驗。

既有公開／自製 full 結果作歷史參考。新的 Standard–SPF 核心比較使用本次相同 construction protocol；除非證明版本、資料與流程相同，不能只把舊 Standard 結果拼到新 SPF 組。

### A4 Retrain reference

對每種進入正式分析的 target recipe，從同一 M0 在 TOFU retain95 上重新訓練：

- Standard target 對應 Standard retain95 reference。
- SPF target 對應 SPF retain95 reference，使用相同安全 anchor、規則與 hyperparameters。
- Mixing target 若進入主分析，對應相同 mixing recipe 的 retain95 reference。

Reference 使用預先定義的 retain-only 訓練 recipe，不為了接近 M_UL 而調參；報告資料減少造成的 steps 差異。公開普通 retain95 可作補充，但不能作 SPF target 唯一的 oracle。

TOFU FQ 可保留原定義：forget truth-ratio distributions 的 KS statistic 與 p-value，對照匹配 reference。另報 QA probability、ROUGE、extraction 與 utility 差距。若尚無匹配 oracle，只報 suppression／efficacy diagnostics，不稱 retrain-equivalent unlearning。

## 五 路線 B 官方 OLMo 已知成員遺忘

### B1 起點與資料來源

先評估 allenai/OLMo-2-1124-7B-SFT、OLMo-2-1124-7B-DPO、OLMo-2-1124-7B-Instruct。依獨立 development safety、utility 與 FLAN 學習訊號選擇；條件相近時優先較早階段。固定 model／tokenizer revision，避免 preview 與修正版混用。[R7]

若沒有合適安全起點，保留結果並暫停此路線；不得因官方名稱含 Instruct 就認定已符合條件。使用 HEx-PHI 作最後確認；若曾用它選模型，明確揭露並用獨立安全評估做確認。

從 allenai/tulu-3-sft-olmo-2-mixture 的 FLAN component 抽樣。官方 card 記載 FLAN 89,982 筆、整個 SFT mixture 939,344 筆；執行時依固定版本重算。[R8]

| Forget set | 原始 FLAN 比例 | 全 SFT mixture 比例 |
|---|---:|---:|
| 1K | 1.11% | 0.11% |
| 5K | 5.56% | 0.53% |
| 10K | 11.11% | 1.06% |

「約 1／5／10%」只指 FLAN component，不是整個 post-training corpus。

### B2 切分與成員證據

- 固定 task／length 分層排序，建立 F_1K ⊂ F_5K ⊂ F_10K；split seed 42。
- 排除 harmful、refusal-training、安全相關及跨集合重複樣本；固定 classifier 與人工抽查 200–500 筆。
- 剩餘 eligible FLAN 分出 R_reg=10K、R_eval=10K、R_dev=2K。這三者與 F_10K 兩兩不重疊，包含 duplicate groups。
- R_reg 只用於 regularization；R_dev 用於選參；R_eval 只用於最終評估。它們仍是原 SFT members，不是 nonmembers。
- 稽核官方 preprocessing、truncation、assistant masking 與後續階段 overlap，保存 ID／hash／篩選原因。無法確認被實際使用的樣本不稱 confirmed member。
- Baseline 先按 task 與答案長度檢查記憶訊號；短分類答案不與長文 verbatim extraction 混成單一成功率。另報事前定義的高低記憶分層，不事後只挑容易成功的例子。

每組對全部 F 計算 NLL；生成評估固定抽 500 筆，包含共用 F_1K subset 與各 size 代表樣本。R_eval 亦保存固定 generation subset。

### B3 執行順序

先做 5K × NPO／NPO+retain 的 pilot，選好設定後凍結，再做 1K／5K／10K。每組從同一官方 checkpoint 開始，不循序對前一個 unlearned model 再刪資料。

正式 scalability 採一完整 epoch，batch 32 約為 32／157／313 updates；實際值依 sampler 和最後 batch 記錄。一 epoch 表示遍歷一次 request，不代表已成功遺忘。

另做 matched-update 敏感度分析，例如各 32 updates，記錄 unique coverage。此對照可能未遍歷 5K／10K；不能把它描述為完成相同 deletion request，也不能宣稱已完全分離 size 與 repeated exposure 的影響。

### B4 Optional oracle

重要結果出現後，先做 10K retain-only oracle：從對應 pre-SFT base，依官方 recipe 在完整 SFT mixture 排除 F_10K 後訓練，並以同 recipe 的 all-data reproduction 校準重現落差。

若 target 是 DPO／Instruct，SFT-only oracle 只能作階段診斷。正式同階段 reference 需重播 downstream stages，稽核重複資料及必要的 model-dependent data regeneration。10K reference 不能替代 1K／5K reference。

## 六 共同 pilot 與訓練軌跡

### NPO 初始設定

每條路線分開選參，不要求 OLMo 與 Llama 共用 LR。以現有 NPO objective 為準：gamma=1、beta=0.1；alpha=0 或 1，retain loss=NLL；forget effective batch=32，retain sampling 1:1，bf16，gradient clipping=1.0。第一輪以 LR {1e-7, 3e-7, 1e-6}、最多一 epoch 為保守起點。

Optimizer 與 reduction 先固定：優先保持已驗證的本地實作，記錄 optimizer 類型、weight decay、scheduler、masking 與 loss 是否以 sequence／token 正規化；不把新平台預設當成既有 recipe。SPF construction 的 optimizer 與 NPO optimizer 分別記錄。

若整個 grid 都沒有可辨識遺忘訊號，先檢查資料、masking 與 baseline 學習程度，再於 development 階段一次擴至 3e-6 或增加預註冊曝光預算。不得因 safety 尚未變差而增加強度。記錄所有失敗與被排除候選。

### 保存與選參

保存 step 0、25%、50%、75%、100%。每個點保存：

- 實際 updates、已見 examples／tokens、unique coverage、elapsed GPU time。
- Forget／retain NLL，固定樣本的 QA／extraction。
- R_dev utility、獨立 instruction-following development score 與生成品質。
- 固定 R_probe 的 KL(M_pre || M_t)，只在 assistant tokens、固定上下文計算。
- Relative weight distance、gradient norm 與實際 optimizer update norm；避免把 gradient norm 誤稱 parameter movement。
- SPF construction 額外記錄衝突率、投影前後 norm 與計算成本。

KL／distance 與 safety 的共變只作機制線索，不單獨證明因果。Weight distance 不跨不同模型大小直接比較。

以遺忘、R_dev、一般能力 development 結果選 operating points。Formal HEx-PHI 不參與 LR、alpha、checkpoint 或 anchor 選擇。Freeze 之後才評估安全軌跡；若成本不足，先做預定 0／50／100% 點，再依事前規則補齊。

同時報告共用設定與 matched-forgetting 比較：在各條路線自己的效能曲線中，依事前定義 forgetting bins 配對，不依 safety 分數挑點。無法達到共同遺忘區間時，直接報告 lack of overlap。

## 七 評估與成功條件

### 四個維度分開呈現

| 維度 | TOFU 路線 A | OLMo 路線 B |
|---|---|---|
| 遺忘 | 匹配 oracle 的 TR／KS／p-value；QA probability、ROUGE、extraction | ΔNLL_f、target likelihood、prompt-response 與 prefix-suffix extraction；依 task 分層 |
| 保留與一般能力 | TOFU MU 底層指標；MMLU、MT-Bench、IFEval | R_eval 表現；MMLU、MT-Bench、IFEval |
| 安全 | HEx-PHI harmfulness、refusal、safe non-refusal、配對轉移 | 相同協定，與各自 M_pre 比較 |
| 生成品質 | Degeneration、離題／空白、長度、截斷及正常任務成功率 | 同左，獨立於 safety label |

MMLU、MT-Bench 與 IFEval 是正式一般能力項目，GSM8K 可選。正式 checkpoint 使用完整測試；pilot 使用分離的 development prompts。新增資料與所有 benchmarks 做 overlap audit。

NLL 定義為每個 example 的 assistant-token mean NLL，再對 examples 平均。ΔNLL_f = L(M_UL,F) − L(M_pre,F)，另報 ΔNLL_r 與 ΔNLL_f−ΔNLL_r。Normalized probability 和 NLL 相關，不算兩份獨立的遺忘證據。

OLMo 的 prefix test 固定前 50% response tokens，僅評 suffix 的 ROUGE-L／exact match；預先定義最低答案長度及 decoding，短分類題另報 task accuracy。新增語意／任務正確性評估，區分改寫與失去原始內容，但不把「忘記某條訓練記錄」等同「所有共享知識都必須答錯」。

可附 return-to-base NLL ratio，但只作診斷；base 不是 retain-only oracle，分母非正或接近零時不報。沒有 oracle 時，結論應稱 selective suppression／forgetting evidence，不宣稱證明訓練影響被完整移除。OLMo 不套用 TOFU FQ 或 TOFU MU 的名稱。

### 預先約定的判讀

| 觀察 | 可支持的說法 |
|---|---|
| 遺忘指標改善、能力保留、品質正常、安全惡化 | 存在安全起點上的 unlearning-associated safety degradation；再以對照檢驗解釋 |
| 遺忘與能力同時下降、安全改變 | 不能排除一般訓練損傷 |
| Harmfulness 降低但能力／品質崩潰 | 不能視為 alignment 改善 |
| SPF 安全但未學會 TOFU | Target 建立未達研究條件 |
| 遺忘有證據且安全無可辨識變化 | 在目前測試與預算內未偵測到安全下降；需報效應界限 |
| 無明顯遺忘、安全穩定 | 不能支持「有效 unlearning 是安全的」 |

實際容忍門檻於 pilot 前預註冊。初稿建議以主要 accuracy 下降 3 個百分點、MT-Bench 下降 0.5 分、degeneration 增加 1 個百分點作為 utility／quality 警戒，並做門檻敏感度分析。這是操作定義，不是證明無損；HEx-PHI 結果不作 unlearning 候選篩選條件。

## 八 安全測量與統計

沿用固定 formal-300 HEx-PHI、checksum、deterministic decoding 與 hexphi-two-dimensional-v1；固定 requested／actual judge model、reasoning effort、prompt version、retry／error policy。安全標籤為 refusal／safe_non_refusal／harmful_assistance，品質標籤為 normal／degenerate。另以 benign prompts 測 over-refusal，避免把一律拒答當成安全且有用。

報告 successful-row rates、judge coverage、category breakdown，以及全體 harmfulness bounds：[h/N, (h+unknown)/N]。Policy-blocked 是未知 judge 結果，不是模型 refusal；bounds 是缺失標籤界限，不是 confidence interval。Harmful-among-nondegenerate 必須附分母。

人工盲評採分層抽樣：safe→harmful、harmful→safe、degenerate、低 MT-Bench／低 degeneration，以及 blocked cases；報 agreement 與 disagreements。Raw safety prompts／responses 不進 GitHub。Formal prompts 不用於 SPF anchors、安全 mixing 或 development。

使用兩種 seed，分開命名：

- construction_seed：建立 Standard／SPF／mixing target 的變異。
- unlearning_seed：從固定 target 執行 NPO 的變異。

第一輪先 construction_seed=0、unlearning_seed=0。正式條件先完成固定起點的 unlearning seeds 0–4；有重要差異時，再新增 construction seeds 1、2，至少以固定 unlearning seed 重複 Standard／SPF 主比較。這兩層結果分開報，不混稱八 seeds 或十五獨立起點。

新正式矩陣的安全與一般能力使用相同五 seeds；若因成本縮至 0、2、4，必須事前決定，配對分析限共同 seeds。報每 seed、mean±sample SD、prompt-paired bootstrap 與 effect size；MT-Bench 以 question 為單位共同重抽兩個 turns。多層結果保留 construction／unlearning／prompt 結構，不把所有 row 當獨立樣本。

事前指定核心 contrast 為 SPF target 上 NPO 相對其 M_pre 的 ΔH，及 matched-forgetting 的 NPO vs NPO+retain；其他跨類別、size、方法比較標註探索性，或採預定多重比較修正。未顯著不等於安全等價；樣本不足時報告可排除的效應範圍。

## 九 機制對照與延伸

### 第一優先的對照

1. **Retain coefficient sweep：** 固定 F、R、LR 和曝光，alpha={0,0.1,1}；必要時另加預定較強係數。TOFU retain90／95／99 屬不同 deletion split，不是 regularization strength。
2. **Matched benign SFT：** 在同一 M_pre，對相同 F 做正向 NLL 訓練，對比 NPO；匹配資料、steps、batch，並另外以 achieved utility loss／實際 update norm 比較。相同 LR 不保證相同更新強度。
3. **Retain-only continued training：** 從 M_pre 在 R_reg 上繼續正向訓練，辨認單獨 retain 更新的影響。它與「從 M0 在 R 上重新訓練的 oracle」是兩種不同操作。

### 第二優先的 nonmember control

只先在 OLMo 5K 做一組補充：從模型發布後新寫、盡量匹配 FLAN task／length／difficulty／baseline NLL 的資料 N 做相同 NPO。用語是 nonmember suppression，不稱成功移除記憶。保留來源與重複稽核；FLAN 的 held-out row 可能仍是 member。

直覺問題是：「對沒有進入訓練的例子做相同負向更新，也會發生類似安全變化嗎？」若兩組 drift 相近，支持一般負向訓練解釋；若不同，仍須考慮 matching 不完全。它不取代 SFT control，也不是本階段必須先造滿 10K×五 seeds 的前提。MIA 僅作獨立校準的次要診斷，不用來證明 membership。

### 第三優先的延伸

有穩健現象後，才加入 unlearning 階段的 safety-retain／KL／SPF-style protection，檢查 matched-forgetting trade-off 及 benign over-refusal。KL reference 若包含 F，也可能抵抗遺忘，必須明確報告。新的 NPO+SPF adaptation 不直接繼承原 SPF 的安全保證。

跨模型先以 Qwen2.5-7B-Instruct 重複路線 A 的代表條件，不跑完整 grid。Dolly／MUSE 暫列後續；新增模型或資料路線不取代安全 baseline 驗證。

## 十 分階段執行與預算

以下只計新的完整訓練 runs；intermediate checkpoints 不另計 run，evaluation、SPF 額外 gradient／SVD、oracle 成本另外記錄。GPU 小時在 smoke profile 後估算，不假設兩條路線成本相同。

| 階段 | 工作 | 第一輪預算 |
|---|---|---:|
| P0 | 重用實驗二資料做配對與 judge 稽核；核對 SPF artifact；OLMo 三個 baseline | 無新完整訓練 |
| P1 | Standard／mixing／SPF 各一個 TOFU target | 3 runs；額外 target 調參另記 |
| P2A | Standard／SPF × NPO／NPO+retain × 3 LR | 12 runs，單 construction／UL seed |
| P2B | OLMo 5K × 兩方法 × 3 LR | 6 runs，單 UL seed |
| P3A | 凍結四個核心 TOFU conditions，擴至五 UL seeds | 共 20 runs，符合設定的 pilot 可計入 |
| P3B | 凍結 OLMo 1K／5K／10K × 兩方法，擴至五 seeds | 共 30 runs，符合設定的 pilot 可計入 |
| P4 | 匹配 A-oracles、construction seeds、SFT／nonmember controls | 依核心結果啟動，另立 manifest |

P2A 只有 M_SPF 通過 target gate 才啟動。Mixing 若合格且要納入 unlearning pilot，額外 6 runs，不算在核心 12 runs。A 的正式 FQ 結論須完成至少 Standard／SPF 各一個 retain95 reference；後續新增 construction seeds 時同步匹配必要 reference。

**優先順序：** 先完成 SPF target gate，因為它最直接延續 TOFU；OLMo 的 baseline 與資料稽核可同階段進行。SPF 無法建立合格 target 時，先釐清問題，再將已通過 gate 的 OLMo 推進。兩條都可行時先完成各自 pilot，不一次投入所有正式矩陣。

最小可行研究包含：最新實驗二整理、SPF target 建立與匹配 oracle、Standard／SPF 上四個 unlearning conditions、一般能力與 safety 聯合評估、至少一組 trajectory／SFT control。OLMo 提供獨立驗證；若只有 OLMo 路線成功，必須降低 oracle-equivalence 的主張並清楚說明範圍。

## 十一 本次審閱決策與交付

建議先確認以下設計，再凍結可執行 protocol：

1. 以 SPF＋TOFU 作優先控制實驗，OLMo 作官方 aligned checkpoint 的互補驗證。
2. 採三種 target 建立方式，但第一輪 unlearning 主比較只包含 Standard／SPF。
3. 正式保留 MMLU、MT-Bench、IFEval，並採上述暫定容忍值；baseline audit 後固定最低 learning／forgetting signal。
4. 保留匹配 SPF oracle 與 construction-seed 驗證；先做小型 SFT control，nonmember 暫為補充。
5. 確認後將本文件轉為 repository 中新的整合規劃，並同步註記舊 OLMo proposal 的關係；歷史實驗結果保留不覆寫。

正式執行需新增 SPF adapter／驗收、TOFU target registry、OLMo dataset split builder、evaluation configs、trajectory logs、pilot selection manifest 與 unified report。發布只包含程式、ID／hash manifests、設定與 aggregate；權重及 raw generations 留在既有 artifact storage。

## 十二 來源與可追溯性

[R1] [實驗二最新聯合報告](https://github.com/AlpacaQueen4180/open-unlearning/blob/repro/npo-h100-ada-5seed/docs/reproduction/safety/llama31_8b_joint_report_20260915.md)，2026-09-15；讀取 blob SHA b6fb502686c830a4dfa4a807ab2a32ac1a872f5c。

[R2] [Machine-readable joint analysis](https://github.com/AlpacaQueen4180/open-unlearning/blob/repro/npo-h100-ada-5seed/results/reproduction/safety/llama31_8b_completion_20260915/joint_analysis.json)；[MMLU aggregate](https://github.com/AlpacaQueen4180/open-unlearning/blob/repro/npo-h100-ada-5seed/results/reproduction/capability/llama31_8b_completion_20260915/mmlu_and_mtbench_generation.json)；[MT-Bench aggregate](https://github.com/AlpacaQueen4180/open-unlearning/blob/repro/npo-h100-ada-5seed/results/reproduction/capability/llama31_8b_completion_20260915/mtbench_terra.json)。

[R3] 研究者提供的《實驗規劃.docx》，檔案修改日期 2026-08-26；以及「安全保留微調」討論，conversation ID 6aa9099d-ab10-83e9-ad63-729db6b7ab13。作為研究脈絡，方法事實另由原始來源核對。

[R4] Zhang et al., [Understanding and Preserving Safety in Fine-Tuned LLMs](https://arxiv.org/html/2601.10141v1)，arXiv:2601.10141v1，尤其 Algorithm 1。本文不將其實驗結論直接外推為 TOFU 或 NPO 的既定結果。

[R5] [作者首頁](https://kevin-zh-cs.github.io/)；其 Code 指向 [Safety at One Shot](https://github.com/Kevin-Zh-CS/safety-at-one-shot)。方法對應仍需執行前核對。

[R6] [SPF 作者軟體封存](https://zenodo.org/records/21289041)，v1，2026-07-10，spf.zip。本規劃只確認封存頁存在，尚未檢驗封存程式。

[R7] [OLMo 2 7B Instruct model card](https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct)。

[R8] [OLMo-specific SFT mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-olmo-2-mixture)。

[R9] [既有 OLMo proposal](https://github.com/AlpacaQueen4180/open-unlearning/blob/repro/npo-h100-ada-5seed/docs/reproduction/safety/olmo_fully_open_unlearning_proposal.md)；本版整合並調整其優先順序，尚未更新遠端文件。
