# forget01／forget05／forget10：已凍結結果比較

整理日期：2026-10-21（Asia/Taipei）。本表只讀已接受的 aggregate；未重訓、重跑推論或呼叫 Judge，未修改任何原 freeze／分數／收據。

**主要發現：PureNPO 在 forget05／10 都出現嚴重退化；retain loss 能保住能力，但未同時保住安全性。forget01 的能力與安全表現相對穩定，但仍不能宣稱與 retraining 等價。**

NPO列是 UL0/1/2 三個 seed 的 mean ± sample SD；只有一個 Mixing construction seed0。H=harmful assistance，D=degenerate。H／D 使用每個 seed 全300題的上下界再取平均，包含缺值；安全性 indeterminate 另納入 H 上界。這些上下界不是信賴區間。

## 完整指標總表（含 Standard 與 Mixing baseline）

Standard 定義為原始 instruct M0，沿用歷史參照；沒有經過本次三種 split 的 FQ／MU 評估，故以「—」標示。Mixing 是同一個初始 checkpoint；FQ 必須分別對 retain99／95／90 reference 計算，因此逐 split 列出。forget01與10共用同一份新 Mixing MMLU／Judge評估，並非兩份獨立樣本。

| Split | 模型 | FQ p（UL0／1／2） | MU native | MU corrected diagnostic | MMLU (%) | MTBench (1–10) | Harmfulness (%) | Degeneration (%) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 歷史參照 | Standard（M0 instruct） | — | — | — | 68.300 | 5.770 | 9.67–10.00 | 0.33–0.67 |
| forget01 | Mixing baseline | 0.00127081 | 3.375e-08 | 0.646 | 65.674 | 5.156 | 9.67–10.67 | 0.00–1.00 |
| forget01 | PureNPO | 0.0143015 / 0.00127081 / 0.00301818 | 3.860e-07 ± 4.00e-08 | 0.512 ± 0.031 | 65.636 ± 0.145 | 4.815 ± 0.312 | 8.67–9.67 | 0.11–1.11 |
| forget01 | NPO+retain | 0.0970748 / 0.00301818 / 0.0541411 | 9.930e-08 ± 1.95e-08 | 0.597 ± 0.026 | 65.781 ± 0.089 | 4.812 ± 0.278 | 8.00–8.89 | 0.00–0.89 |
| forget05 | Mixing baseline | 5.8673e-14 | 3.375e-08 | 0.646 | 65.650 | 5.169 | 9.00–9.00 | 0.00–0.00 |
| forget05 | PureNPO | 0.793362 / 0.393527 / 0.142075 | 1.958e-06 ± 2.15e-06 | 0.019 ± 0.016 | 59.972 ± 1.657 | 1.023 ± 0.004 | 5.33–6.00 | 99.33–100.00 |
| forget05 | NPO+retain | 0.628431 / 0.393527 / 0.793362 | 1.230e-08 ± 8.34e-10 | 0.651 ± 0.011 | 64.312 ± 0.985 | 3.294 ± 0.736 | 46.11–46.78 | 45.56–46.22 |
| forget10 | Mixing baseline | 1.66232e-21 | 3.375e-08 | 0.646 | 65.674 | 5.156 | 9.67–10.67 | 0.00–1.00 |
| forget10 | PureNPO | 4.26068e-10 / 5.0026e-05 / 1.06551e-08 | 1.844e-04 ± 3.19e-04 | 2.737e-04 ± 4.74e-04 | 58.945 ± 2.233 | 1.002 ± 0.004 | 0.33–5.78 | 98.89–100.00 |
| forget10 | NPO+retain | 0.00794521 / 0.000774899 / 5.0026e-05 | 2.453e-08 ± 2.31e-09 | 0.677 ± 0.020 | 65.444 ± 0.596 | 3.083 ± 0.110 | 39.44–40.44 | 35.11–36.11 |

NPO列為三個UL seeds的mean ± sample SD；FQ逐seed列值，不平均。Standard／Mixing為單一checkpoint測量，沒有跨seed SD；Standard數值只保留歷史來源精度。Harmfulness／Degeneration顯示全300題上下界；NPO列為三個seed上下界的平均，不是CI，也不把unknown填0。Standard H=29/299、D=1/299且1題blocked，轉成全300題bounds呈現。

MU native保留原retainTR/PARA錯接缺陷；corrected欄僅為診斷。Standard缺少FQ／MU不以0或reference自比p=1補值。Mixing三個split的MU沿用同一matching retain400／RA／WF utility構成，並不表示三份獨立結果。原matching retain references是FQ的比較對象，並非Standard。

forget05 MMLU 是完整 batch1；forget01／10 是完整 batch8。兩個 campaign 的 Mixing baseline 分別為 MMLU約65.65%／65.6744%、MT5.16875／5.15625、H9.00%／9.67–10.67%、D0%／0–1%。Judge 使用同一 gpt-5.6-terra／medium／4096 profile，但為不同次評分，baseline 分數也有差異；不能視為同一份 verdict。

## 遺忘與保留要一起看

F/R 分別為各官方 forget/retain split；ROUGE-L recall 與 EM 都以百分比表示。不同 split 的題目與組成不同，因此此表描述各自任務的結果，不是對同一題集的因果比較。

| Split | 方法 | F ROUGE-L (%) | R ROUGE-L (%) | F EM (%) | R EM (%) | corrected MU（diagnostic） |
|---|---|---:|---:|---:|---:|---:|
| forget01 | PureNPO | 29.327 ± 7.563 | 59.743 ± 4.567 | 0.000 ± 0.000 | 3.123 ± 2.946 | 0.512 ± 0.031 |
| forget01 | NPO+retain | 30.614 ± 5.564 | 73.184 ± 3.591 | 0.000 ± 0.000 | 21.928 ± 5.914 | 0.597 ± 0.026 |
| forget05 | PureNPO | 9.575 ± 6.229 | 11.009 ± 6.956 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.019 ± 0.016 |
| forget05 | NPO+retain | 28.253 ± 2.532 | 65.740 ± 3.051 | 0.000 ± 0.000 | 16.395 ± 3.261 | 0.651 ± 0.011 |
| forget10 | PureNPO | 5.358 ± 5.174 | 5.853 ± 5.463 | 0.000 ± 0.000 | 0.000 ± 0.000 | 2.737e-04 ± 4.74e-04 |
| forget10 | NPO+retain | 31.604 ± 2.046 | 89.421 ± 1.414 | 0.000 ± 0.000 | 63.991 ± 5.049 | 0.677 ± 0.020 |

corrected MU 僅為辨識 native retainTR/PARA 缺陷的獨立診斷，未取代正式 native MU。原始 native 值保留於各 campaign 報告及隨附 aggregate JSON。F EM皆為0不等於參數中的訓練影響已完全消除；仍有非零 ROUGE／teacher probability，且 PureNPO 的低分可伴隨全面能力崩潰。

## retain loss 的同 seed 差異

NPO+retain − PureNPO；這是各 split 內的配對差異。

| Split | ΔMMLU (pp) | ΔMT | ΔR ROUGE-L (pp) |
|---|---:|---:|---:|
| forget01 | 0.145 ± 0.064 | -0.002 ± 0.437 | 13.441 ± 3.075 |
| forget05 | 4.339 ± 0.869 | 2.271 ± 0.734 | 54.730 ± 6.245 |
| forget10 | 6.500 ± 1.906 | 2.081 ± 0.108 | 83.569 ± 5.485 |

## 發現與解讀

1. **forget05／10 的 PureNPO 是能力崩潰警訊。** MT約1、HEx退化約99–100%，同時 retain ROUGE大幅下降。低有害回答率不能解讀為正常能力下的安全改善。forget01 PureNPO 的 MT約4.81且退化接近0，尚未呈現同樣崩潰，但 retain ROUGE也只有約59.7%。

2. **retain loss 對保留能力的幫助一致，對安全性的保護則不足。** 三個 split 的配對 R ROUGE差異都為正。forget05／10 的 MMLU與MT也明顯較PureNPO高，但 H 約46%／40%，高於各自 Mixing約9–10%的baseline，並伴隨約46%／35%退化。forget01 +retain的H約8–9%、MT約4.81，表現相對較完整。這不是額外安全anchor或projection介入實驗。

3. **forget05 的 FQ 看似最好，卻不代表整體遺忘最成功。** 它的六個逐seed p值皆>.05，同時PureNPO幾乎全面退化。FQ只檢查相對matching reference的特定統計分布，不能代替能力、安全或retraining等價驗證。forget01 +retain有兩seed p>.05、一seed約.003；forget10兩方法均<.05。p值不可跨seed平均，也不宜在不同樣本數間當遺忘程度排名。

4. **結果不支持「forget比例越大，所有指標單調越差」。** 例如+retain的MMLU及保留ROUGE在forget10並不比forget05差，但兩者的安全與品質都受損。資料組成、update次數、warmup占比與隨機波動共同改變；目前不能識別是哪個因素造成差異。

## 逐 seed FQ（不平均）

| Split | 方法 | UL0 | UL1 | UL2 |
|---|---|---:|---:|---:|
| forget01 | PureNPO | 0.014301548 | 0.0012708143 | 0.0030181841 |
| forget01 | NPO+retain | 0.097074844 | 0.0030181841 | 0.054141077 |
| forget05 | PureNPO | 0.79336224 | 0.39352743 | 0.14207465 |
| forget05 | NPO+retain | 0.6284308 | 0.39352743 | 0.79336224 |
| forget10 | PureNPO | 4.2606809e-10 | 5.0026004e-05 | 1.0655054e-08 |
| forget10 | NPO+retain | 0.0079452129 | 0.00077489934 | 5.0026004e-05 |

## 可比性與證據界線

- 相同：同一initial Mixing construction seed0、各方法獨立UL0/1/2；LR1e-5、WD.01、beta.1、gamma1、10epochs、micro4/GAS8、原optimizer、warmup6與scale_wrt_gas=False。matching reference分別是獨立retain99／95／90；NPO frozen loss reference始終是initial Mixing。
- 不同：forget01／05／10有40／200／400題，retain3960／3800／3600；實際updates20／70／130、microsteps100／500／1000，tail2／2／4。warmup6占updates30%／8.57%／4.62%。因此不是等更新量的forget比例實驗。
- MMLU：forget05原batch1，forget01／10 batch8。新baseline只差約.02pp仍不能證明所有模型逐值等價；尤其forget01的約.145pp小差異應謹慎解讀。
- Judge：forget05保留原首輪加真人授權一次補送的凍結結果；forget01／10採全模型exact-payload去重，每payload一次，無重送。兩者覆蓋機制不同，故統一展示全分母bounds。forget05六NPO尚缺12列、indeterminate0；forget01／10十二NPO的缺值及indeterminate見下表。
- 僅單construction seed、每方法三UL seeds；沒有跨construction-seed推論、等價檢定或paired cluster CI。本次未讀舊SPF實驗／ledger／結果；只讀forget05已凍結的aggregate。

| Split／方法 | HEx缺值（三seed合計） | 安全性indeterminate |
|---|---:|---:|
| forget01-pure | 9 | 0 |
| forget01-retain | 8 | 0 |
| forget05-pure | 6 | 0 |
| forget05-retain | 6 | 0 |
| forget10-pure | 10 | 39 |
| forget10-retain | 9 | 0 |

實務結論：本配置下，forget01是目前相對較能同時保留能力與安全表現的工作點；forget05／10顯示單靠retain loss不足以維持安全。這是已觀察結果的描述，尚不是最佳超參數或可部署安全性的證明。若要隔離forget比例的效應，下一輪應另設等updates／warmup占比控制與更多construction seeds；本次沒有啟動新實驗。

來源：[forget05 已凍結報告](mixing-npo-forget05-final-20261019.md)，以及 forget01／10 已通過的最終獨立 CPU 驗收。完整 aggregate、逐 seed 指標、配對描述統計及來源摘要 SHA256 見 [aggregate JSON](../../../results/reproduction/safety/mixing-npo-20261021/forget01-05-10-aggregate-tables.json)。

GitHub 僅發布 aggregate；原始 QA／responses／逐題 labels／reasons／weights／CLI／trace／credentials／ledger 均保留 private。原 forget05 freeze、結果與收據未修改。本次整理沒有新增訓練、推論或付費 Judge 呼叫。
