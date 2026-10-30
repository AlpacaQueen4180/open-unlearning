# Benign controls：已完成成果公開索引

2026-10-30（Asia/Taipei）。本次發布只含已完成的Mixing/NPO逐題分析、clip0/clip1正向控制的訓練與免費評估、兩個clip1控制的最終Judge驗收。新NPO clip1訓練及OLMo不屬本次發布內容。

## 最新閱讀順序

1. [Mixing/NPO逐題配對分析](mixing-npo-paired-analysis-20261029.md)：18個NPO checkpoint；三個相關UL seeds不能當成獨立的900題訓練實驗。
2. [clip0 controls免費結果](benign-controls-free-results-20261030-v3.md)及[原實作勘誤](benign-controls-execution-20261029-v2.md)：原要求clipping1，實際DeepSpeed0；保留歷史protocol deviation。
3. [clip1 controls免費結果](benign-controls-clip1-final-20261030-v3.md)及[訓練/smoke證據](benign-controls-clip1-execution-20261030-v2.md)：有效engine/optimizer clipping1；兩final完整audit與fresh reload max error0。
4. [clip1 Judge最終驗收v2](benign-controls-clip1-judge-final-20261030-v2.md)：527筆新首次請求均已處置，525成功、2筆provider policy拒絕保留unknown、未送0。加歷史exact-payload重用及跨模型去重，共920列中918有驗證標籤。

## 主要已完成結果

| 模型 | MMLU % | TOFU EM F/R % | H full300 % | D full300 % | MT平均 |
|---|---:|---:|---:|---:|---:|
| Mixing | 65.653 | 98.000 / 97.500 | 9.000 | 0 | 5.16875 |
| PureNPO clip0 seed0 | 60.654 | 0 / 0 | 4.333 | 100 | 1.02500 |
| NPO+retain clip0 seed0 | 65.126 | 0 / 14.395 | 36.667–37.000 | 54.667–55.000 | 4.12500 |
| 正向SFT clip0 seed0 | 65.055 | 100.000 / 64.000 | 待Judge | 待Judge | 待Judge |
| retain continued clip0 seed0 | 65.788 | 78.000 / 80.263 | 待Judge | 待Judge | 待Judge |
| 正向SFT clip1 seed0 | 64.991 | 100.000 / 56.289 | 13.000 | 0 | 4.78125 |
| retain continued clip1 seed0 | 65.803 | 76.500 / 81.342 | 17.667–18.333 | 0–0.667 | 4.76875 |

H/D區間為缺值最壞界，不是信賴區間；MT保留80題兩turn結構。單construction/training seed及matched updates只提供機制線索，不匹配tokens/成本/梯度位移。新CE clip1與原NPO clip0另有clipping差異，不能把差異全部歸因objective。native TOFU retainTR/PARA缺陷與corrected diagnostic分開，p>.05不是retraining equivalence。

## 版本與隱私

各舊報告及provenance保留驗收當時的免費/Judge尚未執行、published_to_GitHub=false等快照欄位；現況以本索引及最終Judge v2為準，沒有重寫原freeze或原始收據。clip0兩模型尚無新增Judge授權，不移用clip1或其他campaign預算。

公開只含設定、aggregate、hash及驗證摘要。權重、QA、原始生成、逐題labels/reasons、CLI、憑證及ledger不發布。發布manifest分批保存來源/公開版SHA256與Git blob ID；本機原始檔案不改寫。
