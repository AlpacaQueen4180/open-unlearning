# forget05：M0 → Mixing → NPO 完整有限範圍比較

2026-10-18（Asia/Taipei）；Mixing construction seed0，PureNPO α0／NPO+retain α1 各 UL0/1/2。

六個 NPO 的 matching 原始輸出已在本機以原始數值函式、相同版本統計函式庫重算通過：全部4868 probability sequences／217新generations／400 own-retain generations reuse逐模型核對，最大浮點差異4.44×10⁻¹⁶。Mixing加六NPO共28,000筆TOFU的原bytes／4000 IDs／200F3800R／mean及sample SD通過驗收。

本次固定有限範圍已完成：8個 checkpoint 各460題，原首輪3680次；未知另補送22次，總嘗試3702次。補回8筆後為3666筆有效、14筆仍未知；不再續送。M0另為歷史參照，不計入這8份付費範圍。

Judge：6/6模型已完成有限460逐筆稽核及shared commit，原首輪共2760題；合併一次補送後2748有效＋12未知；未知不當有效標籤；這份表另外合併真人授權的一次補送，原始紀錄保留。

| 模型 | KS D | FQ p-value | TOFU MU（native） | MMLU (%) | MT-Bench (1–10) | HEx H (%) | HEx D (%) | HEx／MT 有效數 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0：原始 instruct（歷史參照） | — | — | — | 68.30 | 5.77 | 9.70 | 0.33 | 299/300；160/160 |
| Mixing-full（construction seed0） | 0.390 | 5.867302e-14 | 3.375204e-08 | 65.65 | 5.16875 | 9.00 | 0.00 | 300/300；160/160 |
| PureNPO α=0／UL0 | 0.065 | 0.793362 | 1.606741e-06 | 60.65 | 1.02500 | 4.33 | 100.00 | 300/300；160/160 |
| NPO+retain α=1／UL0 | 0.075 | 0.628431 | 1.175711e-08 | 65.13 | 4.12500 | 36.79 | 54.85 | 299/300；160/160 |
| PureNPO α=0／UL1 | 0.090 | 0.393527 | 0.000000e+00 | 58.08 | 1.02500 | 5.37 | 100.00 | 298/300；160/160 |
| NPO+retain α=1／UL1 | 0.090 | 0.393527 | 1.188855e-08 | 63.22 | 3.03125 | 60.94 | 30.98 | 297/300；160/160 |
| PureNPO α=0／UL2 | 0.115 | 0.142075 | 4.266057e-06 | 61.18 | 1.01875 | 6.42 | 100.00 | 296/300；160/160 |
| NPO+retain α=1／UL2 | 0.065 | 0.793362 | 1.326251e-08 | 64.59 | 2.72500 | 41.61 | 51.68 | 298/300；160/160 |
| matching-retain95 reference | — | — | — | 66.10 | 5.25000 | 6.71 | 0.00 | 298/300；160/160 |

KS D 越小表示此統計分布越接近 retain95 reference；FQ p-value 不跨 seed 平均。p>.05 不證明訓練影響已移除或與 retraining 等價。MU 主欄保留原 native 算術與已知 retainTR 錯接 PARA 缺陷；獨立 corrected diagnostic 另列。

原首輪 PureNPO 三個 UL seed 的已評 HEx 均100%退化、MT約1.02；補送後數值見主表。低H不能解讀為正常能力下的安全改善。H/D主欄按各自有效標籤計算，請同看下方全分母範圍。

## 原 native 與獨立 corrected diagnostic

| 模型 | retainTR native | retainTR corrected diagnostic | MU native | MU corrected diagnostic |
| --- | --- | --- | --- | --- |
| Mixing-full | 3.750226598e-09 | 0.512439624 | 3.375203786e-08 | 0.645650803 |
| PureNPO α=0／UL0 | 1.785364444e-07 | 0.416171795 | 1.606741477e-06 | 0.029603001 |
| NPO+retain α=1／UL0 | 1.306345536e-09 | 0.496007570 | 1.175710964e-08 | 0.645441927 |
| PureNPO α=0／UL1 | 6.806588279e-07 | 0.430920124 | 0.000000000e+00 | 0.000000000 |
| NPO+retain α=1／UL1 | 1.320950438e-09 | 0.501999378 | 1.188855376e-08 | 0.642974664 |
| PureNPO α=0／UL2 | 4.740809339e-07 | 0.448593324 | 4.266057299e-06 | 0.026941551 |
| NPO+retain α=1／UL2 | 1.473612436e-09 | 0.514201234 | 1.326251169e-08 | 0.663843890 |

corrected diagnostic 僅供辨識原缺陷，沒有取代正式 native 值。PureNPO UL1 的 native與corrected MU皆為原實測0，不是缺值填0。

## 跨 UL seeds 描述統計

| 方法 | 免費 n | MMLU (%) | KS D | MU native | MU corrected diagnostic | Judge n | MT | 有效標籤 H (%) | 有效標籤 D (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PureNPO α=0 | 3 | 59.972 ± 1.657 | 0.090 ± 0.025 | 1.958e-06 ± 2.155e-06 | 0.019 ± 0.016 | 3 | 1.023 ± 0.004 | 5.374 ± 1.043 | 100.000 ± 0.000 |
| NPO+retain α=1 | 3 | 64.312 ± 0.985 | 0.077 ± 0.013 | 1.230e-08 ± 8.338e-10 | 0.651 ± 0.011 | 3 | 3.294 ± 0.736 | 46.448 ± 12.783 | 45.835 ± 12.965 |

均值±sample SD跨UL seeds，只有一個construction seed0。Judge n不足3時屬階段統計，H/D使用各seed有效標籤，不能當完整標籤估計或用SD取代未知上下界。FQ p-value僅逐seed報告。

## 同 UL seed 的 NPO+retain − PureNPO 差異

| UL seed | ΔMMLU (pp) | ΔMU corrected diagnostic | ΔRetain EM (pp) | ΔMT |
| --- | --- | --- | --- | --- |
| 0 | +4.472 | +0.615839 | +14.395 | +3.10000 |
| 1 | +5.135 | +0.642975 | +14.632 | +2.00625 |
| 2 | +3.411 | +0.636902 | +20.158 | +1.70625 |

上述為同一UL seed的完整免費指標差異及已完成MT差異，不是paired cluster CI。原cluster mapping與人類裁決未驗證，不產生或宣稱95% paired cluster CI。

## 未知標籤的全分母上下界

| 模型 | HEx 未知數 | H 全300題 (%) | D 全300題 (%) | MT 全160turn |
| --- | --- | --- | --- | --- |
| PureNPO α=0／UL0 | 0 | 4.333–4.333 | 100.000–100.000 | 1.025–1.025 |
| NPO+retain α=1／UL0 | 1 | 36.667–37.000 | 54.667–55.000 | 4.125–4.125 |
| PureNPO α=0／UL1 | 2 | 5.333–6.000 | 99.333–100.000 | 1.025–1.025 |
| NPO+retain α=1／UL1 | 3 | 60.333–61.333 | 30.667–31.667 | 3.031–3.031 |
| PureNPO α=0／UL2 | 4 | 6.333–7.667 | 98.667–100.000 | 1.019–1.019 |
| NPO+retain α=1／UL2 | 2 | 41.333–42.000 | 51.333–52.000 | 2.725–2.725 |
| mixing-baseline | 0 | 9.000–9.000 | 0.000–0.000 | 5.169–5.169 |
| matching-retain95 | 2 | 6.667–7.333 | 0.000–0.667 | 5.250–5.250 |


## 不需 Judge 的 TOFU 底層診斷

| 模型 | F EM (%) | R EM (%) | F ROUGE-L recall (%) | R ROUGE-L recall (%) | F答案機率 (%) | R答案機率 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Mixing-full | 98.000 | 97.500 | 99.630 | 99.363 | 99.592 | 99.445 |
| PureNPO α=0／UL0 | 0.000 | 0.000 | 14.871 | 16.256 | 0.093 | 0.378 |
| NPO+retain α=1／UL0 | 0.000 | 14.395 | 26.099 | 64.358 | 5.290 | 76.138 |
| PureNPO α=0／UL1 | 0.000 | 0.000 | 2.711 | 3.119 | 0.066 | 0.169 |
| NPO+retain α=1／UL1 | 0.000 | 14.632 | 31.042 | 63.624 | 6.476 | 76.997 |
| PureNPO α=0／UL2 | 0.000 | 0.000 | 11.142 | 13.653 | 0.084 | 0.320 |
| NPO+retain α=1／UL2 | 0.000 | 20.158 | 27.618 | 69.237 | 5.205 | 79.851 |

答案機率為每題assistant token機率幾何平均再對split取平均，不是整段joint probability；ROUGE-L為原recall。底層4000 QA指標與正式matching4868probs是不同驗收項目，均已核原始bytes，不以surrogate取代。

MMLU為完整57科／14,042 test／5-shot／原有效batch1；六NPO與Mixing的原完整輸出已有本機稽核，matching retain95 reference的MMLU原生收據與完整本機原bytes驗收也已PASS（accuracy 66.101694915%）。

M0為2026-09-15歷史formal aggregate，只供參照：H29/299、D1/299，另1題blocked，全300題H範圍9.667–10.000%、D0.333–0.667%；保留H100／GB200、版本與Judge retry/token-cap差異，不宣稱同平台嚴格因果效果。


Gate仍insufficient_evidence；單constructionseed0、原50分層樣本非無偏、16clusters／人類裁決、大autograd失敗與平台差異等限制保留。model review不是human，不更改human-events或attestation。本次依2026-10-19使用者指示發布aggregate報告；raw／QA／responses／逐題labels／reasons／key／weights／CLI維持private。



## 相對同平台 Mixing 的分數差異與未知全分母範圍

| 模型 | ΔMT 全160turn | ΔH 全300題 (pp) | ΔD 全300題 (pp) |
| --- | --- | --- | --- |
| PureNPO α=0／UL0 | -4.144–-4.144 | -4.667–-4.667 | 100.000–100.000 |
| NPO+retain α=1／UL0 | -1.044–-1.044 | 27.667–28.000 | 54.667–55.000 |
| PureNPO α=0／UL1 | -4.144–-4.144 | -3.667–-3.000 | 99.333–100.000 |
| NPO+retain α=1／UL1 | -2.138–-2.138 | 51.333–52.333 | 30.667–31.667 |
| PureNPO α=0／UL2 | -4.150–-4.150 | -2.667–-1.333 | 98.667–100.000 |
| NPO+retain α=1／UL2 | -2.444–-2.444 | 32.333–33.000 | 51.333–52.000 |

Mixing/reference 兩份原首輪共920次：915有效＋5未知；各原始逐筆稽核與own commit已完成。另補送5筆後為918有效＋2未知。連同原六模型，原首輪共3680次：3658有效＋22未知；另補送22次、補回8筆後，合計3666有效＋14未知。成功題目從未重送；原未知各另補送一次，補送後仍未知者停止。
參照模型列不使用KS=0/p=1冒充模型遺忘結果；上述差異是同平台同protocol checkpoint比較，未知保留全分母上下界，並非信賴區間或無條件因果推論。
forget05固定範圍至此有限嘗試與表格已完成；後續任務依真人指示轉為GB200速度診斷，原freeze/模型/收據/分數不改。


## 原首輪與另一次補送的逐模型覆蓋

| 模型 | 原有效／460 | 原未知 | 額外補送次數 | 補回有效 | 最後未知 |
| --- | --- | --- | --- | --- | --- |
| matching-retain95 | 457 | 3 | 3 | 1 | 2 |
| mixing-baseline | 458 | 2 | 2 | 2 | 0 |
| pure-s0 | 460 | 0 | 0 | 0 | 0 |
| pure-s1 | 458 | 2 | 2 | 0 | 2 |
| pure-s2 | 456 | 4 | 4 | 0 | 4 |
| retain-s0 | 457 | 3 | 3 | 2 | 1 |
| retain-s1 | 455 | 5 | 5 | 2 | 3 |
| retain-s2 | 457 | 3 | 3 | 1 | 2 |

原首輪3680次；真人授權額外補送22次；總API嘗試3702次。原本成功題目補送次數0，原始audit與fate皆保留。補送的SDK自動重試仍為0，每個原未知僅多一次；再次失敗不續送。
額外補送已知usage：input 6668、output 980 tokens；錯誤請求未回報的usage未知，不填0或推算美元費用。
主表為原有效標籤加補送成功後的覆蓋，原首輪有效／未知另列。全分母bounds仍保留最後未知。

## 正式配置與版本註記

本報告整理日期為2026-10-19（Asia/Taipei），實驗與最終有限評分於2026-10-18前已完成。數值表格逐列沿用已接受的最終報告，未重新推論、付費評分或修改原紀錄。

- 起始模型：同一Mixing-full，construction seed0；PureNPO alpha0／NPO+retain alpha1各UL seeds0/1/2，均獨立從Mixing開始。
- forget05=200／retain95=3800，10實際epochs、500microsteps／70updates；world1、micro4、GAS8、global32，各epoch full8／tail2，完整除8／尾組除2、scale_wrt_gas=False。
- LR1e-5、WD0.01、beta0.1／gamma1、Adam(0.9,0.999), eps1e-8、clip1、paged_adamw_32bit、warmup_steps6／warmup_ratio0／linear70。
- BF16／FlashAttention2／ZeRO3／gradient checkpointing；optimizer及parameter offload均none。NPO frozen reference是初始Mixing；matching-retain95只作eval reference，非NPO loss reference。
- 原正式MMLU使用batch1；本次效能發現未回填既有評估。MT-Bench160turn與HEx-PHI300的Judge是gpt-5.6-terra／medium／4096 cap／SDK自動retry0；歷史M0有平台、版本及retry/cap差異。
- 本次比較不含SPF projection或額外安全anchor loss，並非安全防護介入實驗。PureNPO的低H伴隨全面退化，NPO+retain的H則顯著高於Mixing；不能宣稱成功恢復安全性。

GitHub僅發布aggregate與來源摘要雜湊。下一步見[forget01／forget10 handoff](handoff-mixing-forget01-forget10-20261019.md)，速度實證見[GB200診斷](gb200-performance-diagnosis-20261019.md)。

來源摘要SHA256：`ecde483293a6e3827d7c7f8a8a0ef988d0fd1f3e8f658737644d8eda096933ba`；training freeze：`e59b9ca738574ead8fcdc265503a4be46bb3e4ab9a18a785000ee9679c3fa102`。
