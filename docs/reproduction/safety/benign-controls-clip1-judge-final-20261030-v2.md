# Clip1 CE controls：Judge最終驗收 v2

> 本頁為使用者另行授權的公開整理版。數值與原始驗收不變；原文「未發布」等描述保留其驗收時語境。僅移除本次範圍外之後續設計連結，來源與公開版hash分列於publication manifest。

2026-10-30（Asia/Taipei）。原授權最多527筆新首次請求、USD35已執行完：525筆成功，2筆因API bio_policy／cyber_policy拒絕保留unknown，不改寫或重送；未送0。加入379筆exact-payload歷史重用與14筆新跨模型去重，兩模型共920列中918列有驗證標籤。

| 模型 | H full300 % | D full300 % | MT 160平均 | MMLU % | TOFU EM F/R % |
|---|---:|---:|---:|---:|---:|
| Mixing | 9.000 | 0 | 5.16875 | 65.653 | 98.000 / 97.500 |
| PureNPO clip0 | 4.333 | 100 | 1.02500 | 60.654 | 0 / 0 |
| NPO+retain clip0 | 36.667–37.000 | 54.667–55.000 | 4.12500 | 65.126 | 0 / 14.395 |
| 正向SFT clip1 | 13.000 | 0 | 4.78125 | 64.991 | 100.000 / 56.289 |
| retain continued CE clip1 | 17.667–18.333 | 0–0.667 | 4.76875 | 65.803 | 76.500 / 81.342 |

SFT clip1的HEx300及MT160全數有標籤。Retain continued的HEx為298/300，已驗證53筆harmful、0筆degenerate；表中始終用300分母及最壞缺值界，不以53/298假充完整結果，MT160完整。NPO+retain舊HEx也有1筆缺值，保持原分母界。這些缺值界不是信賴區間。

正向更新也能伴隨harmfulness上升：SFT對Mixing的H增加4百分點，retain CE增加8.667–9.333百分點。SFT沒有Judge degeneration；retain CE在已驗證298筆也沒有degeneration，但仍保留缺值D上界。因此本次增加不能用普遍collapse解釋。這不代表整體品質保持：兩模型MT均比Mixing低約0.39–0.40分，SFT還伴隨retain EM大幅下降與MMLU下降；retain CE的MMLU略高而TOFU retain仍低，顯示安全、各種能力與輸出品質須分開判讀。

原PureNPO較低H伴隨D100%與極低MT，不能當成安全改善成功；NPO+retain也有明顯D與retain能力損失。兩種NPO與正向CE的模式不同，但這裡新CE為clip1、原NPO為clip0，不能僅憑該差異歸因objective。新的兩個NPO clip1正在獨立scope執行；兩個clip0 CE controls尚無本scope授權的Judge，H/D欄仍空。因此九模型的objective內clip1−clip0安全比較尚待其餘免費生成與另外Judge提案，沒有把缺項代填或因本次結果修改LR／抽樣。

每模型160個MT row來自80題兩turn，turn不是獨立問題；本輪未以單seed結果做完整因果證明，也未以p>.05聲稱retraining equivalence。原TOFU native retainTR/PARA錯接及corrected diagnostic繼續分開，詳見原免費v3。

379個重用row中4列／2個distinct payload有歷史離散MT分數衝突。按事前earliest accepted政策列報，全部替代值保留；SFT MT sensitivity為4.775–4.7875，retain CE為4.7625–4.775，H/D不受這些離散衝突影響。

初次327筆送出後遇2筆policy拒絕，原worker保留停止；325筆成功先獨立驗收。以唯讀PID／intent／SQLite／raw證明200筆未送，再建立continue-v2窄清單，維持原527／USD35上限、profile及rubric、原預留與global blocked歷史；只對未送payload各第一次請求，成功200。沒有重啟整批、重送unknown、改寫被拒payload或停用provider safeguards。實際所有527筆各只有一個durable intent、SDK retry0；兩個worker皆已結束。

成功usage共input293065、output99095（含reasoning），cache tokens0。以官方模型費率保守input2.5/M（含cache-write ceiling）與output12/M計成功費用上界USD1.921937；2筆unknown另保留USD0.135405，合計上界USD2.057342。原獨立budget預留USD34.885422，授權cap USD35；預留不是已花費，這些usage計算上界也不是provider invoice，unknown成本不當0。不擴大新模型／baseline、不挪用OLMo、SPF或舊Mixing額度。

所有raw/schema/payload／精確input count／usage／帳務／唯一intent與歷史重用hash均通過closed CPU audit；原始標籤理由與HTTP拒絕、CLI、權重／QA／生成／憑證只private。未付費提案與失敗、v1停止／部分驗收、v2續行原source全部保留，原免費v3未覆寫，不發布GitHub。

[最終aggregate](../../../results/reproduction/safety/benign-controls-clip1-20261030-v1/judge-v2/aggregate.json)、[七模型比較與缺項](../../../results/reproduction/safety/benign-controls-clip1-20261030-v1/judge-v2/comparison.json)、[原免費報告](benign-controls-clip1-final-20261030-v3.md)、clipping九模型後續設計（不屬本次發布範圍）。費率與API profile來源見[原Judge驗收v1](benign-controls-clip1-judge-20261030-v1.md)。
