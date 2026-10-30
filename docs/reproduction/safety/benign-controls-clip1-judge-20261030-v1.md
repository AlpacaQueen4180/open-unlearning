# Clip1 Judge執行與比較 v1

2026-10-30（Asia/Taipei）。這是原免費交付v3之後的新授權階段；舊提案、收據、數值及原始權重保持凍結。

人類「授權使用Judge」依緊接的具體提案限定為兩個clip1 final、最多527筆distinct首次評分、USD35。保持原gpt-5.6-terra／medium／max_output4096／原rubric及schema／SDK retry0；獨立SQLite完整預留先於credential和API intent，精確server input count全部通過後才開始Judge。沒有追加baseline、clip0或改動其他campaign帳務。

| 模型 | HEx已驗證 | H full300 % | D full300 % | MT已驗證 | MT full160分數範圍 |
|---|---:|---:|---:|---:|---:|
| matched-benign-sft-clip1-s0 | 300/300 | 13.000–13.000 | 0.000–0.000 | 160/160 | 4.78125–4.78125 |
| retain-continued-clip1-s0 | 298/300 | 17.667–18.333 | 0.000–0.667 | 160/160 | 4.76875–4.76875 |

實際新首次送出527筆、成功525筆、unknown2筆、未送0筆；379筆歷史row exact-payload重用與14筆新payload跨模型去重。任何success／unknown不重送。新成功usage之保守模型費用上界USD1.921937，unknown保留USD0.135405；原全額reservation USD34.885422。這些是依usage及官方費率計算的上界，不是provider invoice，unknown絕不當0。

重用379列中有4列／2個distinct payload存在歷史離散label或score分歧；按事前固定earliest accepted選擇，所有替代結果及其aggregate sensitivity保留。不能把同一payload重複row當成獨立Judge樣本，也未為衝突額外付費重評。

完整七模型免費能力與原matching native／corrected缺陷仍見[原免費報告v3](benign-controls-clip1-final-20261030-v3.md)。原Mixing H9%／D0%／MT5.16875；PureNPO H4.333%伴隨D100%／MT1.025；NPO+retain有1筆HEx缺值，full300 H36.667–37%／D54.667–55%，MT4.125，均重用原接受收據。clip0控制仍缺新Judge，這次授權不延伸到clip0。

需聯合判讀H、D、MT與TOFU retain／MMLU：較低H若伴隨degeneration或回答品質下降，不能當成安全改善成功。正向SFT本已造成retain EM由97.5%降至56.289%，retain-only到81.342%；NPO有不同forget與退化模式。新clip1與原NPO0另有clipping差異；單seed、matched updates未匹配tokens、成本或梯度位移，不是完整因果證明。缺標籤的full denominator bounds不是信賴區間，不能用p>.05聲稱retraining equivalence。

官方費率於送出前重新搜尋並開啟[同model頁](https://developers.openai.com/api/docs/models/gpt-5.6-terra)，input2/M、cached.20/M、output12/M，cache-write保守input2.5/M；本輪不假設cached折扣。exact count流程依[OpenAI Docs](https://developers.openai.com/api/docs/guides/token-counting)，與SDK實際wire schema離線逐欄比對。

公開僅aggregate、設定與hash；原始QA、生成、逐筆label／reason、CLI、credential與完整收據留private。不發布GitHub。
