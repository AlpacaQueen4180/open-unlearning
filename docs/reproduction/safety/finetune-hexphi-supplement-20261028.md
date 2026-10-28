# Finetune checkpoints：HEx-PHI 補評

2026-10-28（Asia/Taipei）收尾：**13個可用 checkpoint 的固定 HEx-PHI300 推論、有限 Judge 與原始收據稽核均完成；共3900題，3787題有效、113題未知。Standard SFT（C1）因原 H100 連線不可用仍未評估，因此原報告14個模型尚未全部補齊。** 本次獲使用者授權使用既有本機 key；原 development H 分數保留。

固定300題，資料 SHA256 `f72785518afa1dde3c1324987e123ef307a6e2ee2b69a8646c738c06e051db2e`；原始生成程式、formal-300、greedy、seed0、512 tokens、batch1。CPU threads16，原模型與 tokenizer 不變。Judge gpt-5.6-terra、medium、4096 cap、SDK retry0；原 rubric SHA256 `ddfd529d84ab10b4b3cc8054cecfac8fa8abcc4ef718b77a3b5e5d584eec153a`。

| ID | 模型 | 有害協助／有效 | Harmfulness | 未知 | 退化 | 狀態 |
|---|---|---|---|---|---|---|
| C0 | 原始 M0 | 32/297 | 10.77% | 3 | 0 | 原始收據稽核完成；完整300題界限10.67%～11.67% |
| C1 | Standard SFT（H100） | — | — | — | — | 受阻：原 H100 SSH connection refused；未替換模型 |
| C2 | Mixing（原 H100 checkpoint 保存副本） | 26/298 | 8.72% | 2 | 1 | 原始收據稽核完成；完整300題界限8.67%～9.33% |
| F0 | 原始 SPF β₁=.9 | 228/296 | 77.03% | 4 | 0 | 原始收據稽核完成；完整300題界限76.00%～77.33% |
| S1 | SPF β₁=.5 LR1e-5 | 122/297 | 41.08% | 3 | 0 | 原始收據稽核完成；完整300題界限40.67%～41.67% |
| S2 | SPF β₁=.5 LR5e-6 | 91/296 | 30.74% | 4 | 0 | 原始收據稽核完成；完整300題界限30.33%～31.67% |
| S3 | SPF β₁=.5 LR3e-6 | 75/296 | 25.34% | 4 | 0 | 原始收據稽核完成；完整300題界限25.00%～26.33% |
| S4 | SPF hybrid λ=.1 rank20 | 64/297 | 21.55% | 3 | 0 | 原始收據稽核完成；完整300題界限21.33%～22.33% |
| S5 | SPF hybrid λ=.3 rank20 | 55/298 | 18.46% | 2 | 0 | 原始收據稽核完成；完整300題界限18.33%～19.00% |
| S6 | SPF hybrid λ=.1 rank40 | 59/297 | 19.87% | 3 | 0 | 原始收據稽核完成；完整300題界限19.67%～20.67% |
| R1 | SPF retain95 reference | 208/274 | 75.91% | 26 | 0 | 原始收據稽核完成；完整300題界限69.33%～78.00% |
| F0-157 | 原始 SPF update157 | 215/295 | 72.88% | 5 | 0 | 原始收據稽核完成；完整300題界限71.67%～73.33% |
| F0-313 | 原始 SPF update313 | 191/271 | 70.48% | 29 | 0 | 原始收據稽核完成；完整300題界限63.67%～73.33% |
| F0-469 | 原始 SPF update469 | 215/275 | 78.18% | 25 | 0 | 原始收據稽核完成；完整300題界限71.67%～80.00% |

原始 prompts、responses、逐題 labels、理由、key、ledger 與完整收據保留 private；此文件僅含 aggregate 與狀態。成功或狀態未知的 Judge 不重送；未知保留全分母界限，不當作拒答。C2 使用原 H100 checkpoint 的既存保存副本，未以後續 GB200 重訓 Mixing 取代。

所有結果僅有 construction seed0，人工校準與 paired cluster／獨立 holdout 驗證不因此完成。

R1 的26題未知均為 Judge API 明確400 policy-blocked（cyber_policy22、bio_policy4），未重送；不是被評模型的拒答標籤。R1完整300題有害率界限為69.33%～78.00%，不是信賴區間。

## 結果判讀

Mixing為26/298＝8.72%，原始M0為32/297＝10.77%；原始SPF為228/296＝77.03%。SPF系列中，λ=.3、rank20 hybrid（S5）的55/298＝18.46%最低；rank40 hybrid（S6）為59/297＝19.87%。這些是本批已驗標籤的描述性點估計，不代表多seed或paired差異已驗證。Standard缺項與未知標籤仍保留，不能宣稱完整正式安全gate通過。

原始SPF中途update157／313／469分別為72.88%／70.48%／78.18%；其中313與469有29／25題未知，完整300題界限分別63.67%～73.33%／71.67%～80.00%。界限是缺失標籤的可能範圍，不是信賴區間。

## 帳務與來源

本次新增3335個不同payload的first attempts，重用563個同profile、同完整payload的既有成功結果；其餘既有未知映射保留，不再次送出。所有模型皆從各自已核對的checkpoint生成，沒有以其他模型的不同回答代填。未知不當作refusal或safe；未重送、未改寫或重新措辭。

[完整aggregate與來源SHA256](../../../results/reproduction/safety/finetune-hexphi-20261028/final-audited.json)包含全部13模型、原始audit雜湊、usage與錯誤類別。收尾僅使用已保存本機bytes，沒有再讀key、發送Judge、查詢RunAI或重新雜湊權重。評估收尾JSON中的`GitHub_pushed=false`記錄當時尚未發布的狀態；本次GitHub整合另由 [publication-manifest.json](../../../results/reproduction/safety/finetune-hexphi-20261028/publication-manifest.json)記錄，原始評估aggregate保持不變。

| Judge未知類別 | 題數（跨模型） |
|---|---:|
| BadRequestError / 400 / bio_policy | 42 |
| BadRequestError / 400 / cyber_policy | 71 |

113個未知均有原始Judge API 400 policy-blocked證據（bio_policy42、cyber_policy71），含2個沿用的既有未知；沒有再次送出。

本次追蹤排程已 **PAUSED**。13個可用模型工作已結束；Standard SFT仍待使用者提供原checkpoint的可用連線或保存位置，未宣稱14個模型全部完成。

本表已整合進 [Finetune實驗主報告](finetune-results-report-20261012.md)前段；主報告後段保留原development歷史數值。
