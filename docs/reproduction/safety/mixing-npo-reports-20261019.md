# Mixing → NPO：目前報告與下一session入口

更新：2026-10-19（Asia/Taipei）。本輪只有aggregate發布及交接，未啟動新實驗。

| 文件 | 範圍 | 狀態 |
|---|---|---|
| [forget05完整最終表格](mixing-npo-forget05-final-20261019.md) | M0歷史／Mixing／6個NPO／retain95 reference；所有數值橫向 | 已完成，14個最終未知保留bounds |
| [GB200效能診斷](gb200-performance-diagnosis-20261019.md) | batch8、skip-retain、CPU／RAM／kernel trace與限制 | 已完成，正式原結果不改 |
| [下一session Handoff：forget01／forget10](handoff-mixing-forget01-forget10-20261019.md) | 12個NPO預設矩陣、matching retain99／90、split-aware counts與加速profile | 計畫／尚未launch |

公開branch：`repro/npo-h100-ada-5seed`。原始問題與回答、逐題標籤／理由、API金鑰、權重與主機操作紀錄不在這次發布內容。

資料：[forget05 aggregates](../../../results/reproduction/safety/mixing-npo-20261019/forget05-aggregate-tables.json)、[next campaign plan](../../../results/reproduction/safety/mixing-npo-20261019/next-campaign-plan.json)、[來源SHA manifest](../../../results/reproduction/safety/mixing-npo-20261019/provenance.json)。
