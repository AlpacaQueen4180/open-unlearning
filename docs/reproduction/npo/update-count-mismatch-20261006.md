# NPO 更新次數與 epoch 尾端問題：修正範圍及既有報告分類

核對日期：2026-10-06。這份文件以本 repository 保存的啟動腳本、Trainer state 與 GB200 runtime audit 為依據，區分「訓練更新次數修正」和「評估修正」。

> **10:34最新版本範圍：** upstream `17cbbc8` 的 Transformers5.5.4採ceil步數規劃，Accelerate1.13.0同步DS更新邊界。GB200新版native 1B forget01／05／10 seed0已實測Trainer=DS20／70／130、micro100／500／1000、epoch10，完整checkpoint與評估獨立稽核完成，沒有疊加本地policy。此為另外三個full runs，與六個legacy corrected分開分類；保留容器Torch及eval cast偏離。詳見[新版環境與梯度驗證](environment-gradient-validation-20261006.md)。本文舊版計數不套到原paper的4.45.1或歷史safety的未保存DS計數。

> **Loss scaling補充：** 六格corrected seed0只修更新／boundary，未改舊loss scaling。其後受控舊H100／GB200量測已證實漏除accumulation分母；真實1B TOFU對同DS explicit-mean reference舊完整／短組8／2倍、新版1／1倍。原autograd／DS bf16差異仍未解，不能称為所有梯度等價性通過。見[loss-scaling專文](loss-scaling-20261006.md)。

**截至本次核對，已完成且有 runtime 證據使用 `ceil_epoch_and_sync_ds_boundary_v1` 的正式實驗，只有 2026-10-05 GB200 上兩模型 × 三 splits 的六格 corrected seed0。既有 H100／Ada NPO reproduction 和 safety 的 Llama-3.1-8B NPO 報告都沒有套用這項訓練修正。Safety 文件中的 `aggregate.corrected.json` 是評估修復，不是更新次數修正。**

## 1. 兩個不同的計數問題

在本次固定的 Transformers 4.51.3／Accelerate 0.34.2／DeepSpeed 0.15.4 組合，需分別核對：

1. **訓練計畫提前結束。** Trainer 用 dataloader 長度除以 accumulation 的向下取整規劃每 epoch 更新數，但 loop 在 epoch 最後不足一組時也增加 Trainer step。因而先到達規劃的 `max_steps`，尚未完成設定的 10 epochs 就停止。
2. **Trainer step 與 DS 更新邊界不同。** Trainer 在 epoch 尾端要求同步；原版 DS 預設依連續 microstep 計數決定 accumulation 邊界。兩者未同步時，Trainer 的 `global_step` 不是 DS 的 optimizer 更新數。

向下取整、尾端同步條件可在 [Transformers 4.51.3 Trainer source](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/trainer.py#L4837) 和 [training loop](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/trainer.py#L2293) 核對。DS 的預設邊界與手動設定介面見 [DeepSpeed 0.15.4 engine](https://github.com/microsoft/DeepSpeed/blob/v0.15.4/deepspeed/runtime/engine.py#L1886)。這是固定版本下的機制說明，不能只看目前最新版文件推定舊實驗行為。

上游 [Transformers issue #36297](https://github.com/huggingface/transformers/issues/36297) 已報告計畫與 loop 不一致造成提前停止。另有 [issue #38837](https://github.com/huggingface/transformers/issues/38837) 討論不足一組時的 loss normalization；那是不同問題，本次沒有宣稱一併修復。

## 2. GB200 實際量測

以下為 single GB200、microbatch 4、accumulation 8、名義 global batch 32。兩模型在各 split 的計數相同；`microsteps` 指 forward/backward 批次，不是 optimizer 更新。

| Split | 每 epoch microbatches | 原版 Trainer／DS 更新 | 原版 microsteps | 原版實際 epoch | 修正版 Trainer／DS／強制邊界 | 修正版 microsteps | 修正版實際 epoch |
|---|---:|---|---:|---:|---|---:|---:|
| forget01 | 10 | 10／6 | 50 | 5 | 20／20／20 | 100 | 10 |
| forget05 | 50 | 60／54 | 432 | 8.64 | 70／70／70 | 500 | 10 |
| forget10 | 100 | 120／115 | 924 | 9.24 | 130／130／130 | 1000 | 10 |

以 forget01 為例，原計畫每 epoch 為 floor(10/8)=1 次，共 10 次；loop 每完整 epoch 計入兩個更新邊界，於第五個 epoch 到達 10 次。DS 原版只累積出 6 次更新。修正版以 ceil(10/8)=2 規劃，並同步尾端 DS 邊界，完成 10 epochs、20 次更新。

獨立核對資料：[原版六格](../../../results/reproduction/npo/gb200/matrix-status-20261005.json)、[十格 follow-up 與六格 baseline audit](../../../results/reproduction/npo/gb200/followup-independent-audit-20261005.json)。August H100 seed0 保存的 Trainer state 也記錄 forget01／05／10 為 5／8.64／9.24 epochs、10／60／120 global steps，見 [historical status](../../../results/reproduction/npo/h100/openunlearning_npo_2xH100_llama2_7b_llama3_1b_forget01_05_10_20260828.status.json)。**當時未保存 DS engine 計數，不能把 GB200 的 6／54／115 當成 H100 的實測值。**

## 3. 本次修正如何套用

修正是 reproduction 執行器的 opt-in patch，見 [corrected_probe.py](../../../scripts/reproduction/npo/gb200/corrected_probe.py)；没有全域改寫 `src/trainer` 或套件。

- 有限 dataloader 的 epoch-based 計畫改成 ceil，明確 `max_steps` 的 smoke 則保留其上限。
- 在每個 `training_step` 的 forward/backward 之前，呼叫 active DS engine 的 `set_gradient_accumulation_boundary(accelerator.sync_gradients)`。
- [run_probe.py](../../../scripts/reproduction/npo/gb200/run_probe.py) 保存 Trainer／DS／microstep／強制邊界計數；[audit_cell.py](../../../scripts/reproduction/npo/gb200/audit_cell.py) 在正式 corrected runs 斷言 epoch 與計數。
- 真實 Llama-3.2 forget01 smoke 通過 Trainer=DS=4、microsteps=20、epoch=2；它只作門檻，不是正式 reproduction 結果。

NPO objective、beta／alpha／gamma、既有 loss scaling、套件版本保持不變。epoch 尾端實際短組只有 8 個 examples（forget01／05）或 16 個（forget10），不是 32。Warmup 的既有計算也沒有改寫。此介入同時改變訓練長度、尾端更新與 scheduler 時間軸；尚未拆成因果消融、沒有驗證完整 gradient normalization，也未驗證中途訓練 resume 的所有情形。

## 4. 哪些 NPO 實驗使用訓練更新修正

| 實驗／報告 | 訓練更新修正 | 判定依據及範圍 |
|---|---|---|
| August H100 headline、兩模型 × 三 splits × 五 seeds | **未套用** | [as-run 腳本](../../../scripts/reproduction/npo/as-run/run_h100_matrix_cell.sh) 直接啟動原 `src/train.py`；歷史 seed0 提前停止的 Trainer state 已保存。當時 bf16→NumPy 修復只影響評估序列化。 |
| August Ada6000，兩模型 × 三 splits × 五 seeds | **未套用** | [Ada 腳本](../../../scripts/reproduction/npo/as-run/run_ada6000_seed_cell.sh) 使用一般單卡訓練，沒有本次 ceil patch；DS 邊界問題不適用，實際 epochs 仍需個別 Trainer state 核對。 |
| August 早期 GB200 eager、版本／獨立 trainer／duplicate-retain 控制 | **未套用本次 policy** | 見 [August 報告](reports/OpenUnlearning-NPO-Reproduction-Final-2026-08-28.md)。部分控制另有 objective／collator 改動，不能稱為此次更新修正。 |
| October GB200 ZeRO-3 與 nonDS forget05 seed0 對照 | **未套用** | [control audit](../../../results/reproduction/npo/gb200/independent-audit-20261005.json)。nonDS 無 DS 更新計數。 |
| October GB200 原版六格 seed0 | **未套用** | [原版六格報告](reports/GB200-NPO-Matrix-2026-10-05.md)；保留原停止與 accumulation 行為。 |
| October GB200 原版兩模型 forget01 seeds1／2，合併原 seed0 | **未套用** | [follow-up 報告](reports/GB200-NPO-Followup-2026-10-05.md) 中三 seed 統計僅使用原版，不能混入 corrected seed0。 |
| October GB200 兩模型 × forget01／05／10，corrected seed0 | **已套用並驗證，共六個 full runs** | `corrected=true`，policy 與實際 Trainer=DS=forced 計數均保存；兩模型為 Llama-2-7b-chat-hf、Llama-3.2-1B-Instruct。 |
| October 6 GB200 新版native 1B × forget01／05／10，seed0 | **未套本地policy；新版原生修正已實測，共三個full runs** | TF5.5.4／Accel1.13.0／DS0.15.4原生更新與平均；Trainer=DS20／70／130、epoch10。`corrected=false`仅表示沒有legacy probe；不能解讀為舊scaling。完整證據見[新版驗證](environment-gradient-validation-20261006.md)。 |
| October corrected smoke4 | **已套用；非 full run** | 真實 partial-group 門檻測試，不納入正式指標表。 |
| 已提交兩 GB200 workload | **沒有可分類的完成結果** | 1 GPU hard quota 阻止執行；Pending 不算 reproduction，也不算修正驗證。 |

## 5. Safety 報告逐份分類

判定是針對**本次更新次數與 DS 邊界 policy**。既有 safety training commands 直接啟動 `src/train.py`；repo 的 NPO Trainer 沒有此全域 patch。沒有對舊 checkpoint 重訓，也沒有把本次 Llama-2／3.2 的 corrected 分數替換進 Llama-3.1 safety 表格。

| Safety 文件 | 是否包含本次更新修正的正式實驗 | 說明 |
|---|---|---|
| [experiment2_llama31_8b.md](../safety/experiment2_llama31_8b.md) | **否** | 初始公開 Llama-3.1 full／forget05、with／without retain、seed0 使用原版 two-H100 ZeRO-3。`10 epochs` 為配置。 |
| [llama31_seed0_three_split_report_20260910.md](../safety/llama31_seed0_three_split_report_20260910.md) | **否** | 公開三 splits 的 seed0 與自製 forget05 使用原版訓練。自製 full／retain95 是 fine-tuning baselines，不是 corrected NPO。 |
| [llama31_npo_5seed_report_20260911.md](../safety/llama31_npo_5seed_report_20260911.md) | **否** | 「seed0 修正版」指 TOFU evaluation/reference 修復；seeds1–4 queue 也未使用更新修正。 |
| [llama31_8b_joint_report_20260915.md](../safety/llama31_8b_joint_report_20260915.md) | **否** | 八個 NPO conditions 的 TOFU／HEx-PHI 五 seeds、capability seeds0／2／4，均為既有原版 checkpoints；補評估不改訓練。 |
| [實驗規劃與結果追蹤.md](../safety/實驗規劃與結果追蹤.md) | **既有 NPO 結果未套用；新計畫尚未驗證** | 表內既有 Llama-3.1 結果繼承 September 報告。Standard／Mixing target 的 625 updates／5 epochs 是 target SFT，不能當成 corrected NPO。SPF、OLMo、固定實際 steps 等待跑項目不算已套用。 |
| [實驗規劃與結果摘要_精簡版.md](../safety/實驗規劃與結果摘要_精簡版.md) | **同上** | 保留 baseline、target-development 與待跑 unlearning 的分類；沒有新的 corrected safety NPO 指標。 |
| [safety_preserving_finetuning_unlearning_plan_v2.md](../safety/safety_preserving_finetuning_unlearning_plan_v2.md) | **不適用：研究計畫** | SPF 的 task-gradient projection 是另一種方法；不能與更新邊界修正混稱。 |
| [olmo_fully_open_unlearning_proposal.md](../safety/olmo_fully_open_unlearning_proposal.md) | **不適用：研究提案** | 預定完整 epoch 與步數記錄是驗收要求，不是已完成此 policy 的實驗證據。 |

直接證據：

- [seed0 training queue](../../../scripts/reproduction/safety/run_llama31_8b_tofu_rebuild_npo_seed0_queue.sh) 與 [seeds1–4 queue](../../../scripts/reproduction/safety/run_llama31_8b_npo_seeds1_4_archive_queue.sh) 均使用原 training entrypoint。
- [public forget05 seeds1–4 queue](../../../scripts/reproduction/safety/run_llama31_8b_public_f05_seeds1_4_nojudge_queue.sh) 核對 `Trainer global_step=max_steps`；這不等於核對 DS 實際 updates，也不是 ceil／boundary 修復。
- [seed0 evaluation repair](../../../scripts/reproduction/safety/repair_llama31_8b_tofu_rebuild_seed0_evals.sh) 只從三個既有 checkpoint 執行 `src/eval.py`，修正自製 retain95 reference 和不完整 TOFU evaluation，再寫出 `ledger.corrected.tsv`／`aggregate.corrected.json`。沒有 training call。

此核對能確認保存的執行方式沒有套用本次 policy；但本機 safety aggregate 未提供逐 run DS runtime counters，**不能因此推算所有 Llama-3.1 runs 的實際 DS steps 或斷言與 GB200 完全相同的提前停止幅度**。舊 safety 結論應標為「原版訓練流程下觀測到」，不能宣稱是修正後完整 10 epochs 的結果。

## 6. 如何解讀結果及後續驗收

更新一致並不等於接近 docs/repro.md。Llama-2 forget05 FQ 從原版 0.1421 變成 corrected 0.6284，而 MU／TR 仍約 0.53／0.71；forget01 的兩模型結果也大幅改變。完整比較见 [GB200 follow-up](reports/GB200-NPO-Followup-2026-10-05.md)。FQ 是 KS p-value，不能當作越高越好的線性分數。

未來要把 safety 結果升級為 corrected evidence，需另外授權並完成對應 Llama-3.1 訓練，保存更新 policy、configured／actual epochs、Trainer／DS／microsteps、尾端短組大小、checkpoint ID 與評估來源，重新對相同 checkpoint 評估 TOFU、safety、capability；目前沒有這批實驗。不得只重新評估舊 checkpoint 就改標為 corrected training，也不得把不同訓練 policy 的 seeds 合併為同一組變異統計。
