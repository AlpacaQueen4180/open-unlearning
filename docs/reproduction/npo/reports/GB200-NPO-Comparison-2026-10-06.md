# GB200 NPO 與前次 H100／Ada reproduction 的差異

整理日期：2026-10-06。October 實驗於 2026-10-05 完成。兩模型為 Llama-2-7b-chat-hf 與 Llama-3.2-1B-Instruct；forget01／05／10。這份報告比較 seed0，避免把前次五-seed平均與本次單 seed 混為相同統計量。

**單張 GB200＋ZeRO-3 已接近歷史雙 H100 的 Llama-2 forget05 trade-off；「必須兩張 GPU」不是這批證據支持的必要條件。新的六格訓練修正版完成十個實際 epochs，但不表示六格都重現 docs/repro.md。**

## 1. 實驗範圍及共同設定

August 為 H100、Ada 各六格 × 五 seeds，共 60 個 matrix runs。October 保留六格原版 seed0，另完成六格 corrected seed0 和兩模型原版 forget01 各 seeds1／2；合計 16 個 ZeRO-3 full runs。額外的 forget05 nonDS control、smoke 與未執行的兩卡提交分開分類，不混入矩陣。

共同 recipe：TOFU full 起始 checkpoint、對應 forget／retain split、NPO beta=0.1、alpha=gamma=1、retain NLL、lr=1e-5、weight decay=0.01、paged_adamw_32bit、bf16、gradient checkpointing、十個配置 epochs。October 沒有更換 NPO objective 或既有 loss scaling，也沒有用追逐文件分數的方式調參。

| 設定 | August 雙 H100 | August 單 Ada6000 | October GB200 原版 | October GB200 修正版 |
|---|---|---|---|---|
| GPU／processes | 2／2 | 1／1 | 1／1 | 1／1 |
| 引擎 | ZeRO-3 | 一般單卡，無 DS | ZeRO-3 | ZeRO-3＋opt-in ceil／boundary policy |
| Microbatch × accumulation × world | 4×4×2 | 8×4×1 | 4×8×1 | 4×8×1 |
| 名義 global batch | 32 | 32 | 32 | 32；尾端短組見下文 |
| Attention | FlashAttention 2 | FlashAttention 2 | FlashAttention 2 | FlashAttention 2 |
| Seed coverage | 六格各0–4 | 六格各0–4 | 六格0；forget01補1／2 | 六格各0 |
| 更新 instrumentation | Trainer state；未保存 DS runtime counters | Trainer／status，無 DS | Trainer／DS／microsteps | 加 policy／forced boundaries 與斷言 |

較早的 August GB200 控制使用 eager attention；較早的 Ada 初始版本控制使用 Transformers 4.45.1。這些控制不同於 August 五-seed matrix；Ada matrix 雖使用名稱含 `transformers445` 的環境，實際以 PYTHONPATH overlay 載入 4.51.3，不能用環境名稱推定版本。

## 2. 版本、執行方式與證據差異

| Package | August H100／Ada matrix | October GB200 |
|---|---|---|
| Torch／CUDA | 2.4.1+cu121／12.1 | NVIDIA container 2.7.0a0／12.8 |
| Transformers | 4.51.3 | 4.51.3 |
| Accelerate | 0.34.2 | 0.34.2 |
| DeepSpeed | 0.15.4 | 0.15.4 |
| FlashAttention | 2.6.3 | 2.7.3 |
| bitsandbytes | 0.44.1 | 0.50.2 |

GB200 使用 `nvcr.io/nvidia/pytorch:25.02-py3` 的 aarch64 runtime，隔離 venv 保留容器 Torch，驗證 FlashAttention forward/backward 與 paged AdamW kernel。October 基底 commit 為 `820102411091abba2c5203207391bc1be0f3a863`，歷史 reproduction source 為 `4ad738aaf60f6a4385f6e2506d01da99e76c31f3`；訓練 objective 保持原版，修正只在 opt-in wrapper 生效。

October 明確記錄 model／training-data／reference revisions、retain reference SHA256、命令、Hydra／DS 配置、Trainer／engine runtime。評估仍使用 repository 的標準 dataset loader，不能宣稱所有 evaluation dataset loads 都另外鎖定 revision；也未逐一證明與 August 資產 hash 完全相同。

October 禁用 training-time evaluation，保存後單独評估；August 使用原入口及預設流程。October 各控制使用同一設定。平台、套件、microbatch、分散式 sampling 與更新軌跡均可能影響數值；沒有做完整因果消融。

## 3. 同 seed0 的六格指標

各欄順序 **FQ／MU／TR**。FQ 為 KS p-value，不是越高越好的線性分數。原始精度保留在連結的 JSON。

| 模型 | Split | August 雙 H100 | August 單 Ada | October 單 GB200 原版 | October 單 GB200 修正版 | docs/repro.md |
|---|---|---|---|---|---|---|
| Llama-2-7b-chat-hf | forget01 | 0.0143 / 0.6221 / 0.5495 | 0.0068 / 0.6223 / 0.5378 | 0.0143 / 0.6219 / 0.5493 | 0.5786 / 0.5509 / 0.6868 | 0.4000 / 0.5800 / 0.6500 |
| Llama-2-7b-chat-hf | forget05 | 0.1421 / 0.5292 / 0.7124 | 1.428e-12 / 0.5753 / 0.5377 | 0.1421 / 0.5302 / 0.7058 | 0.6284 / 0.5269 / 0.7090 | 0.0900 / 0.5300 / 0.7100 |
| Llama-2-7b-chat-hf | forget10 | 0.2812 / 0.5301 / 0.7272 | 1.119e-19 / 0.5273 / 0.5810 | 0.3222 / 0.5466 / 0.7170 | 0.2812 / 0.5667 / 0.7135 | 0.4200 / 0.5400 / 0.7300 |
| Llama-3.2-1B-Instruct | forget01 | 0.0030 / 0.5890 / 0.5437 | 0.0068 / 0.5960 / 0.5068 | 0.0068 / 0.5880 / 0.5439 | 0.7659 / 0.5357 / 0.6786 | 0.9200 / 0.5600 / 0.6600 |
| Llama-3.2-1B-Instruct | forget05 | 0.1779 / 0.4292 / 0.7081 | 4.749e-05 / 0.4539 / 0.6092 | 0.2205 / 0.4400 / 0.7048 | 0.1779 / 0.4745 / 0.7013 | 0.1400 / 0.4500 / 0.7000 |
| Llama-3.2-1B-Instruct | forget10 | 0.0158 / 0.5153 / 0.6460 | 3.596e-05 / 0.4024 / 0.6460 | 0.0446 / 0.5228 / 0.6425 | 0.0299 / 0.5385 / 0.6362 | 0.0200 / 0.4600 / 0.7000 |

Llama-2 forget05 歷史雙 H100 為 0.1420746515／0.5292339411／0.7124040519；GB200 ZeRO-3 原版為 0.1420746515／0.5301676313／0.7057618828。相同 GB200、FlashAttention2、batch32 的 nonDS 對照則為 FQ 6.568964586e-12／MU 0.5754775690／TR 0.5385220438。這使「單卡必然失敗」的結論不再成立，但不能據此宣稱 ZeRO 是唯一原因。

August 的 close-reproduction 主結論原本限於 Llama-2 forget05，不能延伸成所有六格成功；例如雙 H100 的 forget01 本來也沒有貼近 docs。更新修正後，forget01 FQ／TR 明顯改變，Llama-2 forget05 FQ 則更偏離文件；MU／TR 仍近似。程式計數修復與 published-score matching 是不同驗收。

## 4. 十 epochs 配置與實際更新

歷史 H100 seed0 與 October 原版的實際 epoch 均為 forget01=5、forget05=8.64、forget10=9.24；H100 只保存 Trainer 10／60／120 steps，沒有觀測 DS 的實際更新數。October 原版量得 Trainer／DS=10／6、60／54、120／115；corrected 則為20／20、70／70、130／130，全部 epoch10，microsteps100／500／1000。

修正 `ceil_epoch_and_sync_ds_boundary_v1` 使用 ceil 規劃及與 Trainer 同步的 DS 更新邊界。尾端短組只有8（forget01／05）或16（forget10）examples；loss scaling 和原 warmup 計算未改。這同時改變訓練長度、尾端 flush 及 scheduler 時間軸，不能當成原文件訓練流程的完全等價實作。完整機制、上游來源與所有 NPO／safety 報告分類見[更新次數問題文件](../update-count-mismatch-20261006.md)。

## 5. Forget01 額外 seeds

原版每模型 seeds0／1／2，FQ 範圍：Llama-2 0.0143–0.0286、Llama-3.2 0.0030–0.0541，仍遠低於文件的0.40／0.92。MU／TR 也沒有靠換 seed 接近文件；本次三 seed 未顯示原版差距可單靠 seed 消除，但 n=3 不足以描述完整分布。

corrected 每格只有 seed0，不能把原版 std 套用到 corrected，不能混成四 seed。逐 seed、mean／sample std／range 見[follow-up 報告](GB200-NPO-Followup-2026-10-05.md)和[統計 JSON](../../../../results/reproduction/npo/gb200/forget01-seed-statistics-20261005.json)。

## 6. 驗證與保存限制

六個原版及十個 follow-up runs 的 FQ／KS D／MU／TR 已獨立重算，新增十格與原 summary 差值均0；樣本40／200／400逐 split 核對。檢查 checkpoint index 所列分片存在且 bytes 符合 inventories；巨大權重保留 RunAI PVC，未下載／雜湊，不能將分片清單稱為權重 checksum 驗證。

419份小檔原始證據包含16個 full runs、references、logs、commands、configs、runtime、source 與 smoke，下載後逐檔 SHA256 驗證。32,548,127bytes archive 的 SHA256 為 `6c3b5119929e89669c22de0791c0b838a2cdccaed793c87d479464b7c554e9ae`。原始 archive 留在本機；GitHub 發布精簡 audits、報告、source。Historical 部分 extra-seed weights 已由 as-run 腳本在評估完成後刪除；October 權重保留 PVC，兩者 artifact retention 不同。

RunAI project hard quota1GPU 使兩 GB200 工作未能執行，故沒有 GB200 一卡對兩卡結果；Pending／smoke 不算正式成功。工作完成後15分鐘 heartbeat已暫停。docs/repro.md 的 batch 說明8×2×4=64卻稱32；本次沿用32並明示此差異。

## 7. 可追溯來源

- [August H100 report](OpenUnlearning-NPO-Reproduction-Final-2026-08-28.md)、[H100 五-seed JSON](../../../../results/reproduction/npo/h100/openunlearning_npo_2xH100_5seeds_all6_20260828.summary.json)、[Ada 五-seed JSON](../../../../results/reproduction/npo/ada6000/openunlearning_npo_1xAda6000_5seeds_all6_20260829.summary.json)。
- [October 原版六格](GB200-NPO-Matrix-2026-10-05.md)、[follow-up](GB200-NPO-Followup-2026-10-05.md)、[獨立 audit](../../../../results/reproduction/npo/gb200/followup-independent-audit-20261005.json)、[ZeRO-3／nonDS control audit](../../../../results/reproduction/npo/gb200/independent-audit-20261005.json)。
- [執行時間軸](../gb200-20261004.md)、[GB200 執行器](../../../../scripts/reproduction/npo/gb200/README.md)、[更新次數／safety 分類](../update-count-mismatch-20261006.md)。
