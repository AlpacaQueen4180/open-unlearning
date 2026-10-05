# GB200 NPO 更新邊界修正與 forget01 三 seed 驗證

> GitHub 報告整理：2026-10-06。下文保留完成時的結果與限制；原始 archive 留在本機，精簡 audit 和 source 隨本次更新發布。修正分類見[更新次數問題文件](../update-count-mismatch-20261006.md)。

2026-10-05（台灣時間）：十組新增正式實驗均完成；遠端完成時間為 2026-10-05 20:39:53。六格修正版修正了觀測到的步數與更新邊界 mismatch，全部實際跑滿10 epochs；這不表示六格都重現 docs/repro.md。原版 forget01 的三個 seeds 均仍偏離文件。

數值順序為 FQ / MU / TR。修正版與原版的六格比較使用相同 seed0；下方三 seed 統計只使用原版 seeds0/1/2。

| 模型 | split | 原版 seed0 | 修正版 seed0 | docs/repro.md |
|---|---|---|---|---|
| Llama-2-7b-chat-hf | forget01 | 0.0143 / 0.6219 / 0.5493 | 0.5786 / 0.5509 / 0.6868 | 0.4000 / 0.5800 / 0.6500 |
| Llama-2-7b-chat-hf | forget05 | 0.1421 / 0.5302 / 0.7058 | 0.6284 / 0.5269 / 0.7090 | 0.0900 / 0.5300 / 0.7100 |
| Llama-2-7b-chat-hf | forget10 | 0.3222 / 0.5466 / 0.7170 | 0.2812 / 0.5667 / 0.7135 | 0.4200 / 0.5400 / 0.7300 |
| Llama-3.2-1B-Instruct | forget01 | 0.0068 / 0.5880 / 0.5439 | 0.7659 / 0.5357 / 0.6786 | 0.9200 / 0.5600 / 0.6600 |
| Llama-3.2-1B-Instruct | forget05 | 0.2205 / 0.4400 / 0.7048 | 0.1779 / 0.4745 / 0.7013 | 0.1400 / 0.4500 / 0.7000 |
| Llama-3.2-1B-Instruct | forget10 | 0.0446 / 0.5228 / 0.6425 | 0.0299 / 0.5385 / 0.6362 | 0.0200 / 0.4600 / 0.7000 |

相對原版的有號差值（修正版−原版）：

| 模型 | split | ΔFQ | ΔMU | ΔTR |
|---|---|---:|---:|---:|
| Llama-2-7b-chat-hf | forget01 | 0.5643 | -0.0711 | 0.1375 |
| Llama-2-7b-chat-hf | forget05 | 0.4864 | -0.0033 | 0.0033 |
| Llama-2-7b-chat-hf | forget10 | -0.0410 | 0.0200 | -0.0035 |
| Llama-3.2-1B-Instruct | forget01 | 0.7592 | -0.0523 | 0.1346 |
| Llama-3.2-1B-Instruct | forget05 | -0.0426 | 0.0345 | -0.0035 |
| Llama-3.2-1B-Instruct | forget10 | -0.0147 | 0.0157 | -0.0063 |

六格修正版更新核對：

| 模型 | split | Trainer / DS / 強制邊界 | microsteps | 實際 epoch | KS D | KS 樣本數 |
|---|---|---|---:|---:|---:|---|
| Llama-2-7b-chat-hf | forget01 | 20 / 20 / 20 | 100 | 10 | 0.1750 | 40 / 40 |
| Llama-2-7b-chat-hf | forget05 | 70 / 70 / 70 | 500 | 10 | 0.0750 | 200 / 200 |
| Llama-2-7b-chat-hf | forget10 | 130 / 130 / 130 | 1000 | 10 | 0.0700 | 400 / 400 |
| Llama-3.2-1B-Instruct | forget01 | 20 / 20 / 20 | 100 | 10 | 0.1500 | 40 / 40 |
| Llama-3.2-1B-Instruct | forget05 | 70 / 70 / 70 | 500 | 10 | 0.1100 | 200 / 200 |
| Llama-3.2-1B-Instruct | forget10 | 130 / 130 / 130 | 1000 | 10 | 0.1025 | 400 / 400 |

原版 forget01 seeds0/1/2：

| 模型 | seed | FQ | MU | TR |
|---|---:|---:|---:|---:|
| Llama-2-7b-chat-hf | 0 | 0.0143 | 0.6219 | 0.5493 |
| Llama-2-7b-chat-hf | 1 | 0.0143 | 0.6185 | 0.5473 |
| Llama-2-7b-chat-hf | 2 | 0.0286 | 0.6229 | 0.5525 |
| Llama-3.2-1B-Instruct | 0 | 0.0068 | 0.5880 | 0.5439 |
| Llama-3.2-1B-Instruct | 1 | 0.0030 | 0.5881 | 0.5381 |
| Llama-3.2-1B-Instruct | 2 | 0.0541 | 0.5879 | 0.5447 |

原版 forget01 三 seed 描述統計（sample std 使用 n−1；n=3）：

| 模型 | 指標 | mean | sample std | min | max |
|---|---|---:|---:|---:|---:|
| Llama-2-7b-chat-hf | forget_quality | 0.0191 | 0.0083 | 0.0143 | 0.0286 |
| Llama-2-7b-chat-hf | model_utility | 0.6211 | 0.0023 | 0.6185 | 0.6229 |
| Llama-2-7b-chat-hf | forget_truth_ratio | 0.5497 | 0.0026 | 0.5473 | 0.5525 |
| Llama-3.2-1B-Instruct | forget_quality | 0.0213 | 0.0285 | 0.0030 | 0.0541 |
| Llama-3.2-1B-Instruct | model_utility | 0.5880 | 0.0001 | 0.5879 | 0.5881 |
| Llama-3.2-1B-Instruct | forget_truth_ratio | 0.5423 | 0.0036 | 0.5381 | 0.5447 |

這三個 seeds 中，兩模型原版 forget01 FQ 的範圍均遠低於文件的 .40 / .92，MU與TR跨seed波動也小於原版和修正版seed0的差異。因此本次樣本未顯示只換seed就能消除forget01差距；n=3仍不足以估計完整seed分布。FQ是KS p-value，40個樣本時對KS D的離散變化敏感，相同FQ不表示權重或輸出完全相同。修正版只有seed0，不能把原版的std當作修正版的不確定性。

修正 policy 為 ceil_epoch_and_sync_ds_boundary_v1：以 ceil(dataloader batches / accumulation) 計算更新計畫，在 training_step 進入 forward/backward 前把 active DS engine 邊界設為 Trainer 的 sync_gradients，處理 epoch 尾端不完整組。套件、NPO objective及既有loss scaling維持原样。本次同時改變訓練長度、尾端更新與相應scheduler時間軸，沒有拆開兩項修正做因果消融，也沒有驗證所有loss/gradient normalization問題。10 epochs修正結果不應被當作原文件訓練流程的完全等價實作。

所有組都用single GB200 / ZeRO3 / FlashAttention2 / microbatch4 / accumulation8，名義globalbatch32、lr1e-5；修正版epoch尾端短組只有8（f01/f05）或16（f10）個examples。原版forget01實際epoch5、Trainer10 / DS6 / micro50；其他原版split計数見matrix-report-20261005.md。docs/repro.md使用2×L40s，其8perdevice×2×4=64的算式與聲稱32不一致。本次未改成64，也未改原loss公式去追逐文件分數。版本為Torch2.7.0a0容器CUDA12.8、Transformers4.51.3、Accelerate0.34.2、DeepSpeed0.15.4、FlashAttention2.7.3、bitsandbytes0.50.2。兩卡workload仍受1GPU硬quota限制，未重複提交或干預其他workload。

核對：六原版seed0加十新增full runs均重新獨立重算FQ（KS）、MU（九項harmonic mean）、TR；新增十格與原始summary差值均0。樣本數40/200/400逐split核對，checkpoint權重分片按索引存在且大小符合manifest。巨大權重保留PVC，未下載或雜湊。真實partial-group smoke通過Trainer=DS=4,micro20,epoch2，只作門檻驗證，不列入正式結果。

證據：evidence-followup-20261005.zip / evidence-followup-20261005/ 包含十新增與六baseline的原始TOFU_EVAL、TOFU_SUMMARY、auditing.log、training/eval commands、Hydra設定、Trainer/runtime計数、官方retain references、執行器source與docs/repro.md快照；逐檔SHA256驗證。舊auditing.log的200v200標籤不適用其他split，使用matrix與followup的實際樣本數。完整精度為followup-independent-audit-20261005.json、forget01-seed-statistics-20261005.json。追蹤完成後暫停；未commit/push。
