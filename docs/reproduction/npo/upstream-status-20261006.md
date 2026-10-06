# OpenUnlearning upstream 查核：更新次數、loss scaling 與相關論文

> **17:03大型模型追加已完成：** 新版native Llama-3.1-8B與Llama-2-7B各forget01 seed0 smoke4，皆核Trainer=DS4／micro20／epoch2，實際backward完整組除8／尾端除2、完整checkpoint reload評估成功。同DS explicit-mean四項scaling gates通過；四項原autograd comparisons未過5% gates。僅短程驗證，不是10-epoch reproduction或safety驗收，不追溯修改歷史分類。原始證據、SHA及版本偏離見[大型模型驗證](large-model-native-validation-20261006.md)。

> **10:34最終實驗範圍：** 有限驗證完成。新版隔離stack完成二十項old／new tiny與十六項真實1B synthetic／TOFU同DS reference對照，舊完整／短組8／2倍、新版1／1倍。新版native三split seed0皆獨立核Trainer=DS20／70／130、micro100／500／1000、epoch10、完整checkpoint／評估。FQ依序.265687／.270474／.054053，未重現全部docs數值。原autograd／DS bf16差異仍未定位；保留容器Torch与兩個eval輸出cast偏離，不能稱為完整upstream預設環境。下方早期「尚未驗證」文字是當時查核狀態；最終證據以[驗證紀錄](environment-gradient-validation-20261006.md)為準。

> **同日使用者授權的驗證進度：** 已開始[獨立環境與梯度驗證](environment-gradient-validation-20261006.md)。舊GB200與雙H100的controlled tiny-model梯度已有漏除accumulation實測；雙H100真實7B的完整四批／尾端一批也已完成，完整四批投影倍率3.97807927，八個受控case的雙rank證據已獨立稽核。此7B測試仍用合成固定tokens。新版尚未完成環境／梯度／正式計數驗收。本文件下方保留源碼查核時的狀態，不能把新版源碼處理視為實驗已通過。

查核日期：2026-10-06。本文補充本次 GB200 reproduction 的版本範圍，不修改既有實驗、分數或訓練環境。

> **09:48真實模型與TOFU補充：** 十六項同DS explicit-mean control已獨立核對，舊版完整／短組8／2倍、新版1／1倍，所有cosine1；原autograd reference比較仍未通過。新版native forget01正式訓練亦完成Trainer=DS20、micro100、epoch10，checkpoint已保存。評估在bf16→NumPy序列化失敗；隔離evaluator只加兩個輸出cast，從原checkpoint恢復評估再續缺少split。新版三split的最終scores仍未完成，且上述兩個eval cast是固定upstream commit的明確兼容性偏離。[最新證據與限制](environment-gradient-validation-20261006.md)。

> **09:03新版實測補充：** 保留GB200容器Torch的隔離TF5.5.4／Accel1.13.0／DS0.15.4環境已完成imports與十個controlled tiny gradient cases，完整與短組對mean-micro reference均約1倍；舊版十個對照約為窗口批數倍。真實1B舊版bf16的autograd comparison因殘差超門檻停止，已保留失敗並追加同DS explicit-mean control續接。正式新版三split、真實TOFU與真實模型等價性仍未驗收；[最新執行紀錄](environment-gradient-validation-20261006.md)。

**以下保留開始實測前的源碼查核紀錄：** 不能將Transformers4.51.3／Accelerate0.34.2的更新mismatch與loss-scaling問題直接描述為最新版或原論文的共同缺陷。最新upstream預設依賴包含對應處理；當時尚未執行新版實測，現已完成的範圍與未解限制見本文開頭及驗證報告。

## 1. 核對版本與證據層級

GitHub API 核對的 upstream `main` 為 `17cbbc87192e6934deb92875c359c91bbd837fb4`，2026-09-29 合併 [PR #191](https://github.com/locuslab/open-unlearning/pull/191)。查核時 [GitHub Releases](https://github.com/locuslab/open-unlearning/releases) 沒有正式 Release，因此本文的「目前版本」指此 commit 與其預設 requirements，不指本機或 PVC 已安裝的版本。

| 範圍 | Transformers | Accelerate | DeepSpeed | 可確認的事實 |
|---|---|---|---|---|
| 原論文發表期間的 repository snapshot | 4.45.1 | 0.34.2 | 0.15.4 | [2025-05-23 snapshot](https://github.com/locuslab/open-unlearning/blob/b71de54c179408d447bc383c86fd1fafcc99dc14/requirements.txt) 與 [2025-10-07 snapshot](https://github.com/locuslab/open-unlearning/blob/a1082537e18a3f58677ab31295dedf03d427bb57/requirements.txt) 均固定此組合；不是論文每個 run 的完整 runtime manifest。 |
| 本次 October GB200 實驗 | 4.51.3 | 0.34.2 | 0.15.4 | 更新次數已有實測；loss normalization 尚未完成梯度等價性測試。 |
| 最新 upstream `17cbbc8` 的預設依賴 | 5.5.4 | 1.13.0 | 0.15.4 | [requirements](https://github.com/locuslab/open-unlearning/blob/17cbbc87192e6934deb92875c359c91bbd837fb4/requirements.txt)；源碼已有下述處理，本次尚未執行此新版矩陣。 |

最新 requirements 亦固定 Torch 2.9.1、bitsandbytes 0.49.2；本次 GB200 使用 NVIDIA 容器的支援 GB200 PyTorch，不能只替換兩個套件就宣稱完全採用 upstream 預設環境。需另外保存實際 manifest 與相容性驗證。

## 2. 最新程式路徑如何處理更新次數

在 [Transformers 5.5.4 Trainer](https://github.com/huggingface/transformers/blob/v5.5.4/src/transformers/trainer.py)，`set_initial_training_values` 對有限 dataloader 使用整數商加上非零餘數，相當於 `ceil(len(dataloader) / gradient_accumulation_steps)`，再依設定 epochs 規劃總更新數。training loop 在每個完整累積窗口與 epoch 尾端設定 `sync_gradients`。

[Accelerate 1.13.0 Accelerator](https://github.com/huggingface/accelerate/blob/v1.13.0/src/accelerate/accelerator.py) 將此 `sync_gradients` 傳給 DS wrapper；[DeepSpeedEngineWrapper](https://github.com/huggingface/accelerate/blob/v1.13.0/src/accelerate/utils/deepspeed.py) 在 backward 前呼叫 `engine.set_gradient_accumulation_boundary(is_boundary=sync_gradients)`，並只在該邊界呼叫 engine.step。

因此，最新預設依賴已包含本次更新修正的兩項核心機制：向上取整的 epoch 計畫與 Trainer／DS 邊界同步。不能再籠統宣稱 upstream 最新版仍有完全相同的更新 mismatch。對本次 microbatch4、accumulation8、10 epochs 的三 splits，仍應實測 Trainer 與 DS optimizer updates 分別為20／70／130、microsteps100／500／1000，以及實際 epoch10。

此結論不等於所有 overflow、resume、分散式拓撲及自訂 trainer 已驗證。若改採新版，不應把現有 `ceil_epoch_and_sync_ds_boundary_v1` monkey patch 原封不動疊加；新版需建立獨立 run tag、保留原生行為並重新稽核。

## 3. 最新 NPO 路徑如何處理 accumulation scaling

本次舊環境的疑點與驗證設計仍保留於 [loss-scaling 文件](loss-scaling-20261006.md)。新版的關鍵差異不是 NPO objective 被重寫，而是 Trainer 多了一個適用本輸入格式的除法條件。

1. upstream [NPO](https://github.com/locuslab/open-unlearning/blob/17cbbc87192e6934deb92875c359c91bbd837fb4/src/trainer/unlearn/npo.py) 使用 `inputs["forget"]` 與 `inputs["retain"]`；[collator](https://github.com/locuslab/open-unlearning/blob/17cbbc87192e6934deb92875c359c91bbd837fb4/src/data/collators.py) 保留這個巢狀結構，`labels` 位於子字典。
2. Trainer 5.5.4 的 `_get_num_items_in_batch` 要求頂層含 `labels` 才計數，所以此 NPO 路徑得到 `num_items_in_batch=None`。
3. `training_step` 在 `(not model_accepts_loss_kwargs or num_items_in_batch is None)` 且沒有 `compute_loss_func` 時，除以 `current_gradient_accumulation_steps`。此值是當次實際取得的 microbatch 數，包含尾端短組；即使模型 flag 為true，也會因分母計數為None而執行除法。
4. DS 路徑仍傳 `scale_wrt_gas=False`，避免 DS 再除一次；Accelerate 的 DS 分支不另做一般累積除法。

**源碼判斷：在上述標準 NPO／collator／Trainer 條件下，本次舊版推導出的「漏除 accumulation」路徑已有對應處理。**尚未量測本次 GB200 新版的 clipping 前梯度，不能改標為「梯度等價性已實測通過」。

這項處理得到的是各 microbatch loss 的平均。retain NLL 有不同有效 token 數時，仍未必等價於一次大 batch 的 token 平均；forget 序列平均與 retain token 權重必須按預定 objective 分開驗證。自訂 collator、頂層 labels、`compute_loss_func` 或 engine scaling 覆寫也會改變呼叫路徑，不能概括所有方法都已正確。

## 4. GitHub 維護者是否曾注意 scaling 差異

[PR #175](https://github.com/locuslab/open-unlearning/pull/175) 於2026-03-07合併，從4.45.1升到4.51.3。作者報告 accumulation不為1時 logged loss尺度與 unlearning trajectory有所變化，並討論 `model_accepts_loss_kwargs=False` 的效果。[PR #191](https://github.com/locuslab/open-unlearning/pull/191) 再次報告新版 logged loss convention 改變，而測試的 unlearning trajectory大致相近。

這些是版本升級的回歸觀察，不能當作 clipping前梯度等價性、所有 short-group 權重或本次 ZeRO3設定已被驗證的證據。反過來，也不能說上游從未注意 scaling差異。

## 5. 原 OpenUnlearning paper 是否沒有處理這兩個問題

查閱 [OpenUnlearning paper](https://arxiv.org/pdf/2506.12618) 與附錄，沒有找到針對本次兩項問題的明確更新計數一致性或梯度累積等價性驗證。附錄D描述後續主要 meta-evaluation／benchmark使用 Llama-3.2-1B、單張A100、BF16、batch size32與paged AdamW；這不足以還原每個 run 的 microbatch、accumulation、實際 optimizer updates及完整版本。

論文發表期間的 repo snapshot 使用4.45.1，而非本次4.51.3。[Trainer 4.45.1](https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py) 不會在 DS backward kwargs中關閉 `scale_wrt_gas`；[Accelerate 0.34.2 wrapper](https://github.com/huggingface/accelerate/blob/v0.34.2/src/accelerate/utils/deepspeed.py) 將 kwargs轉交，[DS 0.15.4 engine](https://github.com/microsoft/DeepSpeed/blob/v0.15.4/deepspeed/runtime/engine.py) 預設會按GAS縮放。因此不能把4.51.3的漏除疑點直接套到原paper。4.45.1也有floor步數規劃，但迴圈與更新條件不同，不能直接沿用本次5／8.64／9.24 epochs或DS6／54／115的數字。

可支持的結論是：原論文未明確報告這兩項驗收，部分歷史訓練預算值得稽核；**目前證據不足以認定原論文所有結果受到同樣影響或整體失效。**需取得論文實際commit、啟動設定、runtime manifest、Trainer state與DS counters逐run確認。論文主結果、本repo的 `docs/repro.md` 方法重現表及本次GB200矩陣，不能僅因使用同一框架就視為同一批訓練。

## 6. 其他使用 OpenUnlearning 的論文

以下依作者論文／官方實作確認實際使用方式，非完整引用清單；使用框架不代表繼承同一套版本與問題。

| 論文 | 作者公開描述的使用方式 | 主要來源 |
|---|---|---|
| LLM Unlearning with LLM Beliefs | 明確以OpenUnlearning進行實驗、使用baselines及評估；方法程式合併回upstream。 | [論文 §1、§6與附錄E](https://arxiv.org/html/2510.19422v2) |
| RepSelect: Robust LLM Unlearning via Representation Selectivity | 官方repo直接fork OpenUnlearning，沿用benchmark harness、baseline implementations與evaluation pipeline。 | [官方repo](https://github.com/filyp/RepSelect)、[論文](https://arxiv.org/abs/2606.17168) |
| Oblivionis: A Lightweight Learning and Unlearning Framework for Federated Large Language Models（AAAI2026） | 官方repo明確表示基於OpenUnlearning開發。 | [官方repo](https://github.com/fyzhang1/Oblivionis)、[AAAI論文](https://ojs.aaai.org/index.php/AAAI/article/view/40045) |
| Measure, Don't Optimize: Forecasting Recovery in LLM Unlearning | 稽核OpenUnlearning發布的398個模型、涵蓋8種方法；使用公開checkpoints與評估，不能因此假設其訓練均沿用同一Trainer。 | [論文，Benchmark, Models and Behavioral Metric](https://arxiv.org/html/2608.11408v1) |

要判定各論文是否受影響，仍需個別取得套件、loss／collator、microbatch、累積設定及實際updates；不能僅從引用關係判定。

## 7. 後續採新版的建議與未執行項目

建議先在隔離環境驗證新版，通過後固定commit與實際依賴作為剩餘正式實驗環境。這是建議，**本次文件更新沒有切換環境、修正loss或提交任何新訓練**。

1. 保留支援GB200的容器PyTorch，先核對新版依賴相容性；記錄與upstream requirements的差異。
2. 先以Llama-3.2-1B／forget01驗證完整與尾端累積窗口的Trainer／DS updates及實際epochs，再涵蓋三splits的正式計數。
3. 固定參數、reference、資料、mask，關閉dropout；按相同objective權重，比較完整8批與尾端短組的clipping前梯度向量、norm、cosine及relative error，分開測forget、retain與合併項。
4. 補一個與舊版相同模型／split／seed的新版完整對照，核對checkpoint、樣本、FQ／KS D／MU／TR；通過後才展開剩餘矩陣。
5. 新版使用獨立tag與manifest；原版、只修更新次數的corrected版與新版，不混合為同一seed平均。既有六格corrected與safety分類維持原判定，詳見[更新次數文件](update-count-mismatch-20261006.md)。

既有實驗仍是4.51.3固定環境的證據；upstream升級不會追溯改變其checkpoint、更新次數或梯度驗證狀態。
