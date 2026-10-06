# NPO 梯度累積的 loss scaling：疑點、證據與待驗證項目

> **17:03大型模型最終結果：** Llama-3.1-8B與Llama-2-7B新版native短程皆完成：正式訓練raw NPO loss到DS backward實際完整组除8／尾端除2，Trainer=DS4、micro20、epoch2，同DS explicit-mean梯度四項scale1／cos1／殘差0；四項原autograd 5% gates仍未過。這些結果支持實際累積分母正確，沒有證實完整autograd／DS等價，更沒有追溯修正歷史safety。版本、完整checkpoint SHA、原始證據與未解限制見[大型模型驗證](large-model-native-validation-20261006.md)。

> **16:48大型模型追加：** 新版Llama-3.1-8B的forget01 seed0短程訓練已直接記錄20次raw NPO loss→DS backward：完整組除8、尾端組除2，Trainer=DS4／micro20／epoch2、邊界8／10／18／20。同DS explicit-mean梯度control完整／尾端皆scale1、cosine1、殘差0；原autograd comparison兩個5% gates仍未通過。這是另一次原生短程驗證，沒有追溯更改September safety訓練，也不是10-epoch reproduction或所有梯度等價性驗收。詳見[大型模型紀錄與原始證據](large-model-native-validation-20261006.md)。

> **10:34最終結論範圍：** 有限驗證完成。舊H100／GB200漏除accumulation已有受控梯度實測；真實1B相同TOFU frozen batch對同DS explicit-mean control，舊完整8批／短組2批為8／2倍、新版原生為1／1倍。新版native三split seed0已完成更新計數、完整checkpoint與評估稽核，未重現全部docs數值。真實1B原autograd reference的16項5% gate仍未通過，沒有宣稱完整autograd／DS等價，也沒有追溯修正歷史實驗。下方保留當時疑點與查核過程；最終實測與限制見[驗證紀錄](environment-gradient-validation-20261006.md)。

> **同日實測進度：** 新增controlled gradient audit已在真實NPO→Trainer→Accelerate→ZeRO-3路徑觀測GB200完整8批FP32梯度約8倍、雙H100完整4批的forget／retain／combined梯度約4倍，bf16 tiny control亦通過。真實Llama-2-7B雙H100、bf16完整四批亦完成：投影倍率3.97807927、cosine0.99992089，消除倍率後relative error1.26%。這證實受控舊環境的漏除，而非已驗證每個歷史run；7B此量測仍使用合成固定tokens。新版與TOFU梯度測試待續接。本文下方保留原疑點查核時的狀態。最新範圍與限制見[環境與梯度驗證紀錄](environment-gradient-validation-20261006.md)。

核對日期：2026-10-06。範圍為本次 GB200 的 Transformers 4.51.3／Accelerate 0.34.2／DeepSpeed 0.15.4，以及 repo 自訂 NPO loss。

> **09:48真實TOFU實測：** 同1B參數及frozen TOFU batch的forget／retain／combined，完整8批與尾端2批，相對同DeepSpeed explicit-mean control，舊版為8／2倍、新版為1／1倍，cosine皆1，初始／最終參數SHA不變。這是scaling分母的實測證據；原autograd／DS bf16 comparison仍超過5% gate，差異保留、未宣稱完全等價。新版forget01正式訓練已核Trainer=DS20、micro100、epoch10，但評估遇bf16→NumPy錯誤，正從原checkpoint恢復。詳細數值、版本偏離與限制見[驗證紀錄](environment-gradient-validation-20261006.md)。

> **同日 upstream 查核補充：** 最新 `main`（`17cbbc8`）預設已升到 Transformers5.5.4／Accelerate1.13.0；標準NPO巢狀輸入得到 `num_items_in_batch=None`，新版Trainer會按當次實際microbatch數除loss。因此本文的漏除疑點限於本次固定舊環境，不能稱為最新版或原paper已證實的共同缺陷。新版源碼處理仍未在本次GB200實測梯度等價性；完整來源、歷史版本、相關論文與切換驗收見[upstream查核](upstream-status-20261006.md)。既有六格corrected未改loss的分類不變。

**以下為開始實測前的源碼查核紀錄，保留當時證據層級：** 當時證據支持「累積梯度可能沒有按預期平均」的源碼推論，尚未完成實際梯度等價性測試。October 六格corrected seed0只驗證更新次數與DS邊界，未改loss scaling；後續新版實測是另三個native runs，最終結果見本文開頭與驗證報告。

## 1. 這裡的 loss scaling 是什麼

這裡指 gradient accumulation 的正規化：多個 microbatches 合成一次 optimizer 更新時，如何加權各批 loss／gradient。它與 FP16 用來避免下溢的 dynamic loss scaling 是不同概念；本次訓練使用 bf16。

假設每個 microbatch 等大小，且每批 loss 都是相同定義的平均，若要等價於整個更新組的平均 loss，累積 K 批應使用：

$$L_{update}=\frac{1}{K}\sum_{i=1}^{K}L_i$$

對 microbatch4、accumulation8 的完整組，這表示平均32個 examples。若直接相加，clipping／optimizer 前的梯度是這個平均版本的8倍。這是上述假設下的數學關係，**不是本次訓練已實測的梯度倍率**。

Adam 的自適應狀態、epsilon、gradient clipping 等會影響參數更新，因此也不能把「原始梯度8倍」直接解讀為「learning rate或參數更新8倍」，更不能宣稱只把LR除以8就完全等價。

## 2. 固定版本下的程式路徑

| 層級 | 已核對的行為 | 依據 |
|---|---|---|
| Repository NPO | `compute_loss` 接收 `num_items_in_batch`，但沒有使用它；組合 forget NPO與retain loss。NPO helper對序列 loss取mean，並非在此顯式除以accumulation。 | [npo.py](../../../src/trainer/unlearn/npo.py)、[utils.py](../../../src/trainer/utils.py)、[grad_diff.py](../../../src/trainer/unlearn/grad_diff.py) |
| GB200 runtime | 保存 `model_accepts_loss_kwargs=true`、Trainer accumulation8、Accelerator accumulation1、DS accumulation8。 | [run_probe.py](../../../scripts/reproduction/npo/gb200/run_probe.py) 記錄的原始runtime；摘錄見下方。 |
| Transformers4.51.3 | 該 training_step 的accumulation除法受 `model_accepts_loss_kwargs` 控制；此flag為true會跳過該除法。DS分支另外傳入 `scale_wrt_gas=false`。 | [Trainer source](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/trainer.py#L3451) |
| Accelerate0.34.2 | DS分支不執行一般路徑的accumulation除法，將loss與kwargs交給DS wrapper。wrapper再呼叫engine.backward及engine.step。 | [Accelerator source](https://github.com/huggingface/accelerate/blob/v0.34.2/src/accelerate/accelerator.py#L2000)、[DS wrapper](https://github.com/huggingface/accelerate/blob/v0.34.2/src/accelerate/utils/deepspeed.py#L149) |
| DeepSpeed0.15.4 | backward只有在啟用 `scale_wrt_gas` 時才執行accumulation除法；engine本身可覆寫此flag，需在梯度測試時一併核對。 | [Engine source](https://github.com/microsoft/DeepSpeed/blob/v0.15.4/deepspeed/runtime/engine.py#L1809) |

以已完成的 `gb200_1gpu_zero3_flash_attention_2_s0_followup_Llama-3.2-1B-Instruct_forget05_corrected` 為例，保存的 `checkpoint/runtime_rank0.json` 包含以下欄位摘錄：

```json
{
  "world_size": 1,
  "deepspeed_enabled": true,
  "micro_batch": 4,
  "gradient_accumulation_steps": 8,
  "accelerator_gradient_accumulation_steps": 1,
  "model_accepts_loss_kwargs": true,
  "engine_gradient_accumulation_steps": 8,
  "correction": {
    "policy": "ceil_epoch_and_sync_ds_boundary_v1",
    "loss_scaling": "unchanged pinned NPO/Trainer behavior"
  }
}
```

原始runtime小檔保存在本機 `results/reproduction/npo/gb200/evidence-followup-20261005/` 及其SHA256驗證的archive；此摘錄不是新增量測。已發布的[follow-up audit](../../../results/reproduction/npo/gb200/followup-independent-audit-20261005.json)亦保存各corrected run的policy及loss-scaling未改狀態。

**推論：** 自訂NPO沒有使用跨批分母、Trainer跳過除法、Accelerate在DS路徑不代做，而Trainer停用DS的GAS縮放，這些條件形成「直接累積各microbatch梯度總和」的疑點。`model_accepts_loss_kwargs=true` 不能證明自訂loss真的按跨批樣本／token數正規化。仍需量測完整呼叫鏈、engine覆寫狀態及clipping前梯度，確認实际行為與倍率；不能把相近FQ／MU／TR當作梯度等價性證據。

## 3. Epoch 尾端短組是另一項驗收

若目標是每個實際更新組的平均loss，只有2個microbatches的尾端組應按該組資料計算分母，而非自動沿用8：

| 等大小microbatch的例子 | 該組平均loss | 直接相加相對平均值 | 固定除以8相對平均值 |
|---|---|---:|---:|
| 完整8批 | `(L1+…+L8)/8` | 8倍 | 1倍 |
| 尾端2批 | `(L1+L2)/2` | 2倍 | 1/4倍 |

這是兩種可能正規化錯誤的示例，不是說本次程式同時發生兩者。若研究預先選擇固定名義batch分母，應明確寫出該objective／短組權重，而不能稱為與實際短組平均等價。

本次corrected forget01／05尾端為2批（8 examples），forget10尾端為4批（16 examples）。更新邊界修正確保尾端執行更新，但不會自動選擇或修復loss分母。[上游issue #38837](https://github.com/huggingface/transformers/issues/38837)報告固定accumulation分母造成尾端縮放不一致；它是相關問題的來源，不是本repo已被證明有同一錯誤的結論。

## 4. 為什麼不能直接全部除以 accumulation

NPO forget項對每個序列的NPO值取平均；retain項来自語言模型NLL，需核對有效token分母。序列長度或各microbatch有效token數不同時，「平均各microbatch的平均loss」未必等於「一次大batch的loss」。

因此要先凍結預期objective：forget按examples／sequences加權、retain按預定token或sequence定義加權，再分別測兩項與合併loss。只改 `model_accepts_loss_kwargs`、只啟用DS除以8或只改LR，可能沒有處理短組、token權重，甚至引入重複除法；不能未驗證就套用。

## 5. 對既有結果的分類

- **October六格corrected seed0：更新次數／epoch／DS邊界已驗證；梯度正規化未驗證、未修正。**
- October原版六格、forget01額外seeds及nonDS control保留原流程；不能套用DS分支推論替它們逐一判定梯度倍率。各模型、版本、flag需分別核對。
- August H100／Ada與September Llama-3.1 safety reports沒有本次loss-scaling驗證，不能直接推算它們的倍率。Safety的 `aggregate.corrected.json` 是評估修復，不是loss normalization修正。
- 舊結果仍描述其實際執行的recipe；問題未驗證不等於所有checkpoint或結論失效，但不能宣稱已滿足大batch梯度等價性。

完整更新次數與逐報告分類見[更新次數問題文件](update-count-mismatch-20261006.md)。未來若新增loss normalization policy，应另記policy與run tag，不與既有原版或只修正更新次數的seeds合併統計。

## 6. 剩餘實驗前建議的驗證（尚未執行）

1. 固定模型參數、reference、forget／retain樣本、mask與dtype，關閉dropout，從相同初始狀態比較一次大batch和分批累積；不要在microbatches之間更新參數。
2. 分別測完整組與尾端短組，涵蓋不同有效token長度；分開核對forget、retain及合併objective。
3. 保存clipping前梯度向量／norm、cosine similarity、relative error及選定參數差異；不只比較顯示的loss或clipping後norm。先用可控FP32基準核對權重定義，再於實際GB200／bf16／DS路徑驗證；容差依基準數值誤差預先訂定。
4. 在真實呼叫鏈記錄loss輸入、實際分母、`model_accepts_loss_kwargs`、backward kwargs、engine `scale_wrt_gas`與更新邊界，排除漏除／重複除法。通過後才核對一次optimizer更新、保存及reload。
5. 若確認需修正，保留同模型同資料的原recipe對照，建立新normalization policy，並再次驗收實際epochs／Trainer／DS updates。通過後再展開剩餘正式矩陣。

這次GitHub更新只補充問題、證據限制與驗證設計，沒有修改loss／Trainer、啟動訓練、重算既有分數或宣稱已完成梯度測試。
