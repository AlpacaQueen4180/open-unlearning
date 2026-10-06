# GB200 新版環境：Llama-3.1-8B 與 Llama-2-7B 短程驗證

2026-10-06 使用者追加授權：主要模型 Llama-3.1-8B 也需新版驗證，並保留原先 Llama-2-7B 的驗證。此輪為相容性、更新邊界、backward scaling 與 checkpoint 評估門檻，不是新的完整 reproduction 矩陣。

**17:03最終狀態：兩模型有限queue已DONE，所有14個stage returncode0；原始證據下載及兩個本機獨立稽核已完成。** 兩模型皆實測完整組除8／尾端除2、Trainer=DS4、micro20、epoch2及完整checkpoint評估；同DS explicit-mean scaling對照通過，但四個原autograd comparisons皆未通過5% gates。以下保留啟動與修復歷史，最終數值見文末。未跑新的10-epoch矩陣、沒有HEx-PHI Judge，不追溯改標任何September safety checkpoint。

以下為原始執行歷史：15:56臺灣啟動有限queue，PID **37690**，PVC task `/data/npo-gb200-20261004/model-validation-20261006`。[本機launch](../../../results/reproduction/npo/large-model-validation-20261006/launch.json)已保存。啟動前確認GPU空閒、沒有重複status／launch或runner PID；啟動後不追加poll，交由15分鐘排程。啟動時尚未取得驗收結果，後續修復及最終稽核見下方。

## 模型與版本範圍

- 優先 `open-unlearning/tofu_Llama-3.1-8B-Instruct_full`，與既有safety主實驗的公開full baseline類型一致；下載前由HF metadata鎖定revision並保存asset manifest。尚未核對September模型的完整權重hash，不宣稱與歷史checkpoint逐byte相同。不是自製full或原始Meta instruct baseline。
- 第二個模型 `open-unlearning/tofu_Llama-2-7b-chat-hf_full`；沿用此前固定revision `cc5b31c69127d5da881608e640f2b453c446435b`。
- 沿用已驗證隔離環境 `validation-20261006/venv`：TF5.5.4／Accelerate1.13.0／DeepSpeed0.15.4；容器Torch2.7.0a0、FlashAttention2.7.3、bitsandbytes0.50.2。upstream固定 `17cbbc87192e6934deb92875c359c91bbd837fb4`；保留之前兩個eval輸出float cast，SHA `acd98144cada6f02dea329bb78c77e317bd651d16fabbb576796253ae6c4ec7b`。
- TOFU revision `324592d84ae4f482ac7249b9285c2ecdb53e3a68`，eval reference revision `f31717fe522c2a6725ca0024980f51169a084ca1`，兩模型各自對應retain99 reference及SHA，不混用不同模型tokenizer或reference。
- 不改舊venv、原loss、既有checkpoint或歷史safety分類。之前1B有限驗證仍為完成狀態，不重跑。

## 每個模型的有限驗證

1. 下載或沿用固定公開full資產，記revision、model path、retain reference及SHA。
2. 由原repo資料／collator與各模型tokenizer鎖定forget01 40筆，seed0 retain採樣，保存mask、有效token數、配置與frozen batch SHA。token tensors只留PVC。
3. 新版真實NPO→Trainer→Accelerate→DS的bf16、Eager受控梯度：完整8 microbatches（前32筆）、尾端2（後8筆），combined loss；相同DS engine明確loss/K對照、SGD lr0、clip0。保留原autograd mean與single-large比較及門檻，任何未通過都如實記錄。這是梯度隔離測量，不是正式optimizer更新。
4. FlashAttention2、ZeRO3、world1、micro4／acc8／global32，forget01、seed0、NPO alpha=gamma1／beta.1／retain NLL、LR1e-5、paged_adamw_32bit，顯式max_steps4。預期兩個實際epochs、20個microbatches、Trainer=DS4，實際更新邊界8／10／18／20。尾端只有8 examples。
5. [short_native_probe.py](../../../scripts/reproduction/npo/gb200/short_native_probe.py)只記錄raw NPO loss、傳入DS backward的loss與window／boundary，不改loss或邊界。預期完整窗口除8、尾端除2；比對實際scalar比例，避免只看logging誤判。smoke存完整checkpoint，再跑TOFU evaluation及原始score稽核。

每個smoke checkpoint的評估結果僅代表短程模型，不與10-epoch docs或September safety結果當作同條件比較，不宣稱已完成正式reproduction／安全性驗證。沒有HEx-PHI Judge或新safety矩陣。

## 執行器與後續排程

[validate_large_models.py](../../../scripts/reproduction/npo/gb200/validate_large_models.py)以exclusive lock與status防重複；先8B、再7B依序執行，遇失敗保留commands／logs／status並停止。新文件名與run目錄隔離既有結果。helpers已gzip上傳並核SHA：

| Helper | SHA256 |
|---|---|
| short_native_probe.py | `31ab3e1a903a1fa6d3918021c0db299c1163c9428a869ee5b283834eccf6e2df` |
| capture_large_model_batch.py | `ab0066e55860fa893e37334c84b2a8aae08993b9b63f02070418e74fcb0cc7a4` |
| validate_large_models.py | `d6a2394477046899d367bf55d85441fdd287dab0676f52b1a1817dabfc4141a0` |

啟動前已通過helper語法檢查、命令template驗證；尚未取得模型執行結果。每次排程只做一次必要snapshot，正常進行保持安靜，連線暫時失敗最多重試一次。完整checkpoint存在而評估失敗時只恢復評估；需要修復／重訓先確認舊PID退出、GPU空閒，保留原失敗並用新tag，完成stage不重跑。模型權重、frozen batch與raw vectors不下載或提交Git；小檔原始證據下載核SHA，核shard index／必要layers／filelength／完整權重SHA、實際更新與scalar比例、樣本40／40和獨立FQ／KS D／MU／TR。兩模型門檻、報告與證據完成後再次暫停追蹤。

## 16:13排程：metadata序列化失敗；16:17修復續接

本輪單次[snapshot](../../../results/reproduction/npo/large-model-validation-20261006/snapshot-20261006-1613.json)取得原queue FAILED，8B `prepare_assets` rc0、`freeze_tofu` rc1；PID37690已為defunct、沒有GPU程序。8B公開full已下載並固定revision **`1a5c5b1a557f8c99bdadecd5168ebd03f640b00e`**，retain99 reference SHA **`d307bb83ea3e3409fd01cc50908501ea16aba49e9ed6a779f8ceb75ec2e5503f`**。梯度／訓練尚未開始。

失敗發生在40筆batch已成功保存後，`OmegaConf.to_container(cfg, resolve=True)` 試圖展開 `paths.work_dir` 的Hydra runtime interpolation；compose未設定HydraConfig，故metadata寫入失敗。此為驗證執行器的設定序列化問題，不能解讀為8B訓練、CUDA或scaling失敗。已下載13個原始小檔（source／commands／完整logs／FAILED status／assets與frozen proof）並逐一核SHA，保存於[attempt0](../../../results/reproduction/npo/large-model-validation-20261006/attempt0/)。

原frozen pt **180,924 bytes**，SHA **`7182432e7af902dc6d45246fa2dbb9384b230e05cbee83c894d5c7d49e589733`**，只留PVC。修復的capture支援 `--resume-metadata`：CPU載入原pt、不重新取樣／tokenize；完整config保留interpolation，data／template另行解析。metadata恢復前後pt SHA一致，40筆有效token分母核對；前置恢復測試returncode0。

新的task `/data/npo-gb200-20261004/model-validation-20261006-r1` 已於UTC08:17:58（臺灣16:17）啟動PID **38122**，[launch](../../../results/reproduction/npo/large-model-validation-20261006/attempt1/launch.json)確認。重用原成功asset stage、原模型revision及原40筆batch，僅續接metadata、梯度與smoke／評估；新run目錄在r1，不覆寫原失敗。啟動命令未回傳stdout，因此只補讀固定launch檔確認PID，未追加訓練進度poll。

新版helper SHA：capture `ddb261ad660a6b2110c6ca268f6861b36e0049c54dccec51625acbedec7a705f`；queue `da3933a1191d595c87c87fed62785a2b00cfd2339f4f492ccaf587810d488ec8`；short probe不變。原helpers／SHA保留在原task/as-run與本機attempt0。此修復不改NPO、梯度或套件；helper語法檢查通過。下一輪只讀r1狀態，不把原FAILED視為仍需重啟；目前仍未取得此輪梯度／smoke訓練通過證據。

## 16:33 排程快照

本輪只做一次必要[snapshot](../../../results/reproduction/npo/large-model-validation-20261006/snapshot-20261006-1632.json)，壓縮短命令及完整JSON framing取得成功。r1 PID38122存活、phase RUNNING，GPU程序38857；8B metadata恢復與完整8批combined梯度stage returncode0，當前為尾端2批梯度。尚未下載完整梯度JSON及獨立稽核，不以stage returncode0宣稱模型驗收；smoke訓練／評估與7B仍待執行。沒有重啟或額外poll，交由下次排程。

## 16:48 排程：8B 短程結果完成並独立稽核；7B 仍執行中

本輪唯一[snapshot](../../../results/reproduction/npo/large-model-validation-20261006/snapshot-20261006-1647.json)確認8B七個stages已完成，7B當前為尾端2批梯度、queue仍RUNNING。随后僅下載已完成8B的固定證據，沒有再次查詢訓練進度。33個原始小檔逐一驗SHA；[獨立稽核](../../../results/reproduction/npo/large-model-validation-20261006/llama31-8b/independent-audit.json)核對原始JSON、loss trace、checkpoint proof及評估資料，原始證據保存在[8B目錄](../../../results/reproduction/npo/large-model-validation-20261006/llama31-8b/)。

| 項目 | 8B 實測結果 | 判定範圍 |
|---|---|---|
| 更新／實際資料消耗 | Trainer=DS=4、microbatches20、epoch2；邊界8／10／18／20 | 短程原生更新門檻通過；不是10 epochs |
| 真實 NPO loss → DS backward scalar | 完整組16行各除8；尾端4行各除2；`scale_wrt_gas=false` | 20行獨立核對通過，沒有漏除或重複除；probe只記錄 |
| TOFU Eager bf16梯度，同DS explicit-mean reference | 完整8／尾端2皆projected scale1、cosine1、殘差0；observed／reference各1次lr0更新，參數SHA不變 | 兩個matched DS scaling gates通過；SGD lr0／clip0是測量配置 |
| 原 autograd mean comparison | 完整8：projected scale0.99053171、cosine0.99831572、`relative_error_after_scale`0.05756247；尾端2：0.95959472／0.99450658／0.10099950 | **兩個原5% gates仍未通過**；不放寬門檻，不直接歸因bf16，不宣稱所有梯度等價 |
| autograd mean micro vs single-large | relative error0.00483423／0.00791938 | 本次combined objective兩組5% gates通過，不能延伸到其他objective或資料 |
| Checkpoint | 32 layers、291 tensors、單檔16,060,556,616 bytes；SHA `f31db39138a43c32778b4ee2bf59cdcb67cf829df137703aefada4a8d80f81a8` | 遠端CPU串流SHA及header／offset／shape／必要layers核對，權重未下載；沒有硬套1B格式 |
| 短程 TOFU evaluation | FQ0.09707484379785859、KS D0.275、MU0.6432044798574184、TR0.5263947704335518；40／40 | 本機exact integer KS／hmean／TR重算，SUMMARY最大delta2.78e-17 |

梯度及正式paged Adam訓練使用同一個固定公開TOFU full來源，測量參數hash `3eaa1821651dea7c6a1c1b371fa7713c6a68b78fe58ca2eb6344114206efb992` 與frozen batch hash已保存。以上證明新版8B的實際除數、更新邊界與checkpoint評估可運作，**不代表完整reproduction、安全性驗收或autograd／DS等價性全部通過**。7B尚未完成，下一輪只讀queue／7B必要狀態，不重跑或重下載已完成8B。

## 17:03 最終驗證：兩模型完成

唯一[排程snapshot](../../../results/reproduction/npo/large-model-validation-20261006/snapshot-20261006-1702.json)確認queue於16:56完成，14個stages全部rc0、runner為defunct、GPU空閒；未重跑已完成模型。7B另下載44個原始小檔核SHA，包含原metadata recovery的命令／log／metadata、最終queue status／完整log／upload manifest，以及實際gradient audit來源；加上8B33個及原失敗13個，共90個manifest entries逐一驗SHA。部分來源為跨模型重複保留，90不是去重後檔案數。下載資料、[7B独立稽核](../../../results/reproduction/npo/large-model-validation-20261006/llama2-7b/independent-audit.json)及[完整checkpoint proof](../../../results/reproduction/npo/large-model-validation-20261006/llama2-7b/checkpoint-proof.json)可核對。

兩模型皆在正式paged_adamw_32bit短程訓練直接記錄20行backward scalar：16行完整組除8、4行尾端除2，DS不再重複除；實際邊界8／10／18／20、Trainer=DS4、micro20、epoch2。這項除數量測與Eager／SGD lr0／clip0梯度隔離測量分開，不能把lr0梯度倍率解讀為Adam或learning-rate倍率。

| 模型 | 短程 FQ | KS D | MU | TR | 樣本／reference |
|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B | 0.09707484379785859 | 0.275 | 0.6432044798574184 | 0.5263947704335518 | 40／40 |
| Llama-2-7B | 0.006760732303569206 | 0.375 | 0.6231267438307369 | 0.5382433879138844 | 40／40 |

本機以exact integer lattice KS、9項harmonic mean及raw truth-ratio重算，SUMMARY最大delta分別2.78e-17／1.73e-18。原finalize helper的`ks_D_200v200`欄名硬寫200，本輪實際count為40／40，以原始value_by_index及獨立audit為準。這是4-update模型評估，不拿數值當成docs 10-epoch或歷史safety的同條件重現。

| 模型／窗口 | 同DS explicit-mean scale／cos／殘差 | 原autograd projected scale | 原autograd cosine | 原`relative_error_after_scale` | 原5% gate |
|---|---|---:|---:|---:|---|
| 8B／完整8 | 1／1／0 | 0.9905317101 | 0.9983157222 | 0.0575624660 | 未過 |
| 8B／尾端2 | 1／1／0 | 0.9595947215 | 0.9945065824 | 0.1009995016 | 未過 |
| 7B／完整8 | 1／1／0 | 0.9280949569 | 0.9922631595 | 0.1161236613 | 未過 |
| 7B／尾端2 | 1／1／0 | 1.0039521650 | 0.9981229626 | 0.0615994089 | 未過 |

`relative_error_after_scale`為扣除投影倍率後殘差norm／reference norm，並非再除以投影倍率後的另一種定義。本輪預期scaling為1，原gate按投影倍率距1及此殘差各5%判定；沒有放寬門檻。四個matched DS gates通過只隔離累積分母，**四個autograd gates未過的原因尚未定位，不宣稱autograd／DS完全等價，也不直接歸因bf16**。autograd mean micro vs single-large四個combined gates通過，7B relative error完整0.0050055350／尾端0.0135504348；不將此結果推廣到其他loss或資料。

7B固定model revision `cc5b31c69127d5da881608e640f2b453c446435b`、retain99 reference SHA `adf8058227db005f1354b1b3c9ef805c80bf1caecff946ec8f814f65911e74c0`、frozen batch SHA `fcd6f998521cb1f4996ff24961daa281f9e958129d50d7c31c3f458f3f0a7cb4`。受控梯度初始／最終參數SHA均 `607036a017c39a0d7730a95e6050b600c3697b5fd409e2227243f8ef037b69d3`；32 layers、291 tensors的完整訓練checkpoint單檔13,476,865,232 bytes，SHA **`4fa10ccb4daf5c7f385ac72920b4f68564a21a06f9fb769bc8ca61d21591b1ec`**。兩模型均由遠端CPU核所有safetensors offsets／shape／必要layers及完整檔案SHA，不下載權重／pt／vectors。

原queue Hydra序列化失敗及原source／SHA保留；續接只恢復metadata、重用原batch，沒有重新取樣。保留固定upstream以外的容器Torch／bnb版本及兩個eval輸出float cast偏離。新版原生更新／scaling與完整checkpoint reload評估短程門檻完成，原autograd限制明確保留。報告與原始證據發布後暫停本次追蹤；後續正式新實驗需另依研究範圍授權。
