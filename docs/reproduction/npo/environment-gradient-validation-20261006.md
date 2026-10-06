# 新版環境與 H100／GB200 舊版梯度驗證

開始日期：2026-10-06。使用者授權依 [upstream 建議](upstream-status-20261006.md) 驗證新版及舊版 loss scaling。有限實驗於同日10:34臺灣完成；本文保存最終結果及按時間排列的原始執行紀錄。長任務採15分鐘排程，沒有反覆 poll。

同日17:03追加的主要模型Llama-3.1-8B及Llama-2-7B短程驗證亦完成，另存[大型模型報告](large-model-native-validation-20261006.md)與獨立證據。皆Trainer=DS4、micro20、epoch2，正式訓練實際backward完整組除8／尾端除2；四個same-DS scaling gates通過、四個原autograd gates未過。這是另外兩個smoke runs，不混入下方三split 1B full結果，也不是歷史safety修正。

## 最終結果與驗收範圍

**有限實驗全部完成，舊版漏除accumulation分母已有實測，新版原生更新計畫與平均行為在本次範圍通過。** H100八項、GB200二十項tiny與十六項真實1B synthetic／TOFU梯度量測均完成；新版1B三split seed0亦完成完整訓練、評估、checkpoint與独立數值稽核。[最終機器可讀稽核](../../../results/reproduction/npo/scaling-validation-20261006/final-audit-summary.json)及[證據索引](../../../results/reproduction/npo/scaling-validation-20261006/README.md)保存原始小檔、SHA、版本及失敗。

舊版GB200完整8批／尾端2批相對同DS explicit-mean control為8／2倍，新版為1／1倍；H100真實7B完整4批相對autograd mean為3.978倍。**真實1B對原autograd reference的16項5% gate仍未通過，差異未定位。** 同DS對照隔離scaling，不能取代完整autograd／DS等價驗證，也不能把梯度倍數當成LR或Adam更新倍數。

| 1B seed0 | 新版 FQ | KS D | MU | TR | 實際樣本／reference | Trainer／DS | microbatches | epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| forget01 | 0.2656871403 | 0.225 | 0.5307198782 | 0.6791958660 | 40／40 | 20／20 | 100 | 10 |
| forget05 | 0.2704743833 | 0.100 | 0.4780954414 | 0.6972605670 | 200／200 | 70／70 | 500 | 10 |
| forget10 | 0.0540529127 | 0.095 | 0.5280958392 | 0.6398316494 | 400／400 | 130／130 | 1000 | 10 |

| Split／seed0 | 原舊版 FQ／MU／TR | 只修updates FQ／MU／TR | 新版native FQ／MU／TR | docs/repro.md FQ／MU／TR |
|---|---|---|---|---|
| forget01 | .00676／.58801／.54393 | .76593／.53570／.67856 | .26569／.53072／.67920 | .92／.56／.66 |
| forget05 | .22054／.44002／.70480 | .17793／.47448／.70133 | .27047／.47810／.69726 | .14／.45／.70 |
| forget10 | .04457／.52278／.64250 | .02985／.53848／.63618 | .05405／.52810／.63983 | .02／.46／.70 |

原版更新／epochs不足，六格corrected只改ceil與DS邊界且保留舊loss scaling；新版三格則使用原生TF5.5.4／Accel1.13.0／DS0.15.4，不套本地legacy update patch。共同名義global batch32（micro4×acc8×world1）、FA2、seed0、configured10epochs，尾端短組仍小於32。原docs文字的8×2×4=64卻標32矛盾保留，未擅改reference。舊分數取既有followup稽核，沒有重跑或替換。

本次保留支援GB200的容器Torch2.7.0a0／bnb0.50.2，與upstream預設Torch2.9.1／bnb0.49.2不同；固定upstream commit `17cbbc87192e6934deb92875c359c91bbd837fb4`另加兩個bf16→NumPy輸出cast，沒有改CE／NPO或訓練loss。新版依賴、upstream及評估計算版本一併變動，單seed不能分離各因素，也**沒有重現所有docs數值**。歷史H100／Ada／safety不追溯改標新版或已修loss；有限任務完成後停止此追蹤，autograd差異作為未解限制保存。

## 已完成的第一階段

真實 NPO → Transformers Trainer → Accelerate Accelerator／DeepSpeedEngineWrapper → DeepSpeed ZeRO-3 的 controlled gradient audit 已取得數值證據。固定 tiny Llama（兩層、hidden32）、dropout=0、相同參數與 reference、合成固定 tokens；SGD lr0、不做 clipping。量測 optimizer step 之前的完整參數梯度，比較 arithmetic mean of microbatch losses 與 single large batch loss。

| 環境 | GPUs | 完整窗口 | 已觀測的梯度倍率 | 證據狀態 |
|---|---:|---:|---:|---|
| GB200 Torch2.7.0a0／TF4.51.3／Accel0.34.2／DS0.15.4 | 1 | 8批 × 4 examples | 8.0000000075 | FP32 combined tiny-model preflight通過；cosine約1，除以投影倍率後relative error約6.30e-8；原始JSON未除倍率的殘差relative error約5.04e-7 |
| H100 Torch2.4.1+cu121／TF4.51.3／Accel0.34.2／DS0.15.4 | 2 | 每rank 4批 × 4 examples | 約4 | forget／retain／combined分項FP32與combined bf16通過；rank0小檔已下載 |

兩個 stack 的 observed `model_accepts_loss_kwargs=true`、`num_items_in_batch=None`，DS backward收到 `scale_wrt_gas=false`。對這些完整窗口、等token權重的受控實驗，先前「漏除 accumulation」的疑點已是實測結果，不只是源碼推論。

H100短組一批的combined FP32／bf16也通過，倍率約1。尾端測試在舊版明確強制DS boundary，以隔離scaling；**不是聲稱原版 epoch loop原生會正確flush尾端**。雙卡兩rank使用相同固定資料，DS平均後與相同local reference比較；這是梯度權重控制，不是重播歷史資料順序。

這些倍數指 clipping前梯度，不能直接解讀為Adam參數更新或有效LR的倍數。SGD lr0只是保留參數的量測器，不是更換正式NPO optimizer。既有checkpoint、歷史分數、舊venv、loss和分類未修改。

H100 evidence：[雙rank JSON目錄](../../../results/reproduction/npo/scaling-validation-20261006/h100/)。GB200 preflight原始小檔已於08:49下載至[本機證據目錄](../../../results/reproduction/npo/scaling-validation-20261006/gb200/)，SHA256 `72a6ebe80aeacbb7b75c28a3b6c6aeb3b98a343b10b053a26fbf50705997f6c0`與遠端一致，並獨立稽核world1／GAS8／window8／DS update1及scaling。這個tiny-model倍率不直接標為每個歷史checkpoint已驗證的倍率。

### 2026-10-06 08:18（臺灣）排程取得的真實 7B 證據

雙 H100、原環境、cached Llama-2-7B full checkpoint、bf16、固定合成 tokens 的完整四批測試已以 returncode0 完成。兩 rank 的 JSON、命令與 log 已下載；本輪31個小檔與遠端 SHA256 一致。

| 測試 | 梯度投影倍率 | cosine | 消除投影倍率後的相對誤差 | 完成範圍 |
|---|---:|---:|---:|---|
| 7B combined，完整四批，rank0 | 3.97807927 | 0.99992089 | 0.01257909 | case完成，rank0／1證據均取得 |
| 7B combined，尾端一批，rank0 | 0.98792375 | 0.99954284 | 0.03024813 | 08:33排程確認case完成，rank0／1證據及returncode0均取得 |

誤差欄為 `norm(observed / projected_scale - reference) / norm(reference)`；原始 JSON 的 `relative_error_after_scale` 使用未除倍率的殘差，完整四批為0.05004061，不能直接當作上表的1.26%。bf16的數值偏差已保留，沒有把結果寫成精確4倍。參考梯度是同一批次序的 microbatch-loss arithmetic mean；相對 single large batch 的倍率為3.97076736，兩個reference本身的relative error為0.00858676。

這提供了真實7B架構在舊雙卡呼叫鏈約4倍的證據，仍是合成固定資料的受控量測，不能當作歷史TOFU每一步或Adam更新的重播。尾端強制boundary的限制不變。

08:33排程確認H100有限queue **DONE，八個case皆returncode0**；所有16份rank JSON、8個commands、8個logs、兩份實際遠端script及status，共35個原始小檔均核SHA。獨立[本機稽核](../../../results/reproduction/npo/scaling-validation-20261006/h100/audit-summary.json)確認world2、GAS4、SGD lr0、clip0、每case DS update1、TF4.51.3／Accel0.34.2／DS0.15.4，且同case兩rank的完整梯度hash及comparison數值完全一致。H100此有限驗證結束，不再反覆查其PID。

## 執行歷程（以下狀態為各次排程當時觀察）

### H100舊版

- SSH：`ssh -p 7525 ai@140.113.26.113`。本機以BatchMode和workspace內 `.npo-h100-known-hosts` 接線。
- 環境：`/home/ai/miniforge3/envs/open-unlearning-repro/bin/python`。開始時兩GPU空閒，約80GB／GPU。
- 有限queue：`/home/ai/alpaca/results/reproduction/npo/scaling-validation-20261006/validate_h100_scaling.py`，PID4008857。
- 六個tiny cases及7B完整四批／尾端一批case全部完成；最終status為DONE、八個returncode0。本機雙rank獨立稽核通過。
- 相同model revision `cc5b31c69127d5da881608e640f2b453c446435b`。真實架構梯度測試仍使用固定合成tokens，尚未使用TOFU collator實際樣本。
- 最終status、兩rank reports、commands／logs及原始執行script已完整下載並核SHA。這組驗證不再重啟，不干涉其他workload。

### GB200舊版、新版

- RunAI CLI：`C:\Program Files\Runai\runai.exe`，project `smart-mfg`，workspace `machine-unlearning-pvc`，PVC根 `/data/npo-gb200-20261004`。
- 新的有限queue：[validate_new_environment.py](../../../scripts/reproduction/npo/gb200/validate_new_environment.py)。08:18排程已確認GPU閒置且status／launch不存在；三個helper均完成gzip上傳與遠端SHA核對。其後background啟動指令在RunAI cluster-api/status遇到EOF，**啟動結果不確定，尚未驗證新版**。下次排程必須先查status／launch／PID，不能直接重複提交。原始tiny8批preflight已成功完成。
- 已核SHA：`gradient_audit.py=f625a3943b113edd8559cbc63748f702ae230cca5f9ed9b8a6f57efb4f541db1`；`validate_new_environment.py=5985e1e00c95551b795ee3ee9d9e336afbce344a33ea77b45048eb71095afc9d`；`capture_tofu_gradient_batch.py=8cda2c12bdb6b80144629696083bce46f31e375b142152e0b70dce33e185639c`。直接base64會造成RunAI exec URL過長400，因此採gzip壓縮。
- 08:33排程再次取得單次snapshot：status／launch／queue.log均不存在、驗證PID不存在、GPU閒置。因此上輪未實際啟動。其後啟動遇cluster-api/status EOF；僅重試一次，重試先設status／launch／PID／GPU防重複條件，再遇TLS handshake timeout。**本輪仍未確認queue啟動**；不再重試，下輪先查狀態。GB200 preflight小檔下載亦未成功，新版仍未宣稱通過。
- **08:49續接成功：** 08:48排程的單次RunAI exec先確認status／launch不存在、無queue PID、GPU閒置，於同一防重複命令背景啟動有限queue，取得PID **18685**並保存遠端與本機launch。GB200 preflight亦同時下載驗SHA。啟動後結束本輪遠端工作，由15分鐘排程讀status及必要current log；啟動成功尚不是新版imports、梯度或正式訓練通過。
- 先查 `validation-20261006/status.json`／`launch.json` 和PID防重複；不存在、GPUidle才background啟動，保存launch／queue.log。腳本使用exclusive lock、拒絕已有status；失敗保留證據，用新tag並跳過完成cases。

### 09:03排程：新版 tiny 通過，真實1B參考比較失敗；09:10續接

原queue已FAILED且PID退出、GPU閒置。失敗前完成舊版十個與新版十個tiny controls，及新版imports、freeze、固定upstream checkout與40筆TOFU frozen batch。已下載原始82個小檔逐一驗SHA；[二十項獨立稽核](../../../results/reproduction/npo/scaling-validation-20261006/gb200/attempt0/tiny-audit-summary.json)通過。

| synthetic tiny control | 舊TF4.51.3／Accel0.34.2 | 新TF5.5.4／Accel1.13.0 |
|---|---:|---:|
| GAS8完整8批，FP32 forget／retain／combined | 約8 | 約1 |
| GAS8短組2／4批，FP32 combined | 約2／4 | 約1／1 |
| GAS4完整4批／短組1批，FP32 combined | 約4／1 | 約1／1 |
| GAS8完整8批／短組2批，bf16 combined | 8／2 | 1／1 |
| GAS8完整8批，FP32 retain、不同有效token數 | 約8 | 約1 |

此表相對於相同microbatch-loss arithmetic mean，所有cosine約1。不同有效token數的single-large-token-mean另保留在JSON，沒有當作相同objective。新版真實imports確認Torch `2.7.0a0+ecf3bae40a.nv25.02`、TF5.5.4、Accel1.13.0、DS0.15.4、PEFT0.21.2；[freeze](../../../results/reproduction/npo/scaling-validation-20261006/gb200/attempt0/freeze_new.log)保存完整版本。這是保留GB200容器Torch的隔離環境，不是upstream Torch2.9.1全套預設。

失敗項 `old_real1b_w8_combined_bf16` 使用cached真實1B及合成tokens；相對原autograd mean的投影倍率 **7.88321590**、cosine **0.99314307**、消除投影倍率後殘差 **11.771244%**，超過原bf165%門檻。原[失敗JSON](../../../results/reproduction/npo/scaling-validation-20261006/gb200/attempt0/old_real1b_w8_combined_bf16.json)、完整log及returncode1均保留。原因尚未確認，不能將其直接歸因於bf16，也不能把新tiny結果當作真實1B梯度等價性已通過。

追加量測revision `ds_explicit_mean_reference_v2`：在**同一DeepSpeed engine**、同一NPO loss／batch／參數下，直接 `engine.backward(loss / K, scale_wrt_gas=False)` 作為DS explicit-mean control，量測另一個lr0窗口的clipping前梯度；保留原autograd及large-batch比較，另外保存其原門檻是否通過。參數初始／最終SHA必須一致；observed update1與reference update1分開記錄。對照驗收使用原5%門檻，**沒有放寬門檻或刪掉原失敗**。這個新增control隔離Trainer scaling與autograd／DS的實作差異；即使通過，也不表示原autograd比較已通過。

09:10確認舊PID不存在、GPU閒置、固定upstream commit及新helpers SHA後，以新目錄 `/data/npo-gb200-20261004/validation-20261006-r1` 啟動PID **26227**，[launch紀錄](../../../results/reproduction/npo/scaling-validation-20261006/gb200/attempt1/launch.json)已保存。沿用原二十項完成測試、新venv與frozen TOFU batch，僅續接未完成真實模型／TOFU量測與三split正式實驗；新full-run tag加`_r1`。原helpers另存PVC `validation-20261006/source-v1`。新版helper SHA：gradient `2d13942540ad74d21804bf06c6e40219f675a9b0605655f9fac26ad18de0d3fb`，queue `28debdd2e740caf0c3816a1e785e9eae3406f40d0db4a94e685c4b09d9064603`。啟動後交給排程，尚未取得v2真實模型或正式訓練結果。

09:33單次[snapshot](../../../results/reproduction/npo/scaling-validation-20261006/gb200/attempt1/snapshot-20261006-0933.json)確認r1仍RUNNING，十五個新量測皆returncode0：old synthetic完整／短組兩項、old TOFU六項、new synthetic兩項、new TOFU前五項。當前為最後 `new_real1b_TOFU_w2_combined_bf16`，GPU程序33179約25.4GiB；不重啟或干涉。嘗試下載已完成十五case的數值／commands／logs時遇WebSocket EOF，僅重試一次再遇TLS timeout，本輪尚未取得這些原始數值，故只記stage狀態、不填入梯度倍率或宣稱獨立數值稽核通過。下一次排程先做必要snapshot，再補取已完成小檔；三個正式split仍未在此snapshot開始。

### 09:48排程：真實1B梯度證據完成，forget01訓練完成／評估需相容性修復

本輪r1 snapshot為FAILED `native_1b_forget01`，原PID已退出且GPU閒置。此前十六個真實1B v2 gradient cases全部returncode0。本輪下載74個小檔（其中73個原始遠端檔案逐一核SHA，另1個為當次遠端檢查結果），完成[十六項獨立梯度稽核](../../../results/reproduction/npo/scaling-validation-20261006/gb200/attempt1/real1b-gradient-audit-summary.json)。old／new的初始model parameter SHA、輸入hash、frozen batch與offset逐項一致；每case初始／最終參數SHA一致，observed update1／reference update1／total2、clip0／SGD lr0亦核對。

| 真實1B bf16、相對同DS explicit-mean control | 舊版完整8批 | 舊版尾端2批 | 新版完整8批 | 新版尾端2批 |
|---|---:|---:|---:|---:|
| 固定合成tokens，combined | 8 | 2 | 1 | 1 |
| 真實TOFU frozen batch，forget | 8 | 2 | 1 | 1 |
| 真實TOFU frozen batch，retain | 8 | 2 | 1 | 1 |
| 真實TOFU frozen batch，combined | 8 | 2 | 1 | 1 |

表中16個comparison的cosine皆1、消除投影倍率後residual皆0。這直接隔離出舊Trainer路徑漏除窗口批數與新版原生平均行為；**相對原autograd reference的16個5% gate仍全部未通過**，不能稱為autograd／DS bf16梯度完全等價。其差异、原失敗及single-large／mean-micro數值均在JSON保留，原因尚未完整定位。

TOFU frozen40 metadata與遠端pt的SHA皆 `637b7f9d1e2ce9fcc796a225ab015e5139cfe13b61efb60a34f63abc39888718`；40筆forget01／retain99、seed0 retain採樣、mask及有效token數已保存，pt只留PVC。synthetic與TOFU兩類不混用。

新版native forget01正式訓練以FA2／micro4／acc8／global32完成，Trainer **20**、DeepSpeed **20**、microsteps **100**、epoch **10**。[本機獨立計數稽核](../../../results/reproduction/npo/scaling-validation-20261006/gb200/native-forget01/counts-audit.json)通過；remote checkpoint含2,471,645,608-byte `model.safetensors`與config／tokenizer／trainer_state。原評估成功載入全部模型權重後，於 `evaluate_probability` 的 `avg_losses.cpu().numpy()` 發生 `TypeError: Got unsupported ScalarType BFloat16`，尚未取得FQ／MU／TR，不以訓練完成代替reproduction完成。

只在隔離upstream evaluator把 `avg_losses`／`normalized_probs` 的NumPy轉換前加 `.float()`，不改loss／CE／token權重或任何訓練程式。原source與unified diff保存在BASE；這與本機舊repo既有的兩個輸出相容性cast一致，是upstream固定commit的明確偏離。修後eval source SHA `acd98144cada6f02dea329bb78c77e317bd651d16fabbb576796253ae6c4ec7b`。

09:54以新task `/data/npo-gb200-20261004/validation-20261006-r2` 啟動PID **34711**，[launch](../../../results/reproduction/npo/scaling-validation-20261006/gb200/attempt2/launch.json)保存，執行器為[resume_native_validation.py](../../../scripts/reproduction/npo/gb200/resume_native_validation.py)，SHA `48bfb74f380a8a52efcbd298cbaf51f1b6b6a83029c953a6035fd1be7f4aabbc`。先核原forget01 checkpoint header／必要layers及SHA，從原checkpoint恢復至新 `eval-recovered-r2` 目錄，保留原FAILED_EVALUATING及log；不重訓forget01。之後僅啟動缺少的forget05／10，各用包含split的獨立r2 tag，避免舊runner未把split放進名稱的碰撞；仍使用原生Trainer／DS，不加舊update patch。啟動後交給排程；恢復評估與兩個缺少full runs尚未驗收。

### 10:03排程：native forget01完成，forget05訓練中

原forget01 checkpoint評估恢復與audit皆returncode0，run為DONE；未重訓。下載32個原始小檔逐一核SHA，包含完整TOFU_EVAL／SUMMARY、固定retain99 reference、commands／logs、checkpoint completeness／SHA證據及eval compatibility patch。[獨立數值稽核](../../../results/reproduction/npo/scaling-validation-20261006/gb200/native-forget01/independent-audit.json)由原始truth-ratio樣本重算exact two-sided KS（整數lattice-path計數）、MU harmonic mean及TR，與SUMMARY差異均0；FQ **0.2656871403**、KS D **0.225**、MU **0.5307198782**、TR **0.6791958660**，樣本 **40／40**，Trainer=DS20／micro100／epoch10。

checkpoint完整header含146 tensors、所有必要layers；2,471,645,608-byte權重SHA為 `fe1bb48dcde412d522a2dfbd7d46afd071c7d2aa0f29e9109777766e820cea4a`。本機native小檔另存較短路徑，避免Windows長路徑影響Python讀取；原下載的長路徑檔案及驗SHA清單保留。

| 1B forget01 seed0範圍 | FQ | MU | TR | Trainer／DS updates | epoch |
|---|---:|---:|---:|---:|---:|
| 原GB200舊環境 | 0.0067607323 | 0.5880107159 | 0.5439338688 | 10／6 | 5 |
| 舊環境、僅修更新次數與boundary | 0.7659314523 | 0.5356969465 | 0.6785598325 | 20／20 | 10 |
| 新版native stack、eval serialization cast | 0.2656871403 | 0.5307198782 | 0.6791958660 | 20／20 | 10 |
| docs/repro.md NPO參考 | 0.92 | 0.56 | 0.66 | 未附此次實際runtime | — |

舊兩列取自既有followup獨立稽核，未改分數或分類。新版FQ較docs參考低0.6543128597；MU低0.0292801218、TR高0.0191958660。單seed且環境／upstream／model與eval計算版本一併變更，不能把分數差異全部歸因於loss scaling，也不能宣稱目前已重現docs數值。原status的`corrected=false`只表示沒有使用本地legacy correction probe；新版更新／平均是原生依賴行為，不表示仍沿用舊scaling。

本轮單次snapshot確認r2 PID34711仍RUNNING，已完成forget01 recovery／audit，當前native forget05正在TRAINING（GPU PID35434）；forget10由有限序列後續自動啟動。沒有額外poll或重啟。

### 10:18排程：native forget05完成，forget10訓練中

本輪唯一狀態[snapshot](../../../results/reproduction/npo/scaling-validation-20261006/gb200/attempt2/snapshot-20261006-1018.json)確認r2 PID34711仍RUNNING，forget05 stage returncode0、run DONE；當前native forget10正在TRAINING（GPU PID36646）。序列自動續接，沒有重複啟動或追加poll。

forget05已下載23個原始小檔並逐一核SHA，包含完整TOFU_EVAL／SUMMARY、固定retain95 reference、commands／logs、runtime及checkpoint proof。[獨立稽核](../../../results/reproduction/npo/scaling-validation-20261006/gb200/native-forget05/independent-audit.json)重算FQ **0.2704743833**、KS D **0.1**、MU **0.4780954414**、TR **0.6972605670**；SUMMARY最大數值差為5.55e-17，實際樣本 **200／200**，Trainer=DS **70**、microsteps **500**、epoch **10**。新版engine_micro_steps為70，因wrapper只在boundary呼叫engine.step；實際microbatch消耗以training_step計數500為準。

checkpoint完整header／146 tensors與必要layers核對通過，權重2,471,645,608 bytes，SHA `71334507c2bd339eb3e7af2e173036edee7b7049a695a810014edc5175de1d62`。固定retain95 reference SHA `88a1ae92c3fa1feecee18154dc1c5c524ab55a711c11a89ab64cd0cb5bac916d`；權重留PVC，僅下載核對證據。

| 1B forget05 seed0範圍 | FQ | MU | TR | Trainer／DS updates | epoch |
|---|---:|---:|---:|---:|---:|
| 原GB200舊環境 | 0.2205412176 | 0.4400220458 | 0.7048015304 | 60／54 | 8.64 |
| 舊環境、僅修更新次數與boundary | 0.1779335279 | 0.4744844346 | 0.7013345331 | 70／70 | 10 |
| 新版native stack、eval serialization cast | 0.2704743833 | 0.4780954414 | 0.6972605670 | 70／70 | 10 |
| docs/repro.md NPO參考 | 0.14 | 0.45 | 0.70 | 未附此次實際runtime | — |

舊兩列來源為[既有followup獨立稽核](../../../results/reproduction/npo/gb200/followup-independent-audit-20261005.json)。新版相對docs：FQ高0.1304743833、MU高0.0280954414、TR低0.0027394330；並未重現全部參考數值。只完成一個seed且stack／upstream一起變更，不能把分數差全部歸因於scaling。forget01／05完整證據已取得，下次排程只查未完成forget10。

### 10:34完成：native forget10與有限queue結束

10:33排程的單次snapshot取得最終queue DONE（UTC02:34:11），四個stage全部returncode0；forget10 run DONE。下載37個原始檔並核SHA，其中包含最終queue status／commands／logs；大型tokenizer副本只保留本機，不提交Git，published archive明列省略項。[forget10獨立稽核](../../../results/reproduction/npo/scaling-validation-20261006/gb200/native-forget10/independent-audit.json)由原始400／400 truth-ratio樣本重算FQ **0.05405291273173722**、KS D **0.095**、MU **0.5280958391885743**、TR **0.6398316493849326**，與SUMMARY最大差1.39e-17；Trainer=DS **130**、micro **1000**、epoch **10**。

完整checkpoint146 tensors與必要layers／filelength核對通過；2,471,645,608-byte權重SHA `1bb59a04aa6392a87ffc863c33618db3a220457f6baa048efa1c31578c8ca7ff`。retain90 reference SHA `ed70b06a33be580fa24ed730911aa2fc4213a9516167093fdb4b2786a442cc39`。相對docs，FQ高0.0340529127、MU高0.0680958392、TR低0.0601683506。

補取15份實際執行來源並核SHA，含v1／v2／r2執行器、native run_probe、upstream NPO與evaluation source；沒有重新執行完成測試。離線[archive auditor](../../../scripts/reproduction/npo/gb200/audit_validation_archive.py)重驗manifest、gradient evidence條件與三split原始評估／checkpoint計數。Windows長路徑的原失敗檔另以短路徑、原bytes／SHA發布，mapping明列；原資料保留。所有有限訓練、評估與報告證據完成；16項autograd gate未通過的限制未消失，不以queue DONE改稱等價性全部通過。

原定序列驗收（現已逐項執行，數值限制如上）：

1. 舊版FP32三項loss、完整8批、2／4批短組，GAS4配方控制、bf16、不同retain有效token數。
2. Clone固定upstream `17cbbc87192e6934deb92875c359c91bbd837fb4`；隔離venv，TF5.5.4／Accel1.13.0／DS0.15.4，保留GB200容器Torch與bitsandbytes0.50.2。保存實際freeze、requirements差異與import/kernel相容性；此配置不是完整upstream Torch2.9.1／bnb0.49.2預設環境。
3. 新版相同gradient controls，原生路徑不疊舊ceil／boundary patch；cached Llama-3.2-1B真實架構bf16固定tokens的完整／短組對照。另以 [capture_tofu_gradient_batch.py](../../../scripts/reproduction/npo/gb200/capture_tofu_gradient_batch.py) 由舊repo的實際data／collator鎖定40筆forget01／retain99、資料revision、mask與seed0 retain採樣。保存frozen batch SHA／有效token數；old／new分別用前32筆完整組與最後8筆短組，逐項測forget／retain／combined。第三個helper亦須完整上傳後才啟動queue。
4. 新版Llama-3.2-1B forget01／05／10、seed0正式計數對照，原生ZeRO3／FA2／micro4／acc8／global32／10epochs；驗Trainer=DS20／70／130、micro100／500／1000、actualepoch10，完整checkpoint與TOFU FQ／KS D／MU／TR，樣本40／200／400。與原版及只修更新次數的corrected seed0分開比較，不展開其他模型矩陣。

## 必須保持的證據界線

- Mean of microbatch means與single large retain token mean分開報告。有效token長度不同時不要求兩個objective等价；先確認實際預定權重。
- Fixed synthetic tokens驗證呼叫鏈／分母；真實TOFU資料、mask、collator的梯度與正式trajectory是後續不同驗收。尚未完成的項目不能用計畫代替結果。
- 梯度vectors留遠端；完整模型的norm/cos/error以chunk方式計算，避免全量float64副本；下载小檔而非巨大weights／vectors。
- 失敗先保存log、manifest與版本；完成checkpoint只恢復評估，不重訓DONE runs，不更動舊環境或自行改原NPO loss。
- 全部有限驗證與報告完成後暫停 `gb200-npo`；若新版相容性阻礙需修執行器或依賴，明確記錄偏離upstream的範圍。
