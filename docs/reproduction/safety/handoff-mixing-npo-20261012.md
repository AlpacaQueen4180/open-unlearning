# Handoff：收尾最後一個 SPF 候選，從 Mixing checkpoint 執行 NPO

交接日期：2026-10-12（Asia/Taipei）。使用者已明確指定：檢查、評估並發布最後一個 SPF 結果；完成目前 rank40 評估與 GitHub 發布後，**結束 SPF 參數搜尋，轉以既有 Mixing checkpoint 的 NPO 為主要實驗**。本文件取代舊 handoff 的「SPF 優先起點」與舊搜尋 prompt 的「H 未低於 Mixing 就繼續產生新 SPF 候選」。歷史失敗、原始紀錄及原 Judge 帳務保留。

## 1. 開始前先讀這些文件

1. 本文件與最新 `work/spf-npo-gb200-20261006/private/spf-safety-search-20261010/continuation-*-published.private.json`。以最新實證及使用者 steering 為準。
2. [SPF 收尾 runbook](spf-safety-search-20261010.md)、[finetune 結果報告](finetune-results-report-20261012.md)。
3. [研究規劃 v2](safety_preserving_finetuning_unlearning_plan_v2.md)、[結果追蹤](實驗規劃與結果追蹤.md)。舊「Mixing gate 通過才可啟動」的操作限制由本次使用者指定起點覆蓋；**科學上的 gate 判定仍保留 insufficient_evidence**。
4. [新版大型模型驗證](../npo/large-model-native-validation-20261006.md)、[loss scaling](../npo/loss-scaling-20261006.md)、[update-count 分類](../npo/update-count-mismatch-20261006.md)。

最新收尾唯一RunAI快照：UTC **2026-10-12T13:47:31.441657+00:00**（台北21:47）。rank40 queue **20435** 在該快照為RUNNING／candidate-development/ifbench；full625／5epochs20k、freshreload error0、新完整audit與all625 rank40／anchoraddition已完成。六development／own792 Judge／newharmfulness尚待完整，不重訓或重跑成功stage。下一routine observation不得早於UTC **2026-10-12T14:47:10.162089+00:00**。沒有第二progress查詢或Mixing NPO提交。

原生排程 `spf-full-npo-seed0` 起初 fresh read 是 **PAUSED**，原TOML已保存。使用者其後明確回答「恢復收尾排程，完成後自動暫停（建議）」；已經原生工具恢復，更新後 fresh TOML核 **ACTIVE／每小時／failed_runs_only**。短prompt只允許目前rank40收尾與GitHub發布，禁止新增SPF候選，完成後PAUSE。Mixing NPO由下一session執行；不能把收尾排程擴成新的Mixing實驗排程。

## 2. 最後一個 SPF 候選如何收尾

候選是 **SPF＋安全 loss hybrid**，不是純 SPF：λ=.1、rank40、Adam β₁=.5／β₂=.999、LR5e-6、WD=.01，construction seed0，預定5epochs／625updates／20k exposures／4000unique。

```text
GB200 root       /data/spf-hybrid01-rank40-full-20261012-r1
Current journal  <root>/runs/cpu-fixture-recovery-1539/status.json
Current queue    PID20435
Unique launch    <root>/validation-code/hybrid01-rank40-cpu-recovery-launch-1539.json
Early log        同名 .raw.log
```

原 `root/status.json` 屬 **19460 CPU fixture 失敗**，維持 immutable FAILED；current state 必須讀新 journal。不可重送20435／19460，或重跑1645／r2／r3部署 helpers。

本機唯一 monitor：`work/spf-npo-gb200-20261006/snapshot_spf_rank40_cpu_recovery_20261012_1539.py`，SHA `e2f09d699134220b9fca432c73b2280a60e84c510bae9cca56ab16d17f5daff1`。先原 budget `claim-check`，再同 RunAI／workspace／正常 require_escalated 查一次；`invoke_readonly_json.ps1` 的 MaxAttempts2 僅內建唯讀 transport retry，外層不得 retry／sleep／poll／補查 queued heartbeats。提早 event 只用 event timestamp 拒絕，不能等到門檻後改判 eligible。

已完成、不得重跑／重取的證據：

| 證據 | SHA256／狀態 |
|---|---|
| 新 CPU scalar oracle actual proof | `5397aabbfcaa4d70b1b3f51004721d6d6f389212301ced815472c890bbfb6c4d`，兩次 AdamW／token mean／projection／reload PASS，只重用 |
| rank40 8B engineering proof | `a68b9782be482c1b6c160a1c88beafb747f8805492a6d6e70cf3b9d7de64844c`，40examples／2updates／32+8、fresh reload error0 |
| 20新工程檔 archive | `c65793ec7a88a2e537d415c3d37d53b0ed6d286759281f40ebb02e7496759862`，private `rank40-engineering-completed-20261012-1846` |
| 原11檔 CPU failure archive | `3f5c58114b262673e8f54d93b14eec2adcff57d75186908eb0ef26fc9a0d346d`，原status／source／raw保留 |

原 queue 自動接 full train → fresh reload → 新完整 byte audit → all625 rank40／anchor-addition acceptance → TOFU／knowledge／IFBench／safety／conversation／IFBench CPU score。不要再提交平行 GPU 任務。

完成後只取**新增** binding／resolved／trajectory／audit／acceptance／commands／raw／development outputs，逐 bytes SHA immutable 保存。完整 checkpoint 只有 reload／eval 失敗，就以 explicit new tag 接未完成 stage；不重訓完整模型、不重跑 successful reload／eval、不重hash已完成16GB。

六development是固定 TOFU4000 teacher512／gen200 batch8、MMLUaux1024 zero-shot ABCD、IFBench300 cap2048、safety650 cap512、WildChat71兩輪142turns cap1024、隔離 CPU IFBench scorer。完成 actual model audit 與全部 outputs 驗收後，製候選自身 **792 Judge packet**，不移植前模型或M0 labels。

完成此候選的 Judge、audit、報告與 non-force GitHub publication 後，**不再選新 SPF recipe**。更新本文件的最終結果欄並用原生工具 PAUSE 本收尾排程；原50次搜尋的停止條件不再要求繼續追到H<13。這是使用者改變研究路線，不能宣称整個 NPO 研究完成。

## 3. Mixing 起點：本輪實際查核

```text
H100 host     ai@140.113.26.113:7525
Known hosts   repo父目錄/.npo-h100-known-hosts
H100 Python   /home/ai/miniforge3/envs/open-unlearning-repro/bin/python
Checkpoint    /home/ai/alpaca/work/spf-p0p1-20260916/artifacts/mixing-full-v1/target
```

本輪沿原 SSH／knownhosts 只讀六個小 metadata、stat四個shards；**沒有读tensor bytes、重新hash16GB、訓練／評估或讀H100 safety.env／API key**。原large frozen 7.4MB超小metadata cap的失敗 source／stdout／stderr保留；明確r2排除該檔後取得六檔。Private證據：`work/spf-npo-gb200-20261006/private/mixing-npo-handoff-metadata-r2-20261012/summary.private.json`。

已核 Mixing：method=mixing，construction seed0、full4000，Trainer global_step625／epoch5，20,000 task examples／4000unique，effective batch32，LR1e-5、β₁=.9／β₂=.999、WD=.01；原singleton安全anchor、`L_task+1.0×L_safety`。原H100 recipe為micro4×acc4×world2；metadata沒有world_size欄，雙rank與profile另由原紀錄支持，不能把null當0。Fresh reload已記 pass／maxlogiterror0。

| 來源 metadata | SHA256 |
|---|---|
| target.json | `b8b313567de70ce5f52076403379ebb77467b43c48106212e255f0550d699a6d` |
| resolved_training.json | `bab7ea59bf05b5c2d8214d1a1e9c11cefe9b4820fab75ea3652b1207f45fb407` |
| trainer_state.json | `f8120e377923ba219fdf00bb98574eb409716792f72410a9f78f022ab7f417bb` |
| config.json | `4fe544dc6144e75ee177447b2d1e57a4d8c560f1ef51b6c079104eefee0ffd95` |
| generation_config.json | `f62859b817ec0ad72540dfb2e2ae64fb70b3a8ffd7fad503193fa76e64b80227` |
| tokenizer_config.json | `cf9caa6e3956c24f5787bff455fccb25532afa5091d6b8cbee01a5b7235a1362` |

四個 safetensors metadata sizes 分別為 `4,976,698,672`、`4,999,802,720`、`4,915,916,176`、`1,168,138,808` bytes，名稱 `model-00001-of-00004.safetensors` 至 `model-00004-of-00004.safetensors`。**這是stat，不是新權重byteaudit**。下一 session 必須找既有可信完整 shard/index manifest；若已驗則重用digest。若缺失，首次受控轉移前才補必要完整源／目的 bytes audit，包含index、291tensors／32layers／shapes／offsets／BF16；不能只憑名稱與總大小認定相同模型。

原 frozen lineage：`b2f48edc1fc9cd95b5fb7c9b6855750796809c21fa02e5e7f8426ac118701d89`。Meta M0／tokenizer為 `meta-llama/Llama-3.1-8B-Instruct`，revision `0e9e39f249a16976918f6564b8830bc894c89659`。**NPO起點必須是上述Mixing產物，不能換成Meta M0、公開TOFU full、原SPF或hybrid。**

Mixing已完成development：H13/300=4.33%（M0為7/300=2.33%），MMLUaux81.35%、IFBench strictprompt20%、WildChat overall約2.908。原 `mixing-complete-statistics.json` SHA `a7088587b544edc31afba6d25bb69e263f237d2d3d2d277bc4afd1c26497f1e3`，gate-v2 SHA `ea74957628f5035b967081114853071cd5ee1cf6dd8887527c4a6ecaeb9d0976`。本輪只重用既有aggregate，未重審792 raw。Mixing gate仍 **insufficient_evidence**，主要有utility區間跨門檻、benign適用性／人工裁決與cluster驗證缺口。

## 4. 執行環境與轉移

本機repo：`C:/Users/Alpaca/Documents/ChatGPT/Machine Unlearning and Safety Alignment/open-unlearning`。

GB200沿原 `smart-mfg` project／`machine-unlearning-pvc` workspace／`/data`。Python `/data/npo-gb200-20261004/validation-20261006/venv/bin/python`；upstream `17cbbc87192e6934deb92875c359c91bbd837fb4`，TF5.5.4／Accelerate1.13.0／DS0.15.4／containerTorch2.7.0a0+ecf3bae40a.nv25.02／bnb0.50.2／FA2.7.3。

Mixing任務建獨立tag；建議 `/data/mixing-npo-20261012-r1`，**本輪未建立、未傳權重或launch**。先一次fresh snapshot核20435退出／GPUidle／容量／既存run；不得在rank40占GPU時搶佔。共享FS的26TB數字不是PVC獨占quota，沿原600GiB planning budget另估source／reference／UL checkpoint空間；不自動刪舊weights。

H100 metadata查核是Mixing來源查核，不是RunAI失敗時的替代route。轉移使用已授權受控通道与private manifest，目的逐size/SHA核；Linux archive names用`as_posix()`，禁止Windows反斜線literal filenames。不要重啟已退出的46523 receiver／49932 forward或其他舊helper；新tag設exclusive receipt。CLI失敗或response不確定先保存source／command／原capture，再必要artifactcheck，僅verifiedmissing stage續行，不blind retry／relaunch，不改VPN／DNS／MTU／TLS。

## 5. 固定資料與模型 reference

TOFU revision `324592d84ae4f482ac7249b9285c2ecdb53e3a68`；full4000 QA SHA `cf6f9c9bd844b60661f1c923e6b188628f00b92c1834efb8f9f7c552274a9a33`，retain3800 SHA `e1bfcea25c3237064c11676e9d8f52145032ccb45a0e7f858e7f8446475e8a3e`，training-only anchor SHA `b15ba24ceae3e227e09c5fa44ed7ea2753b53236aaa8d4080ee033944659c1fa`。

既有private `spf-npo-inputs-20261007-r3` 的 forget05 200 QA SHA `bb54599aa3668435a34f238ebbbbd812b26b38d9520151b11fcc28fb99327ba7`，smoke40 SHA `78ec712fc1c1708539d0ba883111293907a562a7eba2a801c86704f6747593de`。資料來源是同一full／retain差集，可在核Mixing原manifest／QA membership與固定順序後重用bytes；不可重新選subset／按安全分數改資料。另建Mixing身份綁定contract，不直接啟動寫死SPF identity的consumer。

區分兩個 reference：

- **NPO frozen reference model**：每個run初始 `M_Mix` 的完整拷貝；initial target/reference全部291parameter fingerprint相同，reference前後不變、無reference gradients。不能使用retain95 retrain或Meta M0代替。
- **FQ evaluation retrain reference**：從同Meta M0、相同Mixing recipe（λ1／β₁.9／LR1e-5／原anchor）在retain3800重新訓練。預期5epochs／595updates／19k exposure、每epoch尾24、warmup119；以actual trace驗。**此Mixing retain95 reference目前pending**，既有SPF retain95不能代替。未完成匹配reference時先報NLL／ROUGE／extraction／suppression diagnostics，不能稱retrain-equivalent或以普通公開retain95補正式FQ。

## 6. Mixing → NPO 的有限實驗順序

1. 核source／model／tokenizer／data／source-mask／precision／installed API，建立newplan、run registry與absence／PIDexit／GPUidle guards。
2. **Mixing專屬新smoke**：固定原forget前40，seed0、micro4／acc8／world1／global32，一epoch、LR1e-7、pureNPO α0、γ1、β.1、retain NLL／1:1、paged_adamw_32bit、WD0、warmup0／linear、clip1、BF16／FA2。預期2updates／10micro／windows8+2、40forget+40retain exposure。這是新M_Mix載入驗收；不能直接重用舊SPF／公開TOFU full smoke作Mixing接受證據。
3. **Pilot**：forget05 200／retain95 3800，先seed0；LR候選1e-7／3e-7／1e-6、各最多一epoch。PureNPO α0與NPO+retain α1分開登錄、都從原M_Mix開始，不能拿α0輸出續trainα1。先最低LR，再依forgetting／retain／development的預先規則縮小候選；不因H未下降盲目提高LR，不用FormalHExPHI選參。
4. full200一epoch預期 **7actual updates／50micro／epoch1**：六個完整8micro，尾組2micro；正式backward完整÷8／尾÷2，`scale_wrt_gas=False`，Trainer／DS／scheduler一致。由新版native Trainer邊界處理，禁止疊舊ceil／forced-boundary patch、手動額外除数或construction token-weighting。保存所有actualloss/scalars/counters/exposure。
5. Pilot選定profile後才凍結主比較：Mixing無更新baseline、pureNPO α0、NPO+retain α1；建議UL seeds0/1/2，construction seed固定0。必要seed0已完整且同freeze config可重用，不能為「正式」再重跑。是否追加其他forget splits／epochs／seeds須有明確新範圍，不能沿September五seed／10epochs舊表自動擴張。

NPO階段不繼續Mixing的安全anchor loss，也不套SPF projection；α1僅是retain NLL。明確把起點安全程度與UL過程防護分開，後續mitigation另命名。

**注意repo預設 `configs/trainer/NPO.yaml` 的α=1。** PureNPO必須explicitα0；NPO+retain explicitα1。原loss公式 `γ×forget_loss + α×retain_loss` 保持，不藉修相容性修改算術。

## 7. 已知工程陷阱與驗收

- Meta tokenizer固定revision／原chat template SHA `7d3e75bbebc8ca06f05579ff90e1730db88e7982d6c4ff524c55c736b21a10a0`，assistant-only labels／max512／EOS=PAD有效token masking／QA padding與index guards保持。超cap或零有效labelfail，不能丟題湊counts。
- TF5.5.4移除了TrainingArguments的`save_safetensors` keyword。已source-bound API wrapper只省略原True欄位，其他kwargs同原config、回傳原class；不能直接用舊entrypoint。參考 `spf_npo_training_args_api.py` SHA `430feb8b97a3fb0dea5a9f293b50394300e0a947d5a0e2c0875f5c716e2fae52`、GPU import-aware r2 SHA `bba745ba98304b4edb90037c1d24f776b277c00e42df5ac90e5f07ed3d7a90a7`。Mixing需要newtag identity／source binding，不能冒稱舊SPF source執行。
- HF_HOME／HF_HUB_CACHE均明確 `/data/smart-mfg/jimmy-lin/hf_cache/hub`、offline=1、disable_implicit_token=1，避免舊loadercache_dir錯層。TE/bnb import會初始化CUDA；CPU診斷與GPU import-aware模式分开，不把原CPU-only CUDAfalse guard強套GPUtrain。
- 完整HF export／index／291tensors／32layers／BF16／shape／offset／byteSHA與fresh-process logits比較都要驗。原atol.05／rtol.01保持；failedcomparison保存private。
- 原SPF NPOplainHF freshreload曾因BF16 nonpersistent rotary buffers與nativeDS context不同而失敗。新的Mixing freshreload要依source-bound nativeoptimizer-free BF16／ZeRO3 context核capture，不能手動cast rotary／改TF32／降低threshold湊pass；nativecontextpass與plainHFpass分別標示。
- 原四個大型autograd5%等價驗收失敗保持。已驗actual除8／尾除2、sameDS explicitmean及新smoke不能解除此限制，也不能追溯改標舊checkpoint。

## 8. 評估、Judge、額度與發布

每個UL checkpoint：TOFU forget／retain NLL、QA likelihood／ROUGE-L／EM／extraction、truth-ratio；matchingMixingreference完成後再FQ／KS D／p／MU／TR。效用與安全沿固定六development，另正式HExPHI／MMLU5shot／MTBench留獨立確認，不用development欄位冒充。

分开 `ΔH_FT = H(M_Mix)-H(MetaM0)` 与 `ΔH_UL = H(M_UL)-H(M_Mix)`；不能以原SPF或公开TOFU full作Mixing UL基線，也不能用下降H本身宣稱遺忘有效。同步報retain效用、benign過拒答、conversation質量與不確定性。

Judge既有固定profile：gpt-5.6-terra／medium／4096／zero retry／三原rubric／https://api.openai.com/v1。每個模型own792（harmful300／benign350／conversation142），API前durable intent，raw/result fsync後receipt，unparsed／unexpectedmodel／API/filesystemerror立即stop。sent／uncertain不auto retry；成功labels不重跑。新run先驗source/profile/packet/actualoutputs再reserve，才由sameuser DPAPI CurrentUser loader在child memory解密既有專用key；不重新問貼key、不讀H100 safety.env、不存plaintext或送GB200。

本輪帳務仍 **5/50 COMPLETE、3960intents=receipts、45可保留**，rank40 own792尚未reserve，第6套完成後通常44remaining，以實際ledger為準。**SPF收尾後不等於剩餘配額可歸零重開。** Mixing NPO模型的Judge若沿既有總授權執行，要登錄清楚新的task scope並共享總額帳務；不得建立第二個50模型預算而重複計算授權。此文件不啟動付費call或新NPO排程。

GitHub既有明確授權：`AlpacaQueen4180/open-unlearning`，目的branch `repro/npo-h100-ada-5seed`，本機`codex/npo-gb200-deepspeed`，non-force push並GETref核。Public只sources／aggregate／audit/proof／asrun／docs／非敏感SHA；全部rawCLI／QA／anchor／prompts／responses／judge reasons／labels／membership／weights／SQLite private。舊handoff及六GB200 untracked不混stage。失敗原bytes保留；Gitblob逐bytes或Markdown原newline normalization核／credentialpatterns0／strictdiffcheck。

## 9. 下一個 session 可直接貼上的起始訊息

> 請先讀 docs/reproduction/safety/handoff-mixing-npo-20261012.md 與最新private continuation。使用者已確認目前rank40 SPF做最後一次評估／Judge／GitHub發布後停止SPF搜尋，不再開新候選；若尚未收尾，接unique20435的新journal，每小時只一次必要query，使用者已授權並恢復ACTIVE每小時收尾排程，完成目前候選Judge／audit／發布後用原生工具PAUSE，不再開SPF候選。接著從H100既有mixing-full-v1/target做Mixing→NPO主實驗，先核source/destination模型身份與原QA、Mixing專屬smoke、seed0 pureNPO／NPO+retain有限pilot及matchingMixingretain95 reference，再凍結UL0/1/2比較。reference model必須初始Mixing；不套SPF或額外安全anchor loss。保持原loss、512mask、完整÷8尾÷2、actual7updates50micro、291fingerprint/referenceunchanged與原freshreload門檻。原gateinsuff／autograd／單seed跨平台限制如實保留，Judge先記帳／savedDPAPI／zero retry，全部raw/key/weights private，完成source/aggregate/report nonforcepush原branch。

## 10. SPF 最終結果欄：收尾後補入

| 欄位 | 本輪狀態 |
|---|---|
| rank40 full625／reload／audit／all625 acceptance | 已完成：625／epoch5／20k4000unique，reloaderror0、291BF16／32layers byteaudit及all625 rank40+anchor PASS |
| 固定六development | 21:47 snapshot正在IFBench；完整六stage仍待完成 |
| own792 Judge／H300／benign／conversation／usage | 待完成；不得填入rank20的結果 |
| 最終模型identity／weights digest／archive／audit SHA | identity`64d3fd783130bc4936a97cbc31da044dfd5e20487b35f2e71a21f6e20ad30c78`；weights`2e993e10c27c3f7758f88d052b9fe2680df5f70dfa85ce60c719bb6eca8caa1f`；19新完成archive`b1412f757709d9ffe53e37d790966b2be61a33d4d1d4167f50bad55bfc424ada`；audit`071102b49e0112660df85907e892700cf5a7c479c36ce45cc6d2abd1627b839a`；全部actual來源，Judge待完整 |
| 最終GitHub commit／GETref | 本輪會先發布handoff與已核現況；最終評估完成再補 |
| 收尾排程 | 人類明確授權恢復；native fresh核ACTIVE／hourly／failed_runs_only，只收尾rank40，最終發布後PAUSE |
| Mixing NPO開始 | 本輪僅交接；未transfer／launch／建立matching reference |

下一個session完成收尾後要更新上述欄位、finetune結果報告與最新private continuation；不能讓本文件的pending永久充當最終結果。
