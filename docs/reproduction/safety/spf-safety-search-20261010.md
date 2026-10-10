# SPF safety search：每小時續行，新增最多50套模型Judge

2026-10-10 17:04 heartbeat最新：β1=.5／LR1e-5候選的792 Judge全部成功，唯一local觀測UTC09:05:36.416221。實際新增792 raw/schema/payload/order/rubric/model/tokenusage、2376events與792ledger intents/receipts逐核通過，duplicates0／zero retry，Campaign.finish記為COMPLETE；audit SHA4bfc0861f35106f3ff1b07a636153daf116935cd627e85ca7095fbc6441ff5c2。新增額度1/50 complete、49尚可保留，沒有重送舊成功。

Harmfulness93/300=31.00%，safe_non_refusal91、refusal116；原SPF141/300=47.00%，下降16個百分點，但仍高於Mixing13/300=4.33%，停止條件尚未達到。Benign substantive345/non_answer2/refusal3（350）；conversation overall403/142=2.838，relevance545/142=3.838，factualaccuracy349/122=2.861，consistency471/111=4.243，nullable分母保留。Actualmodel792筆均gpt-5.6-terra，usage input418388/output113488。這是seed0固定development描述性比較，非pairedcluster顯著性或合格gate；完整labels/source-map/reasons皆private。

下一個單因素候選保留β1=.5，僅把LR1e-5降為5e-6，從原MetaM0獨立full625/5epochs20k，其他資料／training-only anchor／原loop／SPF投影rank20／micro4acc8global32／warmup125／precision／六development stages皆保持。假說是較小更新幅度可能減少安全能力侵蝕，同報TOFU、MMLUaux、IFBench、benign拒答與conversation代價，不因development結果回灌題目。3new production sources及deployment producer來源已準備；新recipe/decision/plan12 negatives與predecessor/newrun boundary8 negatives只CPU通過，executor inversebytes一致、7原queue functions AST同。這不是actual新train/CUDA验收，新candidate2尚未reserve Judge。

本輪唯一RunAI完整logical snapshot的內建兩次transport嘗試均EOF（WebSocket stream／cluster-status），failed captures279/86bytes私有保存；未取得完整snapshot，因此沒有deploy、remote mutation或新GPU submission。不是新auth或training failure；不外層retry／第二progress／改route／盲目relaunch。下一小時由已prepared `work/spf-npo-gb200-20261006/snapshot_spf_search_lr5e6_20261010.py`只查前一候選完成退出/GPUidle及newROOT不存在；fresh≤10min後才有限傳3newsource＋newplan＋bound snapshot、單次queue，原auditor/352assets/16GB完成weights不重傳重hash。

Native ACTIVE/hourly/failed_runs_only與短prompt未改；最後claim UTC09:05:36.401221，下一routine不得早於UTC10:05:36.401221（Taipei18:05:36），提早觸發直接skip。H≤12/300或50配額用盡即PAUSE搜尋；人類adjudication、pairedclusters、matching-reference metrics、原NPO reload/pilot與autograd限制仍保留。詳見[完整Judge與下一候選證據](../../../results/reproduction/safety/spf-npo-gb200-20261006/spf-beta05-judge-complete-and-lr5e6-ready-20261010-1704.json)。以下pending與舊進度是歷史。


2026-10-10 15:51最新：人類已明確「確認可以送出」，涵蓋前述β1=.5候選650 safety＋142 conversation prompts、candidate own responses與history送至OpenAI https://api.openai.com/v1。確認SHA8f02288e90476522be30ed625bb1830b1f063b234744b5025190f3f2a040a647；原兩次automatic review拒絕保留為歷史。同一prepared source經正常require_escalated審查接受，UTC07:49:58.4133434單次提交loader PID49552，額度先reserve為1/50，再於原程序內解密本機DPAPI savedkey；沒有新key視窗。

唯一啟動觀測UTC07:51:19.758138：runner35156／RUNNING，attempted27、success26/792、1項in-flight intent，actual model26筆均gpt-5.6-terra；medium4096、zero retry與originalrubrics/profile不變。這是啟動時的進度，並非完整792 Judge結果；candidate harmfulness仍未知，不能稱安全改善或合格gate。先前claim帳務16 intents/15 receipts與其後此27/26 job觀測各保存原時點，沒有把in-flight當success。

本轮Judge routine觀測1、RunAIquery0、新GPU0、outerretry0，完成train/reload/audit/development及歷史paid successes皆未重跑。最後claim UTC07:50:49.568547；下次routine觀測不得早於UTC08:50:49.568547（Taipei16:50:49），由下一個符合一小時間隔的排程觸發查一次。提早觸發只skip，不等待、輪詢、reset claim或補查queued heartbeat。不要重新launch本run或開keyUI。

Native ACTIVE/hourly/failed_runs_only、短prompt與schedule未改。Actual792完成後只審核這批新raw/schema/payload/order/model/usage/events及ledger，再Campaign.finish；H≤12/300（嚴格低於Mixing13/300）或50套額度用盡即PAUSE搜尋，否則依結果提出並實作下一個候選。APIfail/unparsed/unexpectedmodel/filesystemerror均stop、不retry sent/uncertain；gate、pairedclusters、人類adjudication、matching metrics、NPO與原autograd限制保留。詳細[新Judge啟動證據](../../../results/reproduction/safety/spf-npo-gb200-20261006/spf-beta05-judge-launch-20261010-1551.json)。以下等待確認與其他進度均為歷史。


2026-10-10 09:57最新：β1=.5候選已完成train/reload/newaudit及六development stages，queue9069已退出、GPUidle。唯一snapshot UTC2026-10-10T01:57:26.424367+00:00，SHA8a8f7e0083af310dc4ea540d6d30458bee5ac9c86f3453c250fc8fa0852dea2e。41完成source/commands/raw/outputs逐SHA immutable保存，archive85371621181b714bb36ad2d1ee04a043797314a8806e5fba8822aa8eaa389350；沒有重跑評估或讀舊權重。

免費描述性結果：TOFU f05/r95 exactmatch98.50%/97.76%、ROUGE-L F1 .993093/.994719；MMLUaux831/1024=81.15%（zero-shot，非formal5shot）；IFBench strictprompt20.67%/instruction22.97%、looseprompt22.00%/instruction24.71%。Harmfulness仍未知。

新的private792 packet已按fixed prompts/order、candidate own responses/context與actual六stage proof驗證；CPU preflight pass、originalsavedDPAPI/SDK/rubrics/profile不變，50row盲審表已準備但未人工判讀。兩次normalrequire_escalated launch均被automatic review在CreateProcess前拒絕，要求拒絕後人類明確確認本批650 safety＋142 conversation prompts/responses送至OpenAI https://api.openai.com/v1、gpt-5.6-terra medium4096零重試。已提出精確確認；目前run/source/plan全可review、沒有launch/任何keyread/API/reservation，額度0/50。勿以其他路徑繞審查或未答即重送。

新actual-result auditor `scripts/reproduction/safety/audit_spf_safety_search_judge.py`已compile，尚未有新paidraw可審；完成後逐核792raw/schema/payload/order/events/model/usage及ledger receipts，才Campaign.finish。Native ACTIVE/hourly/failed_runs_only及短prompt保持；stop規則H≤12/300或50模型額度用盡，gate/人工adjudication/pairedclusters/matchingmetrics/NPO及原autograd限制保留。詳細[完成評估與Judge準備證據](../../../results/reproduction/safety/spf-npo-gb200-20261006/spf-beta05-development-complete-20261010-0957.json)。以下進度是歷史。


2026-10-10 07:56最新必要snapshot：β1=.5 full訓練完成625updates/5epochs/20k/4000unique，reload error0及new candidate完整byteaudit pass；queue9069仍存活，stage candidate-development/tofu，六stage尚未完成、harmfulness未知。Snapshot SHA f516af1dd542b47891f80113203b65b1b8e8c772a0700156d3b7440361af8ccd，UTC2026-10-09T23:56:24.159097+00:00；completed24files immutable archive59c73f01eff386fdaf0d56cd2f9bf5bcac1c1725d74d528091dd9cae97505122；new model identity9f8b68d4f657d2021c63b2aebe8ae76020b1c2d18e779c35451bd4b6d01c79c9。詳細private continuation-20261010-0755.private.json保存，原06:32進度是歷史。

新production runner `scripts/reproduction/safety/run_spf_safety_search_judge.py`與`start_spf_safety_search_judge.ps1`已準備；直接使用SHA-bound原Judge parse loop，只加ledger intent/receipt hooks，兩新增API-after-response/state-after-receipt fail-stop mocks pass、realAPI0/keyread0/reserved0。先完成actual六stage才製新packet與development-proof，CPU preflight通過後loader只一次preflight --reserve在key讀取之前記帳；不要直接啟舊2376cap來源。Prepared completed-only export是`work/spf-npo-gb200-20261006/export_beta05_completed_development_20261010.py`，本輪未執行；下一小時snapshot確認完成才使用，不查本輪進度第二次。


2026-10-10人類授權：訓練完接著評估；harmfulness未低於Mixing時，提出可能改進並實作；新增50次完整模型development Judge；使用每小時排程，禁止輪詢；找到低於Mixing的方法或用完50次即停止。本文件取代舊heartbeat中「新beta05 Judge未授權」及「排程維持PAUSED」的現行限制，舊檔與失敗仍保留。

## 判準、額度與持久狀態

- 固定同一development harmful300題，Mixing為13/300=4.3333%；嚴格低於即最多12/300=4.00%。13/300相等不能停止。缺harmful標籤不得補值或改分母。若harmful300已完整驗證且達標，即停止新增訓練/Judge；benign或conversation的缺口另報，不把部分結果稱完整792。這是使用者指定的描述性停止條件，非統計顯著性或正式安全gate通過。
- 一次額度是一個獨立checkpoint/model identity的完整development評估：免費六stage及固定792 Judge請求（harmful300、benign350、conversation142），不是一個API call。**新增**最多50個模型配額，最大39,600個新request intents；舊157/313/469/625付費結果不計入、也不重跑。每個模型launch前保留792額度；partial/uncertain也占原配額，與complete分開記錄，不能用失敗重置50。
- 唯一private campaign：`work/spf-npo-gb200-20261006/private/spf-safety-search-20261010/`。人類原授權保存在`authorization.private.json`；SQLite `budget.sqlite3`保存model reservations、每筆intent/receipt、結果audit SHA與hourly observation claims。另保存每輪exclusive私有receipt，不覆寫原raw。不要把SQLite、key、packet、QA、response、reason或完整labels放Git。
- 預算工具：`scripts/reproduction/safety/spf_safety_search_budget.py`。`--root <campaign> status`讀取權威狀態；`claim-check`在任何routine RunAI或Judge進度查詢前執行。距上次claimed observation不足3600秒會拒絕，直接結束這輪，不sleep/retry。初始last observation綁06:32 RunAI snapshot，第一個新查詢不得早於07:32 Taipei。
- 使用Campaign的`reserve`在key讀取/API之前綁candidate identity、packet、plan與newrun。plan必須含`judge_profile`、`model_identity_sha256`、`packet_sha256`、`new_campaign_model_limit=50`及actual `development_outputs_and_candidate_audit_verified=true`。不接受synthetic當actual。新runner直接import同SHA-bound Campaign，**每次**API前`intent(slot,index,payload_sha)`，durable raw及result fsync後`receipt(...)`。intent失敗不得送API，receipt失敗立即stop、已intent項目不重試。
- 完成後用source-bound私有auditor逐核新增raw/schema/payload/order/rubric/model/tokenusage，audit須連model/packet/plan與全部ledger receipts；`finish(slot,audit_path)`記錄result及停止條件。`close-incomplete`只在原failure/source/raw/events保存後使用；若確有未送出項，原slot的`resume_unsent`只接受exit及精確boundary proof，explicit新run，不重送sent/uncertain/成功項。50配額已耗尽則停止，不新增替代模型。status `STOP_FOUND_LOWER_H`或`STOP_BUDGET_EXHAUSTED`時，用原生automation工具PAUSE本heartbeat、寫最終報告。不能把搜尋停止誤稱整個NPO研究完成。

## 每輪只有一次必要觀測

1. 先讀本文件、最新private campaign state與人類steering。若已stop，只核並PAUSE本排程；其他舊`gb200-npo`/`spf-checkpoint`仍PAUSED。先`claim-check`，再做一次必要完整logical snapshot與一份必要local Judge receipt，沒有外層retry/poll/睡眠循環。沒有新結果時等下個一小時排程。
2. 原RunAI CLI、project smart-mfg、workspace machine-unlearning-pvc，正常`require_escalated`審查；`invoke_readonly_json.ps1`內建MaxAttempts2只用唯讀transport。不可改VPN/DNS/MTU/TLS/route、換SSH繞審查或補查queued heartbeats。查詢失敗先保存private capture/metadata，不把transport或登入失敗說成模型訓練失敗。
3. Actual completion/failure後，有限immutable evidence export屬必要結果保存，不是再做progress snapshot。逐bytes核source/exactcommand/status/raw/binding/audit/完整outputs；不再下載或評估completed evidence，不重hash舊16GB。不停止其他workload，不覆寫失敗。不重訓已完整checkpoint；若reload/eval失敗，只explicit newtag恢復已證明未完成stage。
4. 要提交下一GPU候選，必须≤10min完整fresh snapshot、old relevant PID已退出、GPUidle與recipe/source/data guards核完。上次snapshot stale就等下一小時，不額外查第二次。每次單一有限train→reload→candidateaudit→六stage queue，拒已有run/target。process wait屬有限stage完成等待，不能套status polling。

## 第一個候選與Judge接續

目前模型為`/data/spf-beta05-full-20261010-r2`；唯一queue9069，torchrun9072/native9140，launch `validation-code/beta05-full-launch-20261009.json`及同名raw.log。最後完整snapshot UTC2026-10-09T22:32:30.116690+00:00，SHA`d824e7bc37b4872e1769a64e9a5af3376e976d7542bfb026d8b32cca873ce180`。當時train RUNNING/completed[]、actual AdamW(.5,.999)/LR1e-5/WD.01其餘TrainingArguments match；469是保存點非actual current update。新harmfulness尚無結果，勿重送9069或重啟舊helper。

Snapshot source `work/spf-npo-gb200-20261006/snapshot_r4.py` SHA`9565282356648bb713c5f9deeb9ccb812c2d78aab040415e98e9c00ec4bf70e3`。Current queue source `run_spf_beta05_full_serialization_r2_queue.py` SHA`9d734587cb49590f275b906405117640e6e1a472f43ac742a9a196dbe787c937`；wrapper SHA`33127eee907d343e18daf29fd064f8551048544ef03f7943dadb8339fd219e32`；development executor SHA`76c5fd9c0a6228643d3c9a11621cfa90db52fed65efde1f403d86a7f23c14191`；plan SHA`e5782f588434ba8b1ecb799882a1f6cef3a1c8347b48b47497b2c8304ebc7ecb`。先依此branch必要query，不rerun完成baseline。train完成並不等於development已完成，queue原本就會接freshreload、完整新candidate byteaudit及六development stages。

Recipe只beta1 .9→.5，beta2 .999/LR1e-5/WD.01/seed0/5epochs625updates20k/原4000及anchor/order/world1 micro4acc8/global32/warmup125/rank20/max512/BF16/FA2；original training loop及SPF projection保持。原report_to guard/setup0steps failure及transport failure全保存。

完成六stage後，以新模型輸出製作新的private792 packet，逐題綁固定dataset/prompt/order、actual full model audit與response SHA。六stage為TOFU4000 teacher512/gen200batch8、MMLUaux1024 zero-shotABCD、IFBench300 cap2048、safety650 cap512、WildChat71兩輪142 cap1024、isolated CPU IFBench scorer。不用舊模型responses或M0 labels替新模型。

第一筆paid call前，明確new source/tag改造已驗`run_spf_checkpoint_judge_append_state.py`與`start_spf_checkpoint_judge_append_state.ps1`，接上述新ledger；不要直接執行舊2376cap/checkpoint-specific helper。保留原三rubric SHA、prompt SHA、`client.responses.parse` AST與schema、SDK openai2.54.0/Pydantic2.13.4/dotenv1.2.3、gpt-5.6-terra medium4096、https://api.openai.com/v1、max_retries0。只驗新增budget/identity hooks及必要mock integration，原完成schemas/765raw/audits不重跑。將runner/launcher/plan/packet/API0 CPUpreflight及newbudget sources全部SHA綁定後reserve一次，單次launch有限sequential792，不逐case查poll。

人類已准許保存key，仍用同Windows帳戶DPAPI CurrentUser的private `local-judge-credentials/openai-judge-key.dpapi`。CPU/profile/source/packet/budget preflight通過才同帳戶runtime解密到child memory；不印值/不讀H100 safety.env/不傳GB200/不貼chat或clipboard，不重開masked UI。key不存在或DPAPI不能解密，保存safe metadata並提出具體人類所需操作。APIerror/unparsed/unexpectedmodel/filesystemerror立即stop，unknown不auto retry；原durable intent保留。PermissionError只記實際operation/type/errno/winerror/frames，不猜原因、不記secret message/locals。

## 未達標時的可執行候選順序

每次先用剛完成result提出1個具體假說、列出改變欄位與預期副作用，寫exclusive candidate plan後實作。first wave優先單因素對照；依新結果調整，禁止盲目跑50組或以development題目作訓練資料。

1. **Optimizer momentum**：先完整評估正在跑的beta1=.5；比較原.9 SPF與Mixing固定300結果。若仍高，不聲稱已驗修復。
2. **降低更新幅度**：以beta1=.5 LR5e-6及3e-6新full候選逐一比較，其他原recipe不變；也可對新beta05已保存且完整audit的early checkpoint做新評估，每個獨立模型占一個50配額。舊.9的157/313/469結果已顯示early stopping alone未大幅改善，不重跑它們。降低LR可能犧牲TOFU學習，必須同報。
3. **SPF加安全loss的hybrid**：從原Meta M0獨立construction，使用既有training-only安全anchor，新增安全loss coefficient .1→.3→1的有限候選，一次一個，保留SPF投影。明確標為hybrid，分別記task/safety/投影貢獻，不冒稱純SPF。訓練資料須原授權、與development完全分離，禁止650 benchmark或FormalHEx-PHI回灌。安全loss定義/mean reduction/optimizer update/serialize reload須用source與actual runtime驗證後才大規模跑。
4. **投影覆蓋與實際更新方向**：評估rank32/64及AdamW經momentum/preconditioning後實際delta的安全方向是否仍受保護；只在新有限工程驗證bitwise/RNG/precision/梯度與實際update geometry後，實作獨立候選。不能把synthetic或CPU pass抹成原大型autograd5%fail已解決。若source/actual診斷不支持此假說，選其他候選。
5. 上述不達標時，用既有獨立training-only anchor探討更廣安全方向或hybrid係數/LR組合，每次以既有結果縮小下一選擇；資料不足就先選可實作家族，不能造label或擴scope去付費新資料。

這些是待驗假說。原SPF final47%、31344%、46944.67～45%的描述性結果支持需要改配方；不能保證會低於Mixing。SPF使用安全梯度投影與Mixing的loss組合不同，原論文也討論utility/safety tradeoff及beta1=.5，但本機recipe、平台與原未過autograd限制不同。[原始研究](https://arxiv.org/html/2601.10141v1)。每個候選同步保存TOFU forget/retain NLL/ROUGE/EM/extraction、MMLUaux、IFBench、benign refusal及conversation quality，避免只報harmfulness而藏utility代價。

## 報告與排程

Native heartbeat `spf-full-npo-seed0`改為ACTIVE每小時，保留failed_runs_only；其他原PAUSED排程不啟動。原長prompt/TOML已保存campaign private history備份，排程prompt只指向此文件。排程每次讀本文件與private權威ledger，不能把旧prompt的未授權/15min/PAUSED內容當current。若人類之後改scope，保存新授權並具體更新runbook/ledger，不擅自重置額度。

必要source/aggregate/result更新`results/reproduction/safety/spf-npo-gb200-20261006`、主report/tracking/provenance；依既有授權exact public scope nonforce push AlpacaQueen4180/open-unlearning `repro/npo-h100-ada-5seed`，Git blob SHA/credential patterns/diffcheck與GitHub GETref核完。原7untracked/private bytes不可混stage。Humanadjudication/pairedclusters、matching-reference metrics、單seed/跨平台與原NPO reload/pilot缺口繼續保留；campaign達停止條件即停止搜尋，後續其他研究要另依原scope，不宣稱總研究complete。
