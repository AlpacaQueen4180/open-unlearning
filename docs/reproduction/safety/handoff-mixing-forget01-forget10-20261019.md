# Handoff：Mixing → forget01／forget10，套用已驗速度改良

日期：2026-10-19（Asia/Taipei）。使用者要求先將現有報告整理到GitHub，並交接下一session執行Mixing→forget01與forget10，套用本輪速度發現。本次是發布與交接，**沒有啟動新訓練、評估或Judge**。

## 先讀與已完成狀態

1. [forget05最終表格](mixing-npo-forget05-final-20261019.md)。M0歷史／Mixing／6個NPO／matching-retain95均已整理；原3680+人工一次補送22=3702次，3666有效／14最終未知。原scope已結束，不第三次補送。
2. [GB200效能診斷](gb200-performance-diagnosis-20261019.md)。所有診斷terminal，本Mixing heartbeat已刪除；不復活舊排程／已完成run，不讀不動SPF。
3. 本地private operational handoff提供現有checkpoint、accepted manifests、原source／環境與new-contract起點；公開頁不放主機位址、credential位置、raw CLI、QA或逐題labels。

舊2026-10-12 handoff中的smoke／LR pilot／舊pending只作歷史，不再拿LR1e-7／WD0／一epoch的pilot當正式設定。

## 下一輪有限範圍（沿forget05設計的預設）

| 設定 | forget01 | forget10 |
|---|---:|---:|
| 正式forget QA預期數 | 40 | 400 |
| 對應retain | retain99／3960 | retain90／3600 |
| 初始target | 同一已接受Mixing-full，construction seed0 | 同左 |
| 方法／UL seeds | PureNPO alpha0、NPO+retain alpha1 × 0/1/2 | 同左 |
| 每epoch microsteps（micro4） | 10 | 100 |
| 每epoch updates（GAS8） | 2 | 13 |
| 10epoch預期micro／updates | 100／20 | 1000／130 |
| 每epoch尾組micro數 | 2 | 4 |
| 10epoch forget exposure | 400 | 4000 |

以上為**尚未執行的算術預期**，不能寫成actual PASS。預設共12個NPO run，先各split的seed0兩方法，驗收／交表後續seed1/2，不因分數改參或新增seed。construction seed仍只有0；NPO各方法與seed均從同一初始Mixing開始，不能接續forget05或另一方法的UL權重。

**資料身份先核**：使用原TOFU固定revision與原順序，核official forget01／retain99、forget10／retain90的IDs、互斥／聯集full4000、QA bytes、tokenizer/template與mask。舊`forget40`工程資料只是forget05的smoke prefix，不能因同為40筆就當official forget01。各新split的bytes SHA與contract目前尚未建立，先建立再launch。

## 保持的正式數值設定

- LR1e-5、WD0.01、beta0.1、gamma1；alpha明確0或1，retain loss=NLL；paged_adamw_32bit，Adam betas0.9/0.999、eps1e-8，clip1。
- 10epochs、world1、micro4、GAS8、global32；BF16、FlashAttention2、ZeRO3、gradient checkpointing；optimizer／param offload=none。CPU threads不因小樣本自行改16，未證實為主要瓶頸。
- **保留絕對warmup_steps=6、warmup_ratio=0**，linear scheduler總長隨actual updates改20或130。這是沿原固定超參數的交接預設；暖身比例將分別30%及約4.62%，不能聲稱跨split暖身比例相同，也不暗改成warmup2／13。
- full accumulation window loss除8；forget01尾2除2，forget10尾4除4；傳入DeepSpeed `scale_wrt_gas=False`，不加forced-boundary patch或再次除8。保存actual full／tail／scheduler／exposure trace。
- max512、assistant-only labels、EOS=PAD的有效mask、原chat template、data_seed=None的Trainer seed fallback與原retain 1:1 sampling/RNG策略保持。不為提速更改LR／WD／precision／epochs／optimizer／ZeRO或丟棄題目。
- NPO frozen ref仍是初始Mixing，初始target/ref完整身份一致、reference不更新且無gradient；alpha1無SPF projection或額外safety-anchor loss。

## 要實際套用的加速設定

| 項目 | 新campaign決定 | split-specific驗收 |
|---|---|---|
| PureNPO alpha0 retain分支 | **採用skip-retain forward/backward graph**；只算原forget NPO loss | 仍讀取／collate／消耗同retain sampling，保持input/order/mask/RNG；本次證據僅forget05首epoch兩端點，新split另驗非零LR與尾組 |
| NPO+retain alpha1 | **完整retain loss保持** | 不把零係數捷徑套到alpha1 |
| MMLU | **明確傳入HFLM(batch_size=8)**，新eval profile獨立version | 原57科／14042／5shot／ordered requests保持；新模型先小量batch1/8檢查，記LL與choice差異，不宣稱逐值等價 |
| Mixing baseline MMLU | 新profile需要**完整baseline batch8**以同protocol比較 | 現有228題microbenchmark不能當full14042；原batch1 65.653%只標歷史protocol，不直接當新batch8 delta基準 |
| MT-Bench sampled兩回合 | 原逐題sampling／各category溫度、seed、停止與turn2 context保持 | 7.18倍只來自forced-greedy64，沒有原MT batching等價證據，因此不直接batch8 |
| HEx greedy與TOFU generation | 先保持原正式生成protocol；batch8列第二階段候選 | fixed64已有2/8題token變，需原自然EOS／cap／ordered token驗證並另記新protocol後才採用，不能當已通過 |
| compile／新ZeRO／microbatch／CPU offload／precision | 不套用 | 本輪未驗證，不混入此次「apply加速」 |

skip-retain已測同39個micro耗時86.086→55.341s（約1.556倍、減35.71%）。另外一個獨立bounded run在更新2與7對全部291張量、8,030,261,248個參數的gradient及post-update FP32 master state逐bit摘要一致，50筆loss/RNG也一致。這不是完整10epoch／所有seed／新split的證明。新split驗收可選新資料小量原版/候選A/B，涵蓋**實際非零LR**與tail2／tail4；不能以第一步LR0或只看loss冒稱全參數update驗證。先明列算力、記憶體、I/O與wall界限，單次launch；不重跑舊完成run作修收據。

## 新matching references與評估

- FQ／KS／MU需要各自**matching-retain99、matching-retain90**，不能借retain95或公開TOFU retain模型。沿Mixing construction recipe由同Meta M0建立，construction seed0、原安全anchor／lambda1、5epochs／LR1e-5／WD0.01，依資料量重新bind actual schedule。
- 以原global32計，retain99預期每epoch124updates、5epoch620updates、每epoch尾24 examples；retain90預期113／565updates、尾16 examples。若沿原construction一epochwarmup則124／113；這是建plan時需核原來源的推導，非actual接受證據。
- 兩者只作evaluation retrain reference；NPO loss reference仍是Mixing。matching尚缺時如實留待完成，不用KS=0／p=1或surrogate填表。
- 四項正式評估保留TOFU／fullMMLU／MT-Bench／HEx-PHI，不用MMLUaux／IFBench／WildChat替代。
- native retainTR錯接PARA的已知缺陷保留native欄，corrected diagnostic另欄，不默默改舊數值。FQ p逐seed列、不平均，p>.05不證retrain-equivalence；不編造paired-cluster CI。
- 每個新checkpoint的HEx300+MT160是460distinct first attempts。12個NPO為5520筆、兩個新reference再920筆，**預計最多6440筆首輪新packet**；是否可重用須exact實際response／payload／profile／schema／source證據。這只是新scope規劃，不是新增paid reservation或沿用已結束scope授權。準備實際packets／重用數量與明確新預算後再處理paid授權；不因這份handoff讀key或launch Judge。

## 下一session執行順序

1. 讀本地新handoff與continuation的`next_session_handoff`。以新使用者指示為準，舊`after_forget05_task_transition`只記已完成診斷，不把它的diagnostic-only限制套作新實驗停點。
2. 唯讀核新campaign existing intent／journal／PID與checkpoint receipts。重用已驗Mixing與tokenizer，stat／manifest身份核對，不重hash既有16GB權重或重訓Mixing。
3. 建official新split資料manifest與versioned config、獨立新root／freeze／run registry。舊200/3800／70updates／tail2的worker與validator不可直接改label套用40/3960或400/3600；新增split-aware版本，舊sources／receipts不改。
4. CPU核count／mask／loss window／scheduler及negative guards；新split bounded A/B驗skip-retain，驗收actual非零LR／RNG／tail。MMLU新模型batch8作小量差異檢查，差異記錄不改原freezes。
5. freeze後單次啟動各split seed0兩方法；每run actual training、完整byte audit與native fresh-process reload驗收。若只reload／eval／receipt失敗，依原因接缺stage，不盲重訓或重launch unknown。
6. 免費正式eval／CPU原bytes聚合與兩matching references先推，不等Judge。依原有限設計續UL1/2；先交各split seed0階段表，最後逐seed＋mean±sampleSD／配對差異／coverage與unknown bounds。
7. 所有新raw、QA、responses、labels、reasons、weights、keys、CLI及trace私存，GitHub只放aggregate與方法。原forget05與已結束paid紀錄完全重用。

RunAI暫時login／readonly：首次失敗後retry9、總最多10，單層、成功即停；tokenexpired先官方login用既存憑證，不印secret。明確權限／invalid credential／完整Python或JSON錯誤／approval拒絕不當transport retry。新mutation不自動重試，未知先查own intent／journal／PID；不改VPN／DNS／MTU／TLS。舊SPF／Mixing automation不讀不復活；如需新實驗monitor，綁新scope、正常無變安靜。

完成依據與機器可讀的planned設定見[public provenance](../../../results/reproduction/safety/mixing-npo-20261019/provenance.json)、[new campaign plan](../../../results/reproduction/safety/mixing-npo-20261019/next-campaign-plan.json)。本頁是交接規格，沒有把計畫預期當已執行結果。
