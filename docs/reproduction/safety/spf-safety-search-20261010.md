# SPF safety search：每小時續行，新增最多50套模型Judge

2026-10-12 16:45 heartbeat最新：本輪先durable hourly claim，額度仍5/50 COMPLETE、0running、3960intents=receipts，45remaining。唯一原RunAI snapshot UTC2026-10-12T08:46:26.826505+00:00、SHA49a1538c4a6e0b058e116650796e57a83c7137b91e2f29dfbef6585cfe338c20確認原queue19460退出、GPUidle，newrecovery六檔／launch／journal均不存在；原CPUfailedstatus/archive/source原bytes保持，不重取11failure raw或重跑成功CPU／old模型。原CPUchild19463另由提交前live exitguard核。

明確new1645只部署3newsource＋前輪actual成功CPUproof＋newplan＋freshsnapshot六檔，portablebundle SHAbf963a90b233007f8352dee47185fa4052c2b50fd9d8675fde2e0cbe62e334f8、8chunks。Preflight與chunks0..2單次成功，chunk3 timeout exit1，333bytes nativejoinLF UTF8 capture SHA91ba1c6037cf677eb903c6920f6c989880b69969e17dcbc13e0298e156f07868與exactcommand/asrun保存private；必要sameCLI唯讀artifactcheck逐SHA核0..2、其餘缺且六目的檔/launch/journal不存在。明確r2只續chunks3..7全部單次return0，但seal cluster-status EOF exit1、86bytes SHA2cb72adf0523e67abf31a9d7f2cc4f131c97b97b397ce6b20d882662ba83126f；第二必要部署artifactcheck核8chunks完整、六目的檔/launch/journal仍不存在，不是第二GPUprogress。明確r3只seal／launch兩stage單次return0、六目的端size/SHA核。成功有限stage共11、兩transportfail原證據全保留，沒有blind retry／relaunch／重傳完成chunks、352assets或舊模型、沒有route/network修改。

UTC2026-10-12T08:52:07.937470+00:00（freshsnapshot後341.111秒，≤10min及liveGPUidle／oldPIDexitguards）已**單次提交新recovery queue PID20435**，SUBMITTED_RESULT_NOT_CHECKED，唯一launch `validation-code/hybrid01-rank40-cpu-recovery-launch-1539.json`與同名raw.log，authoritative journal `runs/cpu-fixture-recovery-1539/status.json`；原root/status.json維持immutableFAILED。成功CPUproof SHA5397aabbfcaa4d70b1b3f51004721d6d6f389212301ced815472c890bbfb6c4d逐bytes重用，沒有CPUcase重跑；原training loss／precision／projection／clip／AdamW／atol.05rtol.01保持。新queue只接unfinished actual8B2updates40examples32+8→independentreload／engineeringacceptance→full625→newaudit/all625rank40anchor→六development。**提交後沒有progressquery，不稱目前RUNNING／8B工程或full已PASS，newharmfulness未知。**勿重送20435／19460、1645/r2/r3或任何oldCPUhelpers。

候選6Judge尚未reserve/API0/keyread0；全部actualtraining/audit/development後才own792、先slot6記帳再savedDPAPIchild解密。最佳已驗λ.1/rank20 H27/300＝9.00%，Mixing13/300＝4.33%，未達H≤12/300停止條件。下一routine不得早於UTC2026-10-12T09:46:12.350664+00:00，提早event直接skip、不sleep/poll/reset/補查。Native ACTIVE/hourly/failed_runs_only與188字元短prompt保持；gate insufficient_evidence、humanadjudication／pairedclusters／matchingmetrics／原NPO／四大型autograd／單seed跨平台限制保持。Public僅source/aggregate/asrun/docs，原raw/key/QA/responses/labels/weights/ledger全private。詳見[續行提交證據](../../../results/reproduction/safety/spf-npo-gb200-20261006/spf-rank40-recovery-launch-20261012-1645.json)。以下進度保留為歷史。


2026-10-12 15:39 heartbeat最新：rank40候選原queue19460與CPUchild19463均已退出，唯一RunAI snapshot UTC2026-10-12T07:42:17.724962+00:00确认GPUidle。原status FAILED_NO_RETRY／completed[]，停在hybrid-cpu-verification、child return1；actual8B工程、full訓練與checkpoint尚未開始。11新failure source／plan／bound snapshot／exactcommand／status／原raw逐bytes SHA immutable保存，archive3f5c58114b262673e8f54d93b14eec2adcff57d75186908eb0ef26fc9a0d346d，全部原raw維持private、不覆寫原run。

同原venv有限唯讀CPU數值觀測定位step1：FP32 staged loss4.290201723575592、direct4.290201187133789，差5.364418029785156e-7略高於原5e-7門檻；task gradient最大誤差1.4901161193847656e-8、anchor4.656612873077393e-10。這是CPU fixture的scalar reduction差異線索，不能當8B訓練或大型autograd驗收。新的明確來源只將CPU CE參考scalar改FP64；parameters／gradients／projection／AdamW仍FP32，原loss／gradient門檻、GPU arithmetic與atol.05/rtol.01不變。新CPU兩次AdamW／直接autograd／unequal tokenmean／conflict projection／儲存重載已actual PASS：最大gradient error1.4901161193847656e-08、update error7.450580596923828e-09，CUDA false、8B/API/key0；實際執行exact source-bound記憶體CPU function，新verifier檔尚未部署。

3newsource與10新增schema negatives通過，source-bound successfulCPUproof將逐bytes重用，不重跑CPUcase。Explicit new journal `runs/cpu-fixture-recovery-1539`及唯一planned launch `validation-code/hybrid01-rank40-cpu-recovery-launch-1539.json`已source-ready；原root/status.json保留FAILED，續行source嚴核原failedstatus／exactcommand／sources／PIDexit，拒任何completed checkpoint或newjournal。只接尚未完成8B工程2updates40examples32+8→原threshold independentreload→full625→newaudit/all625rank40+anchor→六development。原scalar/preparerquote／missingrelay／參數Binding與numericobserver失敗source及metadata均保存，不冒稱不可取得的原tool streambytes。

本輪snapshot已過10min，**沒有remote部署／newGPU submission／paidJudge／keyread或第二progress query**。下一eligible排程先claim，僅原CLI/workspace必要newrecovery snapshot，GPUidle／oldPIDexit與新完整snapshot≤10min才以newtag傳3source＋成功CPUproof＋newplan＋freshsnapshot六檔，單次有限launch；不要重送19460、重跑1336或本輪CPU helpers。額度**5/50 COMPLETE、45remaining**，3960intents=receipts，bestH27/300＝9.00%，Mixing13/300＝4.33%，rank40 H未知；fulldevelopment後才reserve6及own792。下次routine不得早於UTC2026-10-12T08:41:57.211061+00:00；提早事件直接skip，不等候／輪詢／reset／補查。Native ACTIVE/hourly/failed_runs_only及188字元prompt不變。H≤12/300或50額度耗盡才停止；gate insufficient_evidence與全部原研究缺口保持。詳見[CPU失敗與續行修復證據](../../../results/reproduction/safety/spf-npo-gb200-20261006/spf-rank40-cpu-failure-and-recovery-20261012-1539.json)。以下進度保留為歷史。


2026-10-12 13:36 heartbeat最新：第五套hybridλ.3／LR5e-6／β1=.5／rank20的792 Judge全部成功，新raw/schema/payload/order/model/usage／2376events與792ledger intents/receipts逐核，duplicates0／zero retry；Campaign.finish COMPLETE。Harmfulness **33/300＝11.00%**，safe_non_refusal67／refusal200；最佳λ.1為27/300＝9.00%，Mixing13/300＝4.33%，尚未達H≤12/300停止條件。Benign substantive339/non_answer4/refusal7；conversation overall448/142＝3.155，normal140/degenerate2，nullable分母見aggregate。Actualmodel792均gpt-5.6-terra，input429787/output111469。Audit SHA7c1a87cefbec6e1e65e30154c4a81d9642311a9182d311934fc02f30e9184d30；完整labels/reasons/source-map private，前4套只SHA引用、不重審。

λ.3的描述性結果未優於λ.1，不能假定加大安全loss係數必然降低H。免費utility重用已完成aggregate：TOFU f05/r95 EM96.00%/96.63%、MMLUaux847/1024＝82.71%（zero-shot非formal5shot），IFBenchstrictprompt21%。下一候選以最佳λ.1／LR5e-6／β1=.5為控制，**只改SPF投影rank20→40**，保留MetaM0獨立construction、原training-only anchor／5epochs625／六development，不回灌development650或FormalHExPHI。假說是更大的anchor-gradient子空間上限可能減少安全能力侵蝕，同報TOFU／benign拒答／conversation與其他utility代價；明確family hybrid。

4newsource compile、14新recipe/decision/predecessor/absence negatives拒絕，5原queue functions AST與verifierloader/main保持，原generic helper／auditor／precision／reloadthreshold不變。新rank40 CPU直接autograd72task/56anchor tokens／兩次AdamW／reloadfixture與actual8B40examples2steps32+8／micro8+2／原atol.05rtol.01獨立reload，全部通過才full625→newaudit→all625rank40/addition→六development；目前新actual結果未知。

唯一RunAI snapshot UTC2026-10-12T05:52:12.051399+00:00、SHA52f57a0bb9238ad9ca7ccedab117cdefeef77b28e3e445e095b426073730483e確認16733完成退出／GPUidle／newroot不存在。9chunks＋preflight/seal/launch共12stage單次return0，7目的檔逐size/SHA核，無transportfailure／retry／artifactcheck。UTC2026-10-12T05:53:03.801241+00:00（snapshot後51.750秒）單次提交rank40 queue **PID19460**，root `/data/spf-hybrid01-rank40-full-20261012-r1`，SUBMITTED_RESULT_NOT_CHECKED；沒有提交後progress，不稱目前RUNNING／新工程驗收pass，不重送19460／1336／oldhelpers。

額度**5/50 COMPLETE、45可保留**，3960intents=receipts；候選6尚未reserve Judge，全部actualdevelopment完成後才own792且先記帳。下一routine不得早於UTC2026-10-12T06:38:19.104888+00:00，先claim後一次必要snapshot，禁止輪詢／重取completedraw／重hash16GB。Native ACTIVE/hourly/failed_runs_only與188字元短prompt不變；H≤12/300或50用盡才停止並PAUSE本搜尋。Gate insufficient_evidence、humanadjudication／pairedclusters／matchingmetrics／原NPO／四大型autograd／單seed跨平台限制保持。詳見[完整Judge與rank40提交證據](../../../results/reproduction/safety/spf-npo-gb200-20261006/spf-hybrid03-judge-complete-rank40-launch-20261012-1336.json)。以下舊進度保留為歷史。


2026-10-12 11:42 人類續行最新：SPF＋安全loss hybrid λ.3／LR5e-6／β1=.5完整625updates/5epochs20k、independent reload error0、新權重audit與六development全部完成。唯一RunAI snapshot UTC2026-10-12T03:43:42.288707+00:00、SHAa4927226b959af5d7c3f370a4bb6d159f185f3b948ff16afb23d483612315263確認queue16733退出／GPUidle。41新development source／commands／raw／outputs逐bytes SHA immutable保存，archive908071fea442bc94ed1d147dd15060b80d1d17e35a67769df008f5f29fd47e5c；原19train與23engineering只manifest/SHA重用，不重取完成raw、不重hash16GB或重跑模型評估。

免費描述性結果：TOFU f05/r95 EM 96.00%/96.63%、ROUGE-L F1 0.990522/0.991355；MMLUaux 847/1024＝82.71%，zero-shot非formal5shot；IFBench300完整summary見aggregate。全625update anchor addition與original thresholds保持，明確family hybrid，legacy frozen audit methodspf原bytes保留，不能冒稱純SPF。

Own792 packet固定原prompts/order與candidate自身response/context、原rubrics/SDK/profile preflight pass。UTC2026-10-12T03:47:26.1641582Z正常require_escalated單次提交DPAPI loader53980，先reserve第5套再child解密，無新keyUI。唯一localstartup UTC2026-10-12T03:48:28.762677+00:00：runner62236 RUNNING、20success＋1in-flight，actualmodel20均gpt-5.6-terra、medium4096/zero retry；H未知，不能稱full792或低於Mixing。

額度5/50 reserved（4complete／1running、45可保留），目前已驗最佳H27/300＝9.00%，Mixing13/300＝4.33%，只SHA引用。下一routine不得早於UTC2026-10-12T04:43:23.281837+00:00，先claim後只必要localJudge一次；completedRunAI不再查、不得重送53980/62236。H≤12/300或50額度耗盡才停止並PAUSE本搜尋。Native ACTIVE/hourly/failed_runs_only與188字元短prompt不變；human50盲審private未adjudicate，gate insufficient_evidence、pairedclusters／matchingmetrics／原NPO／四大型autograd與單seed跨平台缺口保持。詳見[完成評估與第五套Judge啟動證據](../../../results/reproduction/safety/spf-npo-gb200-20261006/spf-hybrid03-complete-judge-start-20261012-1142.json)。以下舊進度保留為歷史。


2026-10-12 08:32 heartbeat最新：hybridλ.3／LR5e-6／β1=.5完整construction已完成。唯一RunAI snapshot UTC2026-10-12T00:33:10.074062+00:00、SHA6810ec435140fd162080dac238fcbbadbd220f083f5cb3a9bb2a849e87eaa895確認queue16733仍存活，stage candidate-development/tofu／GPU[18645]。Actual5epochs625updates20k4000unique、696525 assistant／10625 training-only anchor tokens computed、272 conflicting updates；獨立freshreload原atol.05/rtol.01通過、maxlogiterror0，新291BF16／32layers／16,060,556,616bytes及157/313/469/625 trajectory audit pass。

全部625actualupdate逐核原SPF投影後加入.3×原training-only anchor梯度，再原clip/AdamW，沒有額外forward/backward，明確family SPF＋安全loss hybrid；original frozen CLI/auditor methodspf原bytes保留。全625acceptance SHAea346a7696d1302acdcae591e99e65cdbbfb65635e594d3e9865d17afc18bb53、audit SHAd27ab4ec05ddfedcc7b0a2d5b7260c32d5971918613d1326af26f825a6ddd486、identity4ab6af98cf33e20e50394530d2fe6de69aa9e985188cd23cbfefd9a4f002d33d。19新完成binding／resolved／trajectory／audit／commands／原raw逐bytes SHA私有保存，archive7a4b1a71b84bb02d37cf512cd01af73da1eb2182bd1b4bb0bc9912e1d20ca250；前23工程檔只SHA重用，不重取完成raw／重hash16GB或重跑工程。公開scope僅aggregate／audit／acceptance／asrun來源／報告與provenance，全部原raw logs維持private。

六development尚未完成，新harmfulness未知；有限queue自動評估，全部驗收後才candidate05 own792 Judge且先reserve第5套。額度4/50 COMPLETE、46可保留，3168intents=receipts，本輪API/keyread/GPUsubmission0。已驗最佳hybridλ.1 H27/300＝9.00%，Mixing13/300＝4.33%，尚未達H≤12/300停止條件。Native ACTIVE/hourly/failed_runs_only與短prompt不變，下一routine不得早於UTC2026-10-12T01:32:54.555877+00:00；禁止輪詢／重送16733。Gate insufficient_evidence，人工adjudication／pairedclusters／matchingmetrics／原NPO／四大型autograd與單seed跨平台限制保持。詳見[完整訓練與評估進度證據](../../../results/reproduction/safety/spf-npo-gb200-20261006/spf-hybrid03-train-complete-development-running-20261012-0832.json)。以下舊進度保留為歷史。


2026-10-12 04:28 heartbeat最新：唯一原RunAI snapshot UTC2026-10-11T20:28:59.555864+00:00、SHA439aaaa764ef2ddcc4216e4178daff27917bdbe91a0a3d1c2bfa245dbc9f6f6c確認hybridλ.3 queue16733存活，stage train、GPU compute [17834]；四個新工程stage均return0且completed。CPU直接autograd／assistant-token mean／兩次AdamW更新／serialization比對PASS，最大gradient error2.9802322387695312e-08、update error0.0。Actual8B／ZeRO3工程40examples、2updates、windows32+8／micro8+2，原SPF投影後加入.3×原training-only anchor梯度、再原clip/step；獨立freshreload原atol.05/rtol.01通過、maxlogiterror0。這是新係數工程驗收，原四大型autograd歷史限制仍未解除。

23新source／exactcommands／四原raw／binding／resolved／trajectory／兩actualproof逐bytes SHA immutable保存，archive612d2f656c483b518fab1e101ea672cbea4852b997ff3020ab9e2d2e50be926b；CPUproofa30986ee9fd7ccdecdbf85af09d10f2dfeba987b1b2366e3b2378ec3fb8be7e6、8Bproofe2117ce5f8b045b5a21f45d6438a35745e88edbb14021f794542f6a912c11ace。Export只取新completed工程檔，不是第二progress、不讀weights或重跑任何工程／舊模型／評估／paidJudge。完整625訓練／audit／六development／新H尚未完成；queue已自動接完整訓練，全部驗收後才第5套own792且先記帳。

額度仍4/50 COMPLETE、46可保留，3168intents=receipts，本輪API/keyread/GPUsubmission0；目前最佳已驗hybridλ.1 H27/300＝9.00%，Mixing13/300＝4.33%，尚未達H≤12/300停止條件。Native ACTIVE/hourly/failed_runs_only與188字元短prompt不變，下一routine不得早於UTC2026-10-11T21:28:44.850823+00:00；禁止輪詢／重送16733／重取completedraw／重hash16GB。Gate insufficient_evidence，人工adjudication／pairedclusters／matchingmetrics／原NPO與單seed跨平台限制保持。詳見[新係數工程完成與訓練進度證據](../../../results/reproduction/safety/spf-npo-gb200-20261006/spf-hybrid03-engineering-complete-train-running-20261012-0428.json)。以下舊進度保留為歷史。


2026-10-12 03:27 heartbeat最新：第四套SPF＋安全loss hybrid λ.1／LR5e-6／β1=.5的792 Judge全部成功，新增raw/schema/payload/order/model/usage／2376events與792 ledger intents/receipts逐核，duplicates0、zero retry；Campaign.finish COMPLETE。Harmfulness **27/300＝9.00%**，safe_non_refusal68／refusal205，是目前已驗最低；同LR純SPF41/300＝13.67%，Mixing13/300＝4.33%，尚未達H≤12/300停止條件。Benign substantive340/non_answer2/refusal8（同LR純SPFrefusal6）；conversation overall439/142＝3.092，quality全部normal，nullable分母見aggregate。Actualmodel792全部gpt-5.6-terra，input432468/output110155。Audit SHAa6ca483d3e4dd34944d93be18460596c8c944c206a13244ade5ef52c6e38cf89；完整labels/reasons/source-map private，前3套只SHA引用、不重審。

免費描述性utility只重用已完成aggregate：TOFU f05/r95 EM96.00%/96.63%、MMLUaux848/1024＝82.81%（zero-shot非formal5shot），IFBench原結果保留。下一候選**只把training-only安全anchor loss係數.1提高到.3**，保留LR5e-6／β1=.5與原MetaM0獨立construction、原資料／rank20／5epochs625／六development；不回灌development650或FormalHExPHI。假說是更強原安全anchor目標可能進一步降低H，同報benign拒答、TOFU與conversation代價。明確family hybrid，不能冒稱純SPF或合格安全gate。

4newsource compile、11新recipe/decision/predecessor/absence negatives拒絕；原generic helper bytes與原auditor route保持。新λ.3先CPU直接autograd/tokenmean/兩次AdamW，再actual8B40examples2steps32+8／micro8+2／原atol.05rtol.01獨立reload工程驗收，全部通過才full625train→reload→newaudit→六development；目前actual結果未知。

唯一RunAI snapshot UTC2026-10-11T19:34:57.174566+00:00、SHAc0cf4fc7c1b74e56d2b472adff54ef1f75b666f4009092f9ffd94447aaac5e1e確認14033完成退出／GPUidle／newroot不存在；9chunks＋preflight/seal/launch共12stage各單次return0，7目的檔逐size/SHA核，無transportfailure／retry／artifactcheck。UTC2026-10-11T19:35:35.112859+00:00（snapshot後37.938秒）單次提交hybrid λ.3 queue **PID16733**，root `/data/spf-hybrid03-lr5e6-full-20261012-r1`，SUBMITTED_RESULT_NOT_CHECKED；沒有提交後progress，不能稱目前RUNNING或newtrain/H pass，不重送16733／0327／oldhelpers。

額度**4/50 COMPLETE、46可保留**，3168intents=3168receipts；候選5尚未reserve Judge，actual全部development完成後才own792且先記帳。下一routine不得早於UTC2026-10-11T20:27:52.811330+00:00，先claim後一次必要snapshot，禁止輪詢／補查／重取completedraw／重hash16GB。Native ACTIVE/hourly/failed_runs_only、188字元短prompt不變；H≤12/300或50用盡才停止並PAUSE本搜尋。Gate insufficient_evidence，humanadjudication／pairedclusters／matchingmetrics／原NPO與四大型autograd／單seed跨平台限制保持。詳見[完整Judge與下一係數提交證據](../../../results/reproduction/safety/spf-npo-gb200-20261006/spf-hybrid01-judge-complete-hybrid03-launch-20261012-0327.json)。以下舊進度保留為歷史。


2026-10-12 02:25 heartbeat最新：SPF＋安全loss hybrid λ.1／LR5e-6／β1=.5完整625updates/5epochs20k、independent reload error0、新權重audit與六development全部完成。唯一RunAI snapshot UTC2026-10-11T18:25:59.884132+00:00、SHA0c2bb060ff1fad203aac824f63eeb1fea0daa99385fa98f23344b7b7ffb383b0確認queue14033退出／GPUidle。41新development source／commands／raw／outputs逐bytes SHA immutable保存，archive88122a73c19ec21eb7866b28f4247d71e791577841579aee7a338179a501b605；原19train與23engineering只manifest/SHA重用，不重取完成raw、不重hash16GB或重跑模型評估。

免費描述性結果：TOFU f05/r95 EM 96.00%/96.63%、ROUGE-L F1 0.989940/0.991739；MMLUaux 848/1024＝82.81%，zero-shot非formal5shot；IFBench300完整summary見aggregate。全625update anchor addition與original thresholds保持，明確family hybrid，legacy frozen audit methodspf原bytes保留，不能冒稱純SPF。

Own792 packet固定原prompts/order與candidate自身response/context、原rubrics/SDK/profile preflight pass。UTC2026-10-11T18:29:26.0638559Z正常require_escalated單次提交DPAPI loader63284，先reserve第4套再child解密，無新keyUI。唯一localstartup UTC2026-10-11T18:30:12.152322+00:00：runner65816 RUNNING、12success＋1in-flight，actualmodel12均gpt-5.6-terra、medium4096/zero retry；H未知，不能稱full792或低於Mixing。

額度4/50 reserved（3complete／1running、46可保留），目前已驗最佳H38/300＝12.67%，Mixing13/300＝4.33%，只SHA引用。下一routine不得早於UTC2026-10-11T19:25:43.131665+00:00，先claim後只必要localJudge一次；completedRunAI不再查、不得重送63284/65816。H≤12/300或50額度耗盡才停止並PAUSE本搜尋。Native ACTIVE/hourly/failed_runs_only與188字元短prompt不變；human50盲審private未adjudicate，gate insufficient_evidence、pairedclusters／matchingmetrics／原NPO／四大型autograd與單seed跨平台缺口保持。詳見[完成評估與第四套Judge啟動證據](../../../results/reproduction/safety/spf-npo-gb200-20261006/spf-hybrid01-complete-judge-start-20261012-0225.json)。以下舊進度保留為歷史。


2026-10-12 00:23 heartbeat最新：hybridλ.1／LR5e-6／β1=.5完整construction已完成。唯一RunAI snapshot UTC2026-10-11T16:24:19.600263+00:00、SHA9f7e0e8e7b5a6b9626e96219bc7bbb8a49f400f523109e7547349620a276b00f確認queue14033仍存活、stage candidate-development/tofu／GPU[15918]。Actual5epochs625updates20k4000unique、696525 assistant／10625 training-only anchor tokens computed、279 conflicting updates；獨立freshreload原atol.05/rtol.01 pass、maxlogiterror0，新291BF16／32layers／16,060,556,616bytes及157/313/469/625 trajectory audit pass。

全部625actualupdate逐核SPF投影後加入.1×原training-only anchor梯度，再原clip/AdamW，沒有額外forward/backward，明確family SPF＋安全loss hybrid；original frozen CLI/auditor仍記methodspf原bytes，不改成純SPF驗收。全625acceptance SHA5457809bceb7539fb13618ee3e3b8c736a2224d02fa838a0fe59a90e68621c6c、audit SHA6ce8057e3d60ea705794c6c308b6a0563555caad05362fe7864a7db1d75c0931、identity7c3636e230dc3c53ee824572488191ae103f5c5fa4f85af28eb7b79a4ef14962。19新完成binding／resolved／trajectory／audit／commands／原raw逐bytes SHA immutable保存，archive6572796b6d18d71814eb4be9f49998428df492885d1bbf3255d2f6c4ce163baa；前23工程檔只原manifest/SHA重用，不重取完成raw／重hash16GB／重跑工程或模型。

六development目前尚未完成、新harmfulness未知；原有限queue自動評估，完整驗收後才candidate04 own792 Judge且先reserve第4套。額度仍3/50 COMPLETE、47可保留，2376intents/receipts，本輪新API/keyread/GPUsubmission0。已驗最佳H38/300＝12.67%，Mixing13/300＝4.33%，尚未達H≤12/300停止條件。Native ACTIVE/hourly/failed_runs_only與188字元短prompt不變，下一routine不得早於UTC2026-10-11T17:24:06.160719+00:00；禁止第二progress／輪詢／重送14033。Gate insufficient_evidence，人工adjudication／pairedclusters／matchingmetrics／原NPO／四大型autograd與單seed跨平台限制保持。詳見[完整訓練與評估進度證據](../../../results/reproduction/safety/spf-npo-gb200-20261006/spf-hybrid01-train-complete-development-running-20261012-0023.json)。以下舊進度保留為歷史。


2026-10-11 21:21 heartbeat最新：唯一原RunAI snapshot UTC2026-10-11T13:26:08.850038+00:00、SHA5288dbaf61c7576ade7f69e82e6f320f43e7a4d3f7688d92182115977f30ca42確認hybrid queue14033存活，stage train、GPU compute [15134]；四個新工程stage均return0且completed。CPU直接autograd／assistant-token mean／兩次AdamW更新／serialization比對PASS：最大gradient error2.9802322387695312e-08、update error0。Actual8B／ZeRO3工程40examples、2updates、windows32+8／micro8+2，原SPF投影後加入.1×training-only anchor梯度、再原clip/step；独立freshreload原atol.05/rtol.01通過、maxlogiterror0。此是工程驗收，四大型autograd歷史限制仍未解除。

23新source／exactcommands／四原raw／binding／resolved／trajectory／兩actualproof逐bytes SHA immutable保存，archivedd29d5295eb064ac74d55f3e5bbd743299b12241a8cf4adcbd57a6ccdab756fe；CPUproof82038714b8837eb716bf17e80402b775007db532eb8acd61f7c020f255eda42a、8Bproofa0a7bd95f69d6f14ce40115ce281c6c7effc9e0550195cafbea3a7c9228e8fae。Export只取新completed工程檔，沒有第二progress、weights read或重跑工程／旧模型／評估／paidJudge。完整625訓練／audit／六development／H尚未完成；queue已自動接完整訓練，沿已授權順序完成後才第4套own792 Judge。

額度仍3/50 COMPLETE、47可保留，2376intents/receipts，本輪新API/keyread/GPUsubmission0。目前最佳已驗38/300＝12.67%，Mixing13/300＝4.33%，停止條件H≤12/300未達。Native ACTIVE/hourly/failed_runs_only與188字元短prompt不變；下一routine不得早於UTC2026-10-11T14:25:45.352658+00:00，禁止輪詢／重送14033／重取完成raw／重hash16GB。Gate insufficient_evidence，人工adjudication／pairedclusters／matchingmetrics／原NPO及單seed跨平台限制保持。詳見[工程完成與訓練進度證據](../../../results/reproduction/safety/spf-npo-gb200-20261006/spf-hybrid01-engineering-complete-train-running-20261011-2121.json)。以下舊進度保留為歷史。


2026-10-11 19:21 heartbeat最新：第三套β1=.5／LR3e-6的792 Judge全部成功且完成新增raw/schema/payload/order/model/usage／2376events與792 ledger intents/receipts逐核；duplicates0、zero retry，Campaign.finish COMPLETE。Harmfulness **38/300＝12.67%**，safe_non_refusal74／refusal188；前一LR5e-6為41/300＝13.67%，Mixing13/300＝4.33%。只下降1個百分點，仍未達H≤12/300停止條件。Benign substantive341/non_answer4/refusal5；conversation overall451/142＝3.176、quality degenerate1/142，nullable分母見aggregate。Actualmodel792均gpt-5.6-terra、input428836/output109500。Audit SHA5969b6411a91c643d7e8cf6317d5c1bb20d69fafb87a97c3e179ae707850e556、完整labels/reasons/source-map保持private；前兩套只SHA引用。

LR3e-6免費TOFU f05/r95 EM86.00%/84.74%（LR5e-6為96.50%/96.39%），MMLUaux846/1024、IFBench300原結果只SHA重用。安全改善小而學習代價明顯，因此下一候選回到LR5e-6／β1=.5控制組，僅新增training-only anchor loss係數.1。明確標為 **SPF＋安全loss hybrid**：原SPF投影後加.1×原安全anchor梯度，再原clip/AdamW；loss均為原assistant-token mean，沒有多一次模型forward/backward，也沒有650 development或FormalHExPHI回灌。原MetaM0獨立construction、rank20／5epochs625／原数据與完整六development保持。

新CPU arithmetic verifier與actual8B40examples兩步32+8／micro8+2／serialization+原atol.05rtol.01 reload驗收在新有限queue前四stage，全部通過才自動接完整625train→reload→newaudit→六development。五新source compile與5新schema negatives完成，僅準備證據，不能稱actual工程或train pass。初始本機predecessor branch/status replacement ValueError、partialsource保存；明確r2修identity，r3只審查還原原auditor path，沒有重跑completedcases。

唯一新RunAI snapshot UTC2026-10-11T11:35:21.715202+00:00、SHA752384e32ec448c38c4dc87e7e65242bc979976ff383d75553467c17759275b4確認原12367退出／GPUidle／newroot不存在；9chunks＋preflight/seal/launch共12有限stage各單次return0，7目的檔逐size/SHA。UTC2026-10-11T11:36:18.467342+00:00（snapshot後56.752秒）單次提交hybrid queue **PID14033**，root `/data/spf-hybrid01-lr5e6-full-20261011-r1`。SUBMITTED_RESULT_NOT_CHECKED，沒有提交後progress query，actualCPU/工程/fulltrain/eval/H未知；不得重送14033或本輪1921/oldhelpers。

新增額度3/50 COMPLETE、47可保留（2376intents/receipts）；hybrid尚未reserve第4套，完整actualdevelopment後才Judge。下一routine不得早於UTC2026-10-11T12:25:20.887831+00:00，先claim後僅原路徑一snapshot、禁止輪詢／補查／重取完成raw／重hash16GB。Native ACTIVE/hourly/failed_runs_only、188字元短prompt不變；H≤12/300或50耗盡才停止並PAUSE。此是單seed描述性搜尋，gate insufficient_evidence、humanadjudication／pairedclusters／matchingmetrics／原NPO及四大型autograd限制仍保留。詳見[完整Judge與hybrid提交證據](../../../results/reproduction/safety/spf-npo-gb200-20261006/spf-lr3e6-judge-complete-hybrid01-launch-20261011-1921.json)。以下舊進度保留為歷史。


2026-10-11 18:20 heartbeat最新：LR3e-6／β1=.5候選完整train625/5epochs20k、independentreload error0、新291BF16/32layers byteaudit及六development全部完成。唯一RunAI snapshot UTC2026-10-11T10:21:06.271976+00:00、SHA3944b3e878184431c3e1e7967398c2fe4ebb43d63fcb16a282656b9a0033dda2，queue12367已退出／GPUidle。41新development source/commands/raw/outputs逐bytes SHA保存，archive25dfa3573740b65c5f315c9b9eac2597747a1913e1f9e39561b22faa68e15841；17:20原23訓練檔只重用local manifest／audit SHA，不重取完成raw、不重hash16GB或重跑評估。

免費描述性結果：TOFU f05/r95 EM 86.00%/84.74%，ROUGE-L F1 0.952805/0.958017；MMLUaux 846/1024＝82.62%（zero-shot，非formal5shot）；IFBench300完整score摘要見aggregate。這些結果不代安全gate。

原fixed prompts/order搭配candidate own responses/context製成792 packet、原rubrics/SDK/profile CPUpreflight pass。依已授權50模型與具體外部payload範圍，正常require_escalated審查接受，UTC2026-10-11T10:24:01.6696497Z單次提交savedDPAPI loader63640，先reserve第3套後才子程序解密，沒有新keyUI。唯一local啟動觀測UTC2026-10-11T10:24:37.690280+00:00：runner61532 RUNNING，10/792 success＋1 in-flight intent，actualmodel10均gpt-5.6-terra，medium4096／zero retry。新harmfulness未知，不能稱完整792或低於Mixing。

新增額度3/50 reserved（2complete／1running，47可保留）；目前已核最佳仍LR5e-6 H41/300＝13.67%，Mixing13/300＝4.33%，只SHA引用，舊Judge不重跑。下一routine不得早於UTC2026-10-11T11:20:49.393483+00:00，先claim後僅查candidate03 localJudge一次，不再查已完成GPU或取舊raw。H≤12/300或50用盡才停止並PAUSE搜尋。Native ACTIVE/hourly/failed_runs_only、188字元prompt不變；50row盲審表private、未人工adjudicate，gate/pairedclusters/matchingmetrics/原NPO/autograd/單seed跨平台缺口保留。詳見[完成評估與第三套Judge啟動證據](../../../results/reproduction/safety/spf-npo-gb200-20261006/spf-lr3e6-complete-judge-start-20261011-1820.json)。以下舊進度保留為歷史。


2026-10-11 17:20 hourly最新：LR3e-6／β1=.5候選的完整訓練、獨立reload與new權重audit已完成。唯一RunAI快照UTC2026-10-11T09:20:31.850722+00:00，queue12367存活，stage candidate-development/safety／GPU13628；完整六development尚未完成，新harmfulness未知，不能稱安全改善。Actual5epochs625updates20k4000unique、696525 assistant／10625 training-only safety tokens、280 conflicting updates；freshreload error0，291BF16／32layers／16,060,556,616bytes及157/313/469/625 trajectory原newaudit pass。

23新完成小檔source／command／raw／binding／runtime／trajectory／audit逐bytes SHA immutable保存，archive1310647eca2d4e5ad4064d8efac839809261056ad09f6ed80efb65b857baa3f2；newaudit SHA839156fd5c6d0e28847991188a95061d513727cf012ec912465697ca9450c181、model identityf7afea12f1363a9fea3b4062fde32ae53c40fadd14978a85f387d0f408087b92。Export不查development進度、不讀weights、不重算audit；舊完成模型、評估與paidJudge未重跑。下一hourly只能一次必要快照，確認actual六stage完成後只取新development證據，重用這23檔與原audit SHA；再製candidate own792 packet／CPUpreflight，先保留第3套額度才用已授權savedDPAPI key送Judge。

額度仍2/50 COMPLETE、48可保留；目前最佳H41/300＝13.67%（LR5e-6），Mixing13/300＝4.33%，本候選尚無Judge。下一routine不得早於UTC2026-10-11T10:20:11.656050+00:00，提前skip、不輪詢／重送12367。Native ACTIVE每小時／failed_runs_only、短prompt與schedule不變；H≤12/300或50配額用盡才停止並PAUSE搜尋。Gate insufficient_evidence，人類adjudication、pairedclusters、matching metrics、原NPO／autograd及單seed跨平台限制保留。詳見[訓練完成與評估進度證據](../../../results/reproduction/safety/spf-npo-gb200-20261006/spf-lr3e6-train-complete-development-running-20261011-1719.json)。以下舊進度保留為歷史。


2026-10-11 11:18 heartbeat最新：第2套β1=.5／LR5e-6的792 Judge actual全部成功。本小時唯一local觀測UTC2026-10-11T03:21:07.198967+00:00，原rubrics／schema／payload／order／rawSHA／model／usage／2376events及792ledger intents/receipts逐核，Campaign.finish COMPLETE，audit SHA2e65b09481c974b4c03fbded6faf5a4271c95f3c136286e669452ba446ace7ac。Harmfulness **41/300＝13.67%**，harmful refusal186／safe_non_refusal73；前一β1=.5／LR1e-5為93/300＝31%，原SPF為141/300＝47%，Mixing為13/300＝4.33%。相較前一套下降17.33個百分點，仍未達嚴格低於Mixing的停止條件。

Benign substantive341／refusal6／non_answer3（350）；conversation overall436/142＝3.070，relevance573/142＝4.035，factualaccuracy338/120＝2.817，consistency471/109＝4.321，quality degenerate3/142，nullable分母保留。Actualmodel792均gpt-5.6-terra、input428891/output111499、medium4096／zero retry。免費TOFU f05/r95 EM96.50%/96.39%、MMLUaux846/1024與六development原完成結果只SHA重用；沒有重跑completed GPU、paid success或舊audit。這是描述性seed0比較，非pairedcluster顯著性或合格安全gate。

依已核改善趨勢與既定順序，下一單因素候選只把LR5e-6→3e-6，保留β1=.5／原MetaM0／5epochs625updates20k4000unique／原training-only anchor、rank20、micro4acc8global32、warmup125、max512、BF16/FA2及完整六development。三newsource inversebytes／九controller AST／9新identity-rate-boundary CPU negatives核通過，並非actual新train。唯一完整RunAI snapshot UTC2026-10-11T03:24:03.961377+00:00、SHAb482ab8261a6e793722bc22f4af32d6ba0ad62a6b31e1f69aabde725ffc0671b確認10758退出／GPUidle／新root不存在。Preflight WebSocket EOF與chunk2 cluster-status EOF原CLI分別279/86bytes保存；兩次同CLI必要artifactcheck只核newfiles/chunks/launchabsence，r2完成0/1、r3只續缺2..6/seal/launch，沒有重傳0/1或completed assets/models。

UTC2026-10-11T03:26:50.537101+00:00在snapshot後166.576秒及liveGPU/PID/source guards後，單次提交newqueue **PID12367**，root `/data/spf-beta05-lr3e6-full-20261011-r1`、唯一launch `validation-code/beta05-lr3e6-full-launch-20261011.json`。狀態SUBMITTED_RESULT_NOT_CHECKED，沒有提交後progress query；新train/eval/H未知，不能稱目前RUNNING或完成。勿重送12367或1118/r2/r3 helpers；下一hourly只用新snapshot source讀此queue必要進度。

新增額度 **2/50 COMPLETE、48尚可保留**，本輪新API/keyread0，第三候選尚未reserve Judge；actual完整train/reload/audit/六development後才做其own792並先記帳。下一routine不得早於UTC2026-10-11T04:21:07.182463+00:00，先claim、提早skip／不等待輪詢；H≤12/300或50耗盡才PAUSE搜尋。Native ACTIVE/hourly/failed_runs_only、188字元短prompt未改。Humanadjudication／pairedclusters／matchingmetrics／原NPO与autograd缺口保留，gate insufficient_evidence。詳見[完整Judge與下一候選提交證據](../../../results/reproduction/safety/spf-npo-gb200-20261006/spf-lr5e6-judge-complete-lr3e6-launch-20261011-1118.json)。以下舊進度保留為歷史。


2026-10-11 09:17 heartbeat最新：原RunAI恢復可讀，唯一完整snapshot UTC2026-10-11T01:19:16.366927+00:00、SHA605364286b1af8675bf23fb0561e8da3913a90da95641aa42403f4e779ddf13a。LR5e-6／β1=.5候選actual full625updates/5epochs/20k/4000unique、freshreload error0、291BF16/32layers/16,060,556,616bytes及157/313/469/625 trajectory原newaudit pass，六development全部return0；queue10758已退出、GPUidle。65新完成source/exactcommands/raw/binding/audit/outputs逐bytes SHA immutable保存，archive6255567ac95ea8341ac169c356b908517f2ea4aecb4801c4eb5f028bb5b4342f；沒有重跑舊模型/評估/Judge或重hash16GB。先前auth失敗完整保留為歷史。

免費描述性結果：TOFU f05/r95 exactmatch96.50%/96.39%、ROUGE-L F1 0.986667/0.990898；MMLUaux846/1024=82.62%（zero-shot，非formal5shot）。IFBench全部300原score完成，完整摘要見本輪aggregate。這些learning/capability結果不代安全gate。

新candidate own responses/context的固定792 packet与actual proof通過原SDK/rubric/endpoint/profile CPUpreflight。沿已授權50模型scope正常require_escalated審查接受，UTC2026-10-11T01:22:00.8955259Z單次提交DPAPI savedkey loader66512，reserve第2套後才程序內解密、無新UI。唯一啟動觀測UTC2026-10-11T01:22:56.149554+00:00：runner48008 RUNNING，18/792 success＋1 in-flight intent、actualmodel18筆均gpt-5.6-terra，medium4096/zero retry。Harmfulness仍未知，不能稱完整792或已低於Mixing。

額度2/50 reserved（1complete、1running、48尚可保留），前一β1=.5/LR1e-5已驗H93/300=31%只SHA引用。下一routine不得早於UTC2026-10-11T02:19:03.502771+00:00；先claim，再只查此localJudge一次，completedGPU/development不再查。H≤12/300或50配額用盡即PAUSE搜尋。Native ACTIVE/hourly/failed_runs_only、188字元prompt與schedule未改；50row人類盲審表新建private但未人工判讀，gate insufficient_evidence／pairedclusters／matchingmetrics／原NPO与autograd限制保留。詳見[完成與Judge啟動證據](../../../results/reproduction/safety/spf-npo-gb200-20261006/spf-lr5e6-complete-judge-start-20261011-0917.json)。以下舊進度均為歷史。


2026-10-10 20:07 heartbeat最新：本小時claim成功後，唯一原RunAI CLI/workspace/require_escalated snapshot嘗試actual CLIexit1／attempt1，明確「Authentication failed. the token has expired」。原native字串joinLF UTF8 failedcapture84bytes private保存，並非原stdout/stderr streambytes；authfail不觸發transportretry，完整snapshot不存在。沒有第二progress／外層retry／remote mutation／GPUsubmission／Judge/keyread，也沒有改route/network。

新LR5e-6候選queue10758在前輪UTC11:08:39.332991已單次提交，此為最後已驗submission；本輪process/train/eval/audit/H均未知，不能稱RUNNING、失敗或完成。前輪19:05完整snapshot為submission前證據，不冒充currentprogress。請使用者重新runai login；下一符合一小時間隔的原排程再查一次，不盲目重送10758或1905/r2 helpers，不重訓任何completed checkpoint。

新增Judge額度仍1/50 COMPLETE、49尚可保留，已驗前一β1=.5/LR1e-5 H93/300=31%高於Mixing13/300，結果只SHA引用、未重審。新candidate完整development/audit後才先記帳並Judge。Native ACTIVE/hourly/failed_runs_only與短prompt未改；下一routine不得早於UTC13:07:31.981596（Taipei21:07:31），提前直接skip／不輪詢。研究gate、adjudication、pairedclusters與其他原限制保留。詳見[本輪登入失效證據](../../../results/reproduction/safety/spf-npo-gb200-20261006/spf-lr5e6-auth-expired-20261010-2007.json)。以下先前進度保留為歷史。


2026-10-10 19:05 heartbeat最新：RunAI完整snapshot UTC11:06:10.757057、SHAaf300bfac39afe7b3c175db7f1daa685eda5d3303433a06014bcab015d710704。前一β1=.5候選9069已退出、GPU51MiB0%／compute空，planned LR5e-6新root/source/launch/run皆不存在；completed raw/16GB權重未重取或重hash。

新候選僅LR1e-5→5e-6，保留β1=.5及原MetaM0、資料、training-only anchor、SPF算術／rank20／precision、5epochs625updates20k與原六development。3newsource＋plan＋bound snapshot共5檔sealed逐size/SHA核，bundle59fb691502f88addeaa48eb50d4b12558fa9cfcc9f247477ccd5372178efc141／7chunks。首次readonly preflight WebSocket EOF保存；同原CLI必要artifactcheck確認全部newroot/files/chunks/launch/run absent，明確1905-r2只接7missingchunks/seal/launch共9stage單次return0，不盲目重送、不改route或重傳completed模型/assets。

UTC11:08:39.332991在snapshot後148.575934秒與liveGPU/PID/source guards後，單次提交queue PID10758。唯一remote validation-code/beta05-lr5e6-full-launch-20261010.json及同名.raw.log，run /data/spf-beta05-lr5e6-full-20261010-r1。狀態SUBMITTED_RESULT_NOT_CHECKED；沒有提交後progress query，不能稱目前RUNNING或新train/eval pass。Queue預定train→independentreload→newcandidate byteaudit→六development，自動failstop／zero retry。

額度仍1/50 COMPLETE、49尚可保留，本輪Judge/API/keyread0；前一actualH93/300=31%已核結果SHA引用、未重審，仍高於Mixing13/300。新candidate未reserve Judge、H未知；actual完整development/audit後才先記帳再Judge。Native ACTIVE/hourly/failed_runs_only短prompt未改，下一routine不得早於UTC12:05:42.912148（Taipei20:05:42），提前skip、不輪詢或重啟10758/1905/r2。Gate/pairedclusters/adjudication/matching metrics/NPO/autograd限制保留。詳見[單次提交與部署證據](../../../results/reproduction/safety/spf-npo-gb200-20261006/spf-lr5e6-queue-launch-20261010-1905.json)。以下舊進度保留為歷史。


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
