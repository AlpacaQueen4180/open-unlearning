# Benign controls：clip1最終免費結果 v3

2026-10-30（Asia/Taipei）。兩個另授權的clip1 seed0控制均完成訓練、step0/18/35/53/70保存、完整291張量byte audit與fresh-process reload，以及原profile全部免費評估與固定生成。兩個engine／optimizer clipping實際均為1，逐次原生縮放通過；reload logit最大誤差皆0。原clip0、Mixing、NPO與matching reference均保留、未重跑。Judge仍未付費。

## 七模型可比結果

F/R為原forget200／retain3800，原ID、順序、資料revision與hash不變。以下EM、ROUGE-L recall、平均逐答token幾何機率列百分比；機率不是整段答案序列機率。七組MMLU皆57科14042題5-shot、原wrapper有效batch1；本輪另與原三模型逐題核對request digest，並與clip0已驗收digest相同，因此七組逐題請求一致。未改為batch8 MMLU。

| 模型 | MMLU % | EM F / R % | ROUGE F / R % | 平均token機率 F / R % |
|---|---|---|---|---|
| Mixing-full | 65.653 | 98.000 / 97.500 | 99.630 / 99.363 | 99.592 / 99.445 |
| PureNPO seed0 | 60.654 | 0.000 / 0.000 | 14.871 / 16.256 | 0.093 / 0.378 |
| NPO+retain seed0 | 65.126 | 0.000 / 14.395 | 26.099 / 64.358 | 5.290 / 76.138 |
| matched-benign-sft-s0 (clip0) | 65.055 | 100.000 / 64.000 | 100.000 / 88.698 | 99.954 / 94.436 |
| retain-continued-s0 (clip0) | 65.788 | 78.000 / 80.263 | 92.296 / 93.798 | 96.238 / 96.829 |
| matched-benign-sft-clip1-s0 | 64.991 | 100.000 / 56.289 | 100.000 / 85.528 | 99.979 / 92.151 |
| retain-continued-clip1-s0 | 65.803 | 76.500 / 81.342 | 93.899 / 94.766 | 96.931 / 96.943 |

正向CE更新本身會伴隨能力變化。clip1 SFT的forget EM100%，retain EM卻由Mixing97.5%降到56.289%；retain-only也使forget／retain EM到76.5%／81.342%。clipping1未消除SFT retain損失：其retain EM比clip0 SFT64%再低7.711百分點，而retain-only比clip0的80.263%高1.079百分點。EM對文字與標點敏感，因此同列ROUGE及機率，不能把EM下降直接當成完全不會回答。

clip1 SFT MMLU為9126/14042，較Mixing9219少93題（-0.662百分點）；retain-only9240題，多21題（+0.150百分點）。相較clip0，SFT少9題、retain-only多2題。這些是單次描述性差異，不證明能力等效或提升，也未對單seed結果作完整因果宣稱。

在有效clipping0相同的原NPO／clip0 controls中，PureNPO有更大的MMLU與TOFU退化；NPO+retain的MMLU變化可接近一般SFT，但forget清除及retain損失模式不同。因此「所有變化都只是一般更新」不能完整描述這些觀察；同時「任何變化都由遺忘目標獨有造成」也不成立。新clip1對照與原NPO另有clipping差異，不能僅從NPO0對control1的差異歸因於目標；保留clip0比較正是為了揭露此點。

新控制安全變化仍未回答：HEx300與MT80題160turn已完整生成，但沒有完整新Judge標籤，不能宣稱harmfulness改善或惡化，也不能完整判斷對話品質。舊PureNPO較低的H伴隨100%歷史degeneration，舊NPO+retain也伴隨大量degeneration及retain下降，不能把這些H差異單獨解讀成安全成功。

## Matching reference與既有native缺陷

每新模型4868序列、156921 assistant tokens，mask hash與原CPU驗證一致；新增217筆RA/WF生成、重用400筆對應retain生成。matching retain95 reference身份及原始收據hash匹配，沒有重訓或重推。遠端原始完整算術audit通過，本機另重算全部逐筆truth ratio、options probability、MU及KS；retain ROUGE沿用完整遠端原算術audit。

| 模型 | KS D | FQ p | native MU | corrected diagnostic MU |
|---|---|---|---|---|
| Mixing-full | 0.390 | 5.8673e-14 | 3.37520379e-08 | 0.645650803 |
| PureNPO seed0 | 0.065 | 0.793362 | 1.60674148e-06 | 0.029603001 |
| NPO+retain seed0 | 0.075 | 0.628431 | 1.17571096e-08 | 0.645441927 |
| matched-benign-sft-s0 (clip0) | 0.325 | 9.59424e-10 | 3.18552553e-08 | 0.619117862 |
| retain-continued-s0 (clip0) | 0.360 | 6.56896e-12 | 2.8198003e-08 | 0.656636857 |
| matched-benign-sft-clip1-s0 | 0.370 | 1.42757e-12 | 4.33003909e-08 | 0.6136133 |
| retain-continued-clip1-s0 | 0.375 | 6.54087e-13 | 3.02239454e-08 | 0.645063413 |

native retainTR的correct／wrong皆錯接PARA，原值完整保留；corrected只另外把wrong接至PERT作diagnostic，沒有覆寫舊欄或聲稱官方benchmark已修復。舊matching欄保持原接受精度。retain400子集合的MU與全retain3800的EM並非同一分母。p>.05不代表retraining-equivalent unlearning，p也不是安全指標。

## 訓練、驗證、保存及差異

兩個run皆獨立由原Mixing起跑，step0完整權重SHA256皆為92bc85a29a772eeeb5831c55bbb9a04da3d9b61faae7550434610e9f74bf2de7。A實際2000 forget曝光／200 unique、input188130／assistant71650 tokens；B2000 retain曝光／1532 unique、183198／69577 tokens。兩者完整2000個索引順序均與原NPO seed0的對應支路逐項一致。B不是3800筆跑10個完整epochs，也不是從M0重訓的retain-only oracle。

LR1e-5、WD.01（沿原no-decay group0）、paged_adamw_32bit、betas(.9,.999)、eps1e-8、有效clipping1、micro4/GAS8/world1、BF16/FA2/ZeRO3/gradient checkpointing、兩種offload皆none、max_length512、原chat template與assistant-only有效EOS/PAD128009 mask保持。warmup6、linear70；TF5.5.4實際warmup_ratio=None以免ratio0覆蓋steps，逐次LR已驗。10個50micro週期，每週期6個full8＋tail2，full除8／tail除2且scale_wrt_gas=False，沒有重複縮放。五點對應micro0/132/250/382/500、曝光0/528/1000/1528/2000。

A train wall552.865s、B563.097s，包含保存、固定probe與clipping觀測；Torch allocated峰值均約65.737GiB，與NVML總量不同。原NPO兩資料支路合計371328 nonpadding input／141227 assistant tokens，另有frozen-reference forward；不宣稱tokens、padding compute、GPU成本、梯度大小或參數位移匹配，參數位移未量測。

CPU4000筆資料、mask、CE autograd方向、sampling及scheduler通過；原生CPU clipping將norm5縮到約1，clip0負對照仍5。兩個成功smoke各50micro／7updates，在非零LR update2與tail update7驗全部291梯度有限與FP32參數真變，並通過byte audit／fresh reload。SFT full update6與tail7原生clip將norm約1.972與15縮至約1；小於1者保持原norm。正式每一步均核engine與optimizer實際1，以及每個subgroup的原生縮放固定位置值，不另施一次clipping；完整norm旁路量測在smoke7步與正式update2/7進行。

初始smoke-v1的浮點linspace觀測索引在大張量端點越界，於第一個optimizer更新完成前失敗，沒有正式run開始。保留失敗source／stderr；只改確切整數索引，9個邊界測試通過。CPU函式AST未改，按receipt hash重用；新smokes-v2各自從原Mixing重做後才啟動兩個正式run。傳輸unknown均先查片段hash、intent、真PID建立／子程序與產物，僅補已證明缺少階段。沒有改網路、實驗超參數、其他session或OLMo/SPF/Mixing排程／ledger／來源／額度。

固定teacher F8/R8涵蓋五點，HEx前8／MT前4題兩turn小probe涵蓋中間點；final用完整生成相同prefix。兩個step0的prefix輸出與generation config皆與原Mixing逐項8/8相同，只對該prefix主張。未依分數選樣、調參，也未對所有中間點加全套昂貴評估。所有原始逐次loss／LR／gradient norm／wall／GPU memory／examples／tokens／coverage與checkpoint hash均私存。

## 可重現的免費輸出診斷

兩個新模型TOFU4000均無空回覆；SFT無hit-cap，retain-only僅retain有1/3800。兩者F的重複4gram回答皆1/200，R皆5/3800。

| 模型 | HEx / MT平均實際tokens | HEx / MT hit-cap | HEx / MT有重複4gram回答數 |
|---|---|---|---|
| matched-benign-sft-clip1-s0 | 44.233 / 86.594 | 4/300 / 0/160 | 20 / 33 |
| retain-continued-clip1-s0 | 45.230 / 74.669 | 1/300 / 0/160 | 21 / 27 |

HEx／MT皆無空回覆。tokens由原generate旁路直接記錄含EOS長度，原tensor原樣返回，不改RNG／decoding；hit-cap為未EOS且達原上限512／1024。重複指標為小寫whitespace4grams的多餘重複次數／所有4gram數，不是Judge degeneration標籤。這些診斷不顯示PureNPO那種普遍hit-cap崩潰，但不能取代語義品質或安全Judge。

## 未付費Judge提案

兩個clip1 final共920候選row，379筆可重用已接受exact payload（HEx350／MT29）；未快取541筆row中有14筆跨模型重複，所以真正新distinct首次請求527＝HEx237＋MT290。本次歷史unknown exact hit為0，baseline新增0；success／unknown均不自動重送。

重用完全匹配Judge model/profile/rubric/schema/input，不只比對答案文字。歷史full judgment含理由／confidence差異的重用row有270，其中離散label／score分歧4 row、2 distinct payload。所有替代結果與事前固定earliest accepted政策均保留供人工審閱，沒有自動付費重評；本報告未匯總新H/D或MT分數。

原Judge保持gpt-5.6-terra、medium、max_output4096、SDK retry0與原rubric/schema。2026-10-30使用OpenAI Docs流程搜尋並實際開啟[官方同model頁](https://developers.openai.com/api/docs/models/gpt-5.6-terra)核對：標準短context input USD2/M、cached .20/M、output12/M，cache-write1.25倍；提案保守input按2.50/M且不假設折扣，不切換model或Fast/Batch/Flex。

實際新payload UTF8共1,434,284 bytes，加每請求4096 framing reserve，規劃input 3,592,876 tokens；最大output 2,158,592 tokens，模型費用保守估計USD34.885294。**審閱提案為最多527筆首次請求、hard cap USD35；批准USD0／request0、API calls0、Judge key reads0。** 這是規劃reserve而非精確server token count，日後付費前仍需精確reservation／當時費率與另一次明確人類USD及request授權，超cap不得發送。

新scope獨立帳務設計已封存，但未建立/占用舊ledger或移用OLMo USD150、SPF或closed campaign額度；clip0的525/USD35提案不是clip1授權。每筆先保留最大成本、unknown billing不當0且不釋放reservation，success/unknown不自動重送。本輪沒有API client或key操作。

## 交付與限制

缺項僅為新控制完整Judge標籤與分數、額外seeds、多ratio及matched tokens/compute/displacement；沒有以其他版本代填。單construction/training seed0的matched-update與clipping對照只提供機制線索，不能完整因果證明。已完成所有授權訓練、免費評估、固定生成、身份/CPU算術核對及未付費提案。

[七模型比較](../../../results/reproduction/safety/benign-controls-clip1-20261030-v1/results-v3/comparison.json)、[實際registry](../../../results/reproduction/safety/benign-controls-clip1-20261030-v1/results-v3/run-registry.json)、[免費aggregate](../../../results/reproduction/safety/benign-controls-clip1-20261030-v1/results-v3/free-aggregate.json)、[matching native／diagnostic](../../../results/reproduction/safety/benign-controls-clip1-20261030-v1/results-v3/matching-aggregate.json)、[固定probe](../../../results/reproduction/safety/benign-controls-clip1-20261030-v1/results-v3/fixed-probe-aggregate.json)、[Judge提案](../../../results/reproduction/safety/benign-controls-clip1-20261030-v1/results-v3/judge-proposal.json)、[provenance](../../../results/reproduction/safety/benign-controls-clip1-20261030-v1/results-v3/provenance.json)。完整訓練設定/實際source及smoke證據另見[執行v2](benign-controls-clip1-execution-20261030-v2.md)。

原始權重與所有checkpoint、QA／ID順序、raw／逐筆labels及理由、CLI／憑證、intent／真PID／子程序、失敗與完成收據均private；公開僅設定、aggregate與hash。finite parent及20個免費stage均自然成功結束，最後GPU idle；已完成模型不重跑。本聊天每30分鐘排程已於交付完成後暫停，不恢復其他暫停任務。不發布GitHub。
