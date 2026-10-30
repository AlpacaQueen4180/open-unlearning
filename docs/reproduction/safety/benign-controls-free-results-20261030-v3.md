# Benign controls：免費結果與交付 v3

2026-10-30（Asia/Taipei）。本頁為當前交付；v1連線失敗準備、v2執行過程及全部原失敗收據保留。只新增兩個seed0正式控制，各完成70 updates／500 microsteps、五個checkpoint、291張量完整byte audit與fresh-process reload，以及完整免費評估。沒有付費Judge或新增其他訓練。

**仍有未獲接受的protocol deviation：兩個新控制與原NPO實際DeepSpeed clipping皆為0，使用者明列的有效clipping要求是1。本輪不能標成完全符合協定。** 先前我只驗名義max_grad_norm=1，漏核DS JSON缺省值；這是驗收疏漏。進一步固定版本CPU解析及Accelerate／DeepSpeed原碼確認，該路徑依賴DS內部clipping，缺省0。原freeze不變，偏差單獨揭露。使用者尚未選擇保留這兩個實際runtime匹配控制，或另授權兩個有效clip1的替代run；沒有自行增加第三／第四個正式run。

## 可比結果與解釋

F/R為原forget05 200／retain95 3800，所有原ID、順序、revision、QA bytes與mask保持。EM、ROUGE-L recall與逐答token幾何機率均列百分比；機率不是整段答案序列機率。三個舊模型只重用已接受輸出，不做新推論。五組MMLU的逐題request digest完全相同，皆57科、14042題、5-shot、原wrapper實際batch1；名義batch16不代表有效batch16，本輪未採batch8 MMLU。

| 模型 | MMLU % | EM F / R % | ROUGE F / R % | 平均token機率 F / R % |
|---|---|---|---|---|
| Mixing-full | 65.653 | 98.000 / 97.500 | 99.630 / 99.363 | 99.592 / 99.445 |
| PureNPO seed0 | 60.654 | 0.000 / 0.000 | 14.871 / 16.256 | 0.093 / 0.378 |
| NPO+retain seed0 | 65.126 | 0.000 / 14.395 | 26.099 / 64.358 | 5.290 / 76.138 |
| matched-benign-sft-s0 | 65.055 | 100.000 / 64.000 | 100.000 / 88.698 | 99.954 / 94.436 |
| retain-continued-s0 | 65.788 | 78.000 / 80.263 | 92.296 / 93.798 | 96.238 / 96.829 |

正向SFT使forget EM到100%，retain EM由97.5%降至64%；retain-only持續訓練也使forget／retain EM降至78%／80.263%。所以一般CE參數更新本身可伴隨TOFU能力變化，不能將所有能力變化歸因於遺忘目標。EM對文字、標點敏感；ROUGE與機率一併列出，不能把EM下降直接當成完全不會回答。

SFT的MMLU較Mixing低0.598百分點（9135 vs9219題），retain-only高0.135百分點（9238題）。逐題配對，SFT新答對250／原答對轉錯334；retain-only292／273。這是描述性單次結果，不證明能力等效或提升。PureNPO的MMLU與TOFU退化遠大於兩個正向控制；NPO+retain的MMLU接近SFT，但forget清除與retain損失模式不同。只有一個construction seed與training seed，不能作完整因果證明。

安全問題仍未回答：新模型完整HEx300與MT80題／160turn生成已固定，但沒有新Judge標籤，不宣稱harmfulness改善或惡化，也不能完整判定對話品質。舊PureNPO的低H伴隨100%歷史degeneration，不能解讀成安全成功；舊NPO+retain也有高degeneration與retain下降。新控制的免費診斷未顯示同樣普遍hit-cap崩潰，這不等於安全或品質已驗收。

## Matching reference、native缺陷及診斷

沿用原4868序列、156921 assistant tokens與相同mask hash；每新模型只新增217筆RA/WF生成，重用400筆對應retain生成。原retain95 reference身份與byte hash完全匹配，沒有重訓或重推reference。遠端完整原算術audit通過；本機另重算4868筆mask／variants、逐筆truth ratio、options probability、MU及KS。retain ROUGE沿用遠端完整原算術audit，未在本機重跑模型。

| 模型 | KS D | FQ p | native MU | corrected diagnostic MU |
|---|---|---|---|---|
| Mixing-full | 0.390 | 5.8673e-14 | 3.37520379e-08 | 0.645650803 |
| PureNPO seed0 | 0.065 | 0.793362 | 1.60674148e-06 | 0.029603001 |
| NPO+retain seed0 | 0.075 | 0.628431 | 1.17571096e-08 | 0.645441927 |
| matched-benign-sft-s0 | 0.325 | 9.59424e-10 | 3.18552553e-08 | 0.619117862 |
| retain-continued-s0 | 0.360 | 6.56896e-12 | 2.8198003e-08 | 0.656636857 |

既有native retainTR將correct與wrong都接至PARA，產生不正常的極小MU；保留原native欄位，不覆寫舊值。corrected欄位僅另修wrong至PERT的diagnostic，不能直接稱官方修正benchmark。舊matching欄位沿用已接受公開精度，新欄位保留實際精度。FQ p>.05不代表retraining-equivalent unlearning，兩個控制的p更不能當成安全指標。

## 實際訓練與保存

固定實際版本為Torch2.7.0a0+ecf3bae40a.nv25.2、Transformers5.5.4、Accelerate1.13.0、DeepSpeed0.15.4、bitsandbytes0.50.2、FlashAttention2.7.3、datasets3.0.1；Torch版本以實際收據為準，不沿用舊文件nv25.02拼字。原training repository commit為17cbbc87192e6934deb92875c359c91bbd837fb4；TOFU revision為324592d84ae4f482ac7249b9285c2ecdb53e3a68。實際來源另外綁定檔案hash，commit不代表所有working tree檔案未改。[完整runtime／資料／來源身份](../../../results/reproduction/safety/benign-controls-20261029-v1/execution-v3/runtime-source-identities.json)。


兩模型step0的權重SHA256均為原Mixing `92bc85a29a772eeeb5831c55bbb9a04da3d9b61faae7550434610e9f74bf2de7`，各自獨立起跑。A正向assistant-token CE使用forget200，2000曝光／200 unique，input／assistant tokens188130／71650；B沿原retain replacement sampling，2000曝光／1532 unique，183198／69577 tokens。兩支路2000個實際索引順序均與原NPO seed0的對應支路完全一致。B不是3800筆跑10個完整epochs，也不是從M0重訓的retain-only oracle。

LR1e-5、WD.01（沿原no-decay group為0）、paged_adamw_32bit、betas(.9,.999)、eps1e-8、world1、micro4／GAS8、BF16／FA2／ZeRO3／gradient checkpointing、兩種offload均none、max_length512。原chat template、assistant-only labels及EOS=PAD128009有效mask保持。clipping名義1／有效0的偏差如上。TF5.5.4實際warmup_ratio=None，使warmup_steps6生效；若實傳ratio0會覆蓋steps，這是版本相容性差異，逐次LR已驗證為warmup6／linear70。

每週期50microsteps＝6個full8窗口＋tail2，10週期＝500microsteps／70實際optimizer更新。full除8、tail除2，DeepSpeed scale_wrt_gas=False，未雙重縮放。step0／18／35／53／70分別為microsteps0／132／250／382／500，曝光0／528／1000／1528／2000。每點保存實際token、unique、LR、loss、gradient norm、wall與記憶體trace。訓練wall含保存與probe：A558.550s、B545.350s；Torch allocated peak各約65.737GiB，與NVML總量不是同一統計。

CPU真實資料身份、mask、CE autograd方向、full/tail、70-step scheduler與唯一intent拒絕重送通過。兩個工程smoke各50micro／7updates，update2及tail update7皆有非零LR、全部291梯度有限且FP32參數實際改變，並完成byte audit／fresh reload。A smoke曾於最後診斷receipt寫入遇steps_per_print=inf，保留失敗，僅修復收據，未重訓。正式final皆291 BF16張量／32層／8,030,261,248參數有限，fresh-process reload max logit error=0、0額外更新。

固定F8/R8 teacher probe涵蓋五個保存點，HEx前8／MT前4題兩turn小probe涵蓋中間點；final使用完整評估的相同prefix。沒有用分數選樣、調LR或為所有中間點做完整昂貴評估。兩個step0的HEx8／MT8輸出及generation config與原baseline prefix逐項完全相同；只對該prefix作此主張。

原NPO兩支路合計371328 non-padding input／141227 assistant tokens，另有frozen-reference forward；A/B單側CE不匹配tokens、padding compute、GPU成本、梯度或參數位移。參數位移未量測，不聲稱相等。逐次gradient norm範圍A0.0363–13.6017、B0.3576–17.3115，均有限；它們不能當成有效clip1證據。

## 輸出診斷與缺項

TOFU新A沒有空回覆或hit-cap，B沒有空回覆、retain有1/3800 hit-cap。A HEx平均44.29 tokens、1/300 hit-cap，MT83.25、0/160；B HEx40.843、1/300，MT76.625、0/160。HEx重複whitespace4gram的回答A18/300、B21/300；MT25/160、28/160。這只是文字統計，不能當作Judge的degeneration標籤。所有精確token／EOS旁路記錄不改變原generate輸出、RNG或decoding。各模型raw、profile、逐筆診斷及hash私存，公開僅aggregate。

缺項：兩控制的H/D、完整MT Judge評分；有效clipping1的合規替代結果（尚未授權新增run）；多seed穩健性、matched tokens／compute／parameter displacement。沒有以其他模型版本代填。

## Judge提案：尚未批准、尚未執行

兩新final共920筆HEx/MT候選row：375筆可重用已接受exact-payload結果（HEx354／MT21），545筆未快取row中有20筆跨模型重複，故真正新distinct首次請求525＝HEx227＋MT298。此次payload沒有命中歷史unknown；baseline新請求0。success／unknown不自動重送，API calls=0、key reads=0、批准與支出USD0。

重用要求完整model/profile/rubric/schema/input payload hash一致。歷史多次同payload結果中，270個重用row含文字理由／confidence等差異，其中僅4個row（2個distinct payload）存在離散label或score分歧；全部替代結果及原先事前固定的earliest accepted選擇政策均保留，另列人工審閱，不自動付費重評。此頁未對新控制匯總H/D或MT分數。

原Judge為gpt-5.6-terra、reasoning medium、4096 output cap、SDK retry0、原rubric與Pydantic schema。依2026-10-29核對的[官方模型價格](https://developers.openai.com/api/docs/models/gpt-5.6-terra)與[定價文件](https://developers.openai.com/api/docs/pricing)，標準短context input USD2/M、cached .20/M、output12/M；cache-write保守reserve按2.50/M，不假設cache折扣。本次525個新payload共1,422,614 UTF8 bytes；加每請求4096 framing reserve，規劃input3,573,014 tokens，最大output2,150,400 tokens，保守模型費用估計USD34.737335，**提議hard cap USD35、首次請求最多525，並非付費批准**。這不是精確server token count；日後啟動前需核對token reservation與當時費率，超cap不得發送。

獨立scope帳務設計已保存，不建立或占用舊ledger、不挪用OLMo／SPF／已結束campaign額度；每請求先保留最大費用，unknown billing不當0且保留reservation。付費啟動需另有明確人類USD與request上限授權，模型身份／payload清單／profile均凍結；此次沒有建立API client、讀key或送請求。

## 交付與當前狀態

[實際run registry](../../../results/reproduction/safety/benign-controls-20261029-v1/execution-v3/run-registry.json)、[比較表](../../../results/reproduction/safety/benign-controls-20261029-v1/execution-v3/comparison.json)、[完整免費aggregate](../../../results/reproduction/safety/benign-controls-20261029-v1/execution-v3/free-aggregate.json)、[固定probe](../../../results/reproduction/safety/benign-controls-20261029-v1/execution-v3/fixed-probe-aggregate.json)、[Judge審閱提案](../../../results/reproduction/safety/benign-controls-20261029-v1/execution-v3/judge-proposal.json)及[provenance](../../../results/reproduction/safety/benign-controls-20261029-v1/execution-v3/provenance.json)。

權重與五點checkpoint、QA、ID順序、raw生成／逐筆標籤理由、CLI、真PID建立資料、原始成功／失敗收據及private handoff皆私存。兩個finite queues已自然完成，沒有自動排程需要恢復；其他session、OLMo controller／來源／ledger／USD150與舊SPF／Mixing排程均未修改。最後補取reference唯讀操作10/10暫時逾時，停止重試，改重用本機已接受、完全匹配hash的原reference bytes完成核算；不是缺少評估或重新訓練。沒有修改網路設定或發布GitHub。

目前可回答：一般持續訓練也造成能力變化，而NPO出現不同的遺忘／retain／退化模式；新控制安全性與完整對話品質仍待Judge。唯一訓練協定未決是有效clipping0：保留實際runtime對照，或另授權兩個clip1替代run。這個選擇會改變「只新增兩個正式run」的範圍，不能由繼續免費核算推定為批准。
