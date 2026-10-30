# Benign controls：實際執行 v2（2026-10-29）

兩個 seed0 控制均已完成 70 updates／500 microsteps、完整 checkpoint byte audit 與 fresh-process reload；免費評估仍在執行。**兩者發現有效 gradient clipping=0 的 protocol deviation，不能宣稱完全符合使用者要求的 clipping=1。** 本頁取代 v1 作為當前執行狀態；v1 原失敗、來源與收據完整保留。

## 重要勘誤：名義 clipping=1，DeepSpeed 有效值=0

先前驗收只核 `TrainingArguments.max_grad_norm=1`。進一步檢查實際 plugin 設定、固定程式版本與 CPU-only DeepSpeedConfig 解析後，兩個新 run 的有效值都是0，因為 DS JSON 缺少 gradient_clipping 欄位。Accelerate 的 DeepSpeed clip_grad_norm_ 分支只取回 norm，依賴 DeepSpeed 內部 clipping；該版本 DeepSpeed 的缺省值為0。

原 forget05 PureNPO／NPO+retain 的已綁定同版 DS 設定也缺少該欄位，解析亦為0；舊數值、來源與freeze沒有修改。這是另立的實作／配置勘誤，與既有 native retainTR/PARA 錯接缺陷分開。

目前兩個新 run 與原實際 runtime 相符，但偏離本輪明列的有效 clipping=1。已向使用者請求設計選擇：保留現有兩個、明列clip0，或另授權兩個有效clip1替代run。**尚未自行增加訓練或把設定偏差視為批准。** 若選後者，必要改動為新 DS config 明列 gradient_clipping=1.0，並在新 smoke 核 engine.gradient_clipping()／optimizer.clip_grad 實際值及非零LR/full-tail後才訓練；原兩個結果仍保留。

## 已實際完成

| run | updates / microsteps | exposures / unique | input / assistant tokens | train wall（含保存） | fresh reload max error |
|---|---|---|---|---|---|
| Matched benign SFT seed0 | 70 / 500 | 2000 / 200 | 188130 / 71650 | 558.550s | 0 |
| Retain continued seed0 | 70 / 500 | 2000 / 1532 | 183198 / 69577 | 545.350s | 0 |

兩個run均保存 step0／18／35／53／70；相應microsteps為0／132／250／382／500，曝光0／528／1000／1528／2000。兩個step0權重檔SHA256皆等於已接受原Mixing的`92bc85a29a772eeeb5831c55bbb9a04da3d9b61faae7550434610e9f74bf2de7`，不是由NPO、另一control或smoke續訓。兩個final完整291張量／32層／8,030,261,248參數皆BF16、有限，native fresh reload均無訓練optimizer、0額外updates，max logit error=0。

Torch allocated峰值均約65.737 GiB；NVML程序使用量在訓練期間另有更高觀測值，兩者不是同一記憶體統計。SFT保存／probe耗時108.247s，已包含在558.550s中；不可把這個wall當純GPU計算時間。

真實CPU gate已核全4000筆資料的有效mask／EOS=PAD=128009、原chat template與不截斷，實際CE autograd方向正確、scheduler70／warmup6正確。兩個smoke均50micro／7updates，在update2及tail update7驗全部291梯度有限、非零LR及FP32參數真實改變，並通過byte audit與fresh reload。SFT smoke最後收據曾因DS診斷值steps_per_print=inf序列化失敗；50/7更新與checkpoint已完成，修復的是收據，沒有重訓該smoke。

最初RunAI DNS錯誤已定位為沙箱與主機執行環境差異；主機環境唯讀連線成功，沒有改VPN／DNS／MTU／TLS。所有暫時連線與mutation未知收據保留；僅在唯讀證明part／intent／PID不存在後補缺。另一session的兩個暫停程序及所有OLMo／SPF／舊Mixing排程均未停止或修改。曾退役的免費queue僅是本聊天尚無子程序、尚未評估的等待程序；真PID建立時間與命令已核，GPU工作停止數0。

## 可比性與目前結論界線

兩個run的實際2000筆資料索引順序，分別完全等於原forget05 seed0 NPO的forget／retain支路順序；B的unique coverage為1532/3800，並非3800筆跑10個epochs。原NPO兩個資料支路合計371328個non-padding input tokens、141227個assistant tokens；這是由原已接受索引和相同mask推導，不含frozen reference額外forward或padding計算。A/B只有單側CE，tokens、GPU成本、gradient與參數位移都不匹配，只有更新排程匹配。B不是從M0重訓的retain-only oracle。

固定F8/R8 teacher probe已在五個保存點完成，且另有原生成profile的HEx前8／MT前4題兩turn probe，並未依分數選樣或調參。小probe不取代full4000 TOFU、57科14042題5-shot MMLU或Judge。訓練CE跨cycle也不是固定validation loss，尤其B每cycle抽樣不同。

完整免費評估目前執行中：原TOFU F/R probability、EM、ROUGE、matching-retain95 reference、原native與獨立corrected diagnostic、MMLU有效batch1、HEx300與MT80題160turn。原資料／模板／decoding／版本保持；生成旁路只記錄實際token數、EOS與hit-cap並原樣返回原張量。沒有重跑舊Mixing/NPO/reference。

新控制的能力差異待完整免費結果；安全性H/D與MT評分保持待Judge，不由未評分輸出推論harmfulness。單construction seed與training seed0、matched-update仍只是機制線索，不證明完整因果，FQ p>.05也不代表retraining-equivalent。

## 付費界線與交付位置

Judge API calls=0、key reads=0、批准額度USD0。已唯讀核對原8模型與一次人工補送的3666筆有效raw結果，形成3439個distinct成功payload cache；未修改共享ledger，也未讀OLMo資料。新final尚待完整固定生成，因此真正新請求／exact reuse清單與審閱預算稍後以實際payload凍結。上限仍為兩模型最多920筆首次請求，baseline新增預設0；success／unknown不自動重送。

原model/profile為gpt-5.6-terra／medium／4096 output cap／SDK retry0。2026-10-29已開啟核對[官方model價格頁](https://developers.openai.com/api/docs/models/gpt-5.6-terra)：標準短context每百萬tokens input USD2、cached input USD0.20、output USD12，cache write另按已核費率作保守reserve。這只是後續提案依據，不是付費授權。

[實際訓練aggregate與來源hash](../../../results/reproduction/safety/benign-controls-20261029-v1/execution-v2/training-aggregate.json)。權重、raw／QA、ID順序、逐步trace、CLI、audit/reload原bytes與付費packet均在private scope；未發布GitHub。完整免費結果與預算將在本v2報告續補，原v1及舊凍結結果不覆寫。
