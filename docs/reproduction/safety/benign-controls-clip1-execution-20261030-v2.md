# Clip1 controls：實際執行 v2（2026-10-30）

兩個另授權的seed0控制均已完成70 updates／500 microsteps／2000曝光，保存step0／18／35／53／70，完整byte audit與fresh-process reload均通過。**兩個實際DeepSpeed engine與optimizer clipping值皆為1，並逐次驗證原生縮放，不再只核名義設定。** 同一finite parent已完成全部免費評估；數值與未付費Judge提案另於最終結果報告封存。舊clip0與原Mixing/NPO/reference均保留，未重跑。

CPU真實4000筆資料／mask／CE autograd／scheduler，以及原生clipping norm5→約1通過；完整mask hash與原clip0完全相同。兩個smoke的非零LR update2／tail update7，全部291梯度有限、FP32參數實際改變；byte audit及native fresh-process reload皆通過且logit error=0。SFT smoke在full update6與tail update7分別將preclip norm約1.972及15縮至約1；低於1的梯度保持原norm。

原第一個smoke因觀測索引使用浮點linspace而在大張量端點越界，於第一個optimizer更新完成前失敗，正式run尚未開始。失敗及source-v1完整保留；修復僅改成確切整數位置並通過9個邊界測試，未改CE或再次施加clipping。CPU函式AST不變，因此按完整receipt hash重用；新smokes-v2各自從原Mixing重新起跑，通過後才啟動兩個正式run。

| run | updates/micro | exposures/unique | input/assistant tokens | wall含保存(s) | reload max error |
|---|---|---|---|---|---|
| matched-benign-sft-clip1-s0 | 70/500 | 2000/200 | 188130/71650 | 552.865 | 0.0 |
| retain-continued-clip1-s0 | 70/500 | 2000/1532 | 183198/69577 | 563.097 | 0.0 |

兩個step0權重SHA256與原Mixing完全相同，皆非clip0／另一control／NPO／smoke續訓。匹配原full8/tail2窗口、warmup6／linear70與相應支路抽樣策略；B並非3800筆10個完整epochs或M0 retain-only oracle。名義warmup_ratio0在固定TF5.5.4會覆蓋steps，因此實際None以維持已驗證warmup6。原NPO有效clipping0，新的clip1對照與其有此額外實作差異，後續會同時並列已完成clip0結果，不能把所有差別歸因於目標。tokens、GPU成本、梯度與參數位移不保證一致。

正式train wall包含保存、probe與clipping觀測；postclip量測在smoke全7步、正式update2/7進行，其他更新仍保留逐組固定位置原生縮放檢查。Torch allocated peak與NVML是不同統計。沒有修改optimizer超參數、LR、資料比例、建構seed，沒有projection或anchor。

免費評估沿原profile：TOFU4000與matching-reference；MMLU57科14042題5-shot、原wrapper有效batch1；HEx300／MT80題160turn固定生成。只有事前固定的小型中間probe。native retainTR/PARA缺陷保持原值，corrected diagnostic獨立列報；p>.05不當成retraining-equivalent。新模型沒有完整Judge標籤，不能宣稱harmfulness或完整對話品質結果。

[實際registry](../../../results/reproduction/safety/benign-controls-clip1-20261030-v1/execution-v2/run-registry.json)、[訓練aggregate](../../../results/reproduction/safety/benign-controls-clip1-20261030-v1/execution-v2/training-aggregate.json)、[smoke證據](../../../results/reproduction/safety/benign-controls-clip1-20261030-v1/execution-v2/smoke-aggregate.json)、[實際版本／來源hash](../../../results/reproduction/safety/benign-controls-clip1-20261030-v1/execution-v2/runtime-and-source-identities.json)。權重、QA、raw生成／逐筆標籤理由、CLI、intent/PID/子程序、完整trace與原始失敗只存private。Judge key reads／API calls／USD批准皆0，沒有動OLMo／SPF帳務或其他session／排程，也未發布GitHub。
