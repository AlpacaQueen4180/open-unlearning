# GB200訓練與評估速度診斷：公開摘要

整理：2026-10-19（Asia/Taipei）；量測與驗收完成於2026-10-18。

最具體的改善方向是把評估的有效batch設定明確傳入，以及在PureNPO的alpha=0路徑省略零權重retain分支。本次同步取樣沒有顯示RAM耗盡或可見CPU配額節流；GPU低活動與逐筆／小kernel工作型態一致。這份診斷使用獨立有界測試，原forget05正式結果與freeze保持。

| 項目 | 原版實測 | 候選實測 | 幅度 | 數值與適用限制 |
|---|---:|---:|---:|---|
| PureNPO alpha=0，同39個microstep | 86.086s | 55.341s | 時間減35.71%／1.556倍 | 省略零權重retain；非完整10epoch保證，alpha1不適用 |
| MMLU固定57科×4題，912選項requests | batch1 11.070s | batch8 3.993s | 2.773倍 | 228題3次choice不變，但LL最大差.375；非全14,042題等價 |
| 固定8題greedy生成、各64tokens | batch1 36.58 tokens/s | batch8 262.58 tokens/s | 合計吞吐7.18倍 | 2/8題tokens改變；不直接套原MT抽樣與二回合 |

訓練對照排除前8個micro及張量比較端點16、50，使用相同39個資料批次；僅一次原版→候選的有序比較，無隨機交錯或信賴區間。模型載入、驗證與正式完整291張量指紋成本不算進39筆速度；全張量端點驗證的雜湊成本另計，不拿來聲稱加速。評估時間是各3次的中位數。不同測試的倍數不能相乘，也不是GB200/H100硬體比。

## alpha=0候選的數值驗證

更新2與7的全部291張量梯度及更新後FP32參數，原始bits的SHA256全相同。
兩者各50micro／7updates，50筆loss最大差0；輸入、mask、順序、step前後及loss前後RNG一致。每端點驗全部8,030,261,248個參數，梯度在optimizer前、FP32主參數在optimizer後；每role1,164筆、兩role共2,328筆逐一核名稱、shape、dtype、元素數及摘要。更新2 LR1.6666666666666667e-6、更新7 LR1e-5，明確確認實際參數非零改變，後者涵蓋tail2。
原beta/gamma、precision、optimizer、ZeRO3、scheduler70/warmup6與full8/tail2正規化保持。這只證明同初始模型的首epoch與指定兩個端點，不能擴稱全部70updates／10epoch、其他seed或alpha1等價。候選只存在獨立診斷程式，沒有替換原正式訓練。alpha1的retain項仍有權重，應保留。

## CPU與RAM的100%怎麼解讀

同步取樣中，本程序平均約0.996個CPU core，RSS 2.68–3.15GiB，swap為0；host MemAvailable最低821.27GiB。可見cgroup約522GiB，其中filecache最低99.26%，CPU throttle／OOM與memory events增量為0。這次窗口沒有顯示程序把RAM用盡或被可見CPU配額節流。
RunAI配置記錄出allocatedCPU0.1 core、memory quantity104857600（若按bytes即100MiB），request／limit、實際usage及node／workload／pod作用域是不同量。官方文件亦分別定義CPU_USAGE_CORES與CPU_UTILIZATION、記憶體usage與allocation，未填request時可套預設且Auto limit可允許使用更多節點資源。因此100%不能直接讀成整機資源耗盡；精確UI面板分母與使用者當時時間窗尚未核實，不能把本次結論倒推到過去。[RunAI metrics](https://run-ai-docs.nvidia.com/saas/platform-management/monitor-performance/metrics)、[CPU／memory allocation](https://run-ai-docs.nvidia.com/saas/platform-management/runai-scheduler/resource-optimization/allocation-of-cpu-and-memory)。

## GPU低活動的直接證據

短trace窗口3.322s有16,850個kernels，合併kernel區間約19.92%，另外有2,241次stream同步與19次device同步。大量小kernel與間隙支持框架呼叫／同步排程有優化空間，但API呼叫本身只占部分時間，不能把全部空檔都叫Python成本。這是一個有profiler且未warmup的microstep，不是未profile訓練的精確duty，更不是SM／Tensor occupancy。
原完整首epochphase量測中forward約50.37s、backward62.97s、DeepSpeed step3.62s；mask／索引與loss讀回合計不到0.07s，沒有證據把它們當主要瓶頸。完整291張量稽核指紋未在該phase重測，不估占比。DeepSpeed參數與optimizer offload均為none；沒有UM page-fault量測，因此既不主張CPU offload／UM migration是原因，也不聲稱已排除。

## 後續採用順序

1. PureNPO alpha=0可優先以獨立新版本採用skipretain候選，保留原資料／mask／RNG與尾批算術，再針對實際後續訓練範圍驗證；alpha1保留原retain。原已完成實驗不重跑或改寫。
2. 新的MMLU評估設定應明確把batch_size傳給HFLM建構器。此次已核原wrapper漏傳，實際為1；batch8可當較快的新protocol候選，若要求重現原逐值結果則保留batch1。官方Python API亦將batch_size明列於HFLM初始化；PyTorch說明batched與逐筆運算即使數學相同也未必逐bit一致。[Harness Python API](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/python-api.md)、[PyTorch 2.7 numerical accuracy](https://docs.pytorch.org/docs/2.7/notes/numerical_accuracy.html)。
3. 生成batching適合先建立新的greedy／固定長度吞吐流程。MT原抽樣、每題RNG、停止條件與turn2 context需另行維持與驗證，不能用本次7.18倍結果直接改正式MT。
4. 儀表板記錄使用cores、RSS／anon／filecache、host available與GPU活動的同步時間窗，對齊request／limit與scope再判讀百分比；本輪未更改任何資源配置。
5. compile、ZeRO階段、microbatch／GAS、checkpointing、optimizer與precision變更仍是未驗候選；本輪沒有為追速度改loss、LR、epochs或重新搜尋超參數。

## 為何不能說GB200比舊機慢

舊H100只有版本與評估aggregate，缺相同checkpoint、token數、batch與同步計時。不同模型生成長度也差很多，文字字元不能當tokens。本次固定工作量能證明上述流程改善空間，現有資料無法計算公平的GB200/H100硬體速度比。

本輪所有GPU測試皆已有限結束、原始收據與本機audit私存。forget05原3680次首輪加22次人工補送早已有限結案，不再Judge／金鑰／API；最終表格與training/eval freezes不動。SPF automation未讀未動。


## 來源與下一步

原診斷摘要SHA256：`6c0563761e729ab64e9895495e3bd0a435f376224aaccc331c7a953a29a09562`。

抽樣數值／timing audit：`944a8f8f2d872dd7fe6c81af9f33a3e44c2948c91a335d9aa451ca3dc6bc7b1d`；全部291張量端點audit：`45469ee816106758315bea97200a4a199518705dcd8365a4f993fe52f7637d50`。

原始trace、逐題輸入輸出、tensor摘要逐筆紀錄及主機操作紀錄私存；此頁只呈aggregate。完整來源hash見[public provenance](../../../results/reproduction/safety/mixing-npo-20261019/provenance.json)。下一session實驗採用方式見[forget01／forget10 handoff](handoff-mixing-forget01-forget10-20261019.md)；原[forget05正式結果](mixing-npo-forget05-final-20261019.md)不改。
