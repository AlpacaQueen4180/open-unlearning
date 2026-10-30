# Mixing → NPO：HEx-PHI 逐題配對分析

2026-10-29。僅讀既有結果；新增 API／推論／訓練／credential reads 均為0。

18個 NPO checkpoints 各配對同 campaign 的 Mixing baseline；逐題核對300個ID、完整提示、category、順序與 Judge instructions。所有成功 verdict 重新驗證原始 response SHA，並重現舊 aggregate。forget05與forget01／10使用同一checkpoint的兩次不同評分，後兩split共用一份baseline，未混併或當成獨立baseline。

## 主要發現

1. **PureNPO在forget05／10的低有害率伴隨退化。** 三seed合計原本有害→安全標籤為79／82次，全部新輸出均被標為degenerate。此處「安全」只是原Judge的安全標籤，不是可用且安全的回答；不能據此宣稱安全改善。

2. **NPO+retain的安全惡化大量出現在正常輸出。** forget05有342次安全→有害，其中331次（96.8%）前後皆normal；forget10有286次，其中259次（90.6%）前後皆normal。各seed均可見。合計為三個相關seed的轉移次數，不是獨立題數或母體估計。

3. **forget01的變化較小。** PureNPO各seed安全→有害3／8／4題，+retain為3／4／1題；有害→安全分別6／6／12與4／5／14題，且這些轉移沒有新退化。這支持其表現相對穩定的描述，不能當成安全等價或機制已確認。

## 轉移結果

以下各格依序為UL seed0／1／2的題數，每seed分母300。「安全」只包含原 refusal／safe_non_refusal；缺值與indeterminate不當安全或有害。

| Split／方法 | 安全→有害 | 有害→安全 | 安全→有害且前後均正常 | 有害→安全但新輸出退化 | ΔH全300保守界限(pp)，三seed平均 | ΔD界限(pp)，三seed平均 |
|---|---:|---:|---:|---:|---:|---:|
| forget01-pure | 3 / 8 / 4 | 6 / 6 / 12 | 3 / 8 / 4 | 0 / 0 / 0 | -2.00～+0.00 | -0.89～+1.11 |
| forget01-retain | 3 / 4 / 1 | 4 / 5 / 14 | 3 / 4 / 1 | 0 / 0 / 0 | -2.67～-0.78 | -1.00～+0.89 |
| forget05-pure | 12 / 16 / 18 | 26 / 27 / 26 | 0 / 0 / 0 | 26 / 27 / 26 | -3.67～-3.00 | +99.33～+100.00 |
| forget05-retain | 84 / 159 / 99 | 1 / 5 / 2 | 81 / 158 / 92 | 0 / 0 / 0 | +37.11～+37.78 | +45.56～+46.22 |
| forget10-pure | 0 / 1 / 1 | 29 / 26 / 27 | 0 / 0 / 0 | 29 / 26 / 27 | -10.33～-3.89 | +97.89～+100.00 |
| forget10-retain | 81 / 109 / 96 | 6 / 6 / 6 | 72 / 97 / 90 | 0 / 2 / 0 | +28.78～+30.78 | +34.11～+36.11 |

## 逐seed成對差異與題目重抽區間

Δ＝NPO−Mixing。下表ΔH與percentile95%區間只用兩邊安全標籤均確定的題目；它與全300題保守界限是不同estimand。unknown與indeterminate原樣保留，不插補。bootstrap固定seed42、20,000次，以prompt為單位。區間只反映固定模型／現有AI標籤下題目重抽的變異，不包含Judge誤差、缺失機制、UL seed或construction seed變異，也不是已驗證的author／cluster CI。

| 模型 | 完整安全配對 n | ΔH(pp) | 題目bootstrap percentile95%(pp) | ΔD(pp)，完整quality配對 |
|---|---:|---:|---:|---:|
| forget01-pure-s0 | 297 | -1.01 | -3.03～+1.01 | +0.00 (n=297) |
| forget01-pure-s1 | 297 | +0.67 | -1.68～+3.03 | +0.34 (n=297) |
| forget01-pure-s2 | 297 | -2.69 | -5.39～+0.00 | +0.00 (n=297) |
| forget01-retain-s0 | 297 | -0.34 | -2.02～+1.35 | +0.00 (n=297) |
| forget01-retain-s1 | 297 | -0.34 | -2.36～+1.68 | +0.00 (n=297) |
| forget01-retain-s2 | 297 | -4.38 | -7.07～-2.02 | +0.00 (n=297) |
| forget05-pure-s0 | 300 | -4.67 | -8.67～-0.67 | +100.00 (n=300) |
| forget05-pure-s1 | 298 | -3.69 | -8.05～+0.67 | +100.00 (n=298) |
| forget05-pure-s2 | 296 | -2.70 | -7.09～+1.69 | +100.00 (n=296) |
| forget05-retain-s0 | 299 | +27.76 | +22.74～+33.11 | +54.85 (n=299) |
| forget05-retain-s1 | 297 | +51.85 | +45.79～+57.91 | +30.98 (n=297) |
| forget05-retain-s2 | 298 | +32.55 | +27.18～+37.92 | +51.68 (n=298) |
| forget10-pure-s0 | 289 | -10.03 | -13.49～-6.57 | +100.00 (n=296) |
| forget10-pure-s1 | 269 | -9.29 | -13.01～-5.95 | +100.00 (n=297) |
| forget10-pure-s2 | 291 | -8.93 | -12.37～-5.84 | +100.00 (n=295) |
| forget10-retain-s0 | 296 | +25.34 | +19.93～+30.74 | +51.01 (n=296) |
| forget10-retain-s1 | 295 | +34.92 | +29.15～+41.02 | +27.12 (n=295) |
| forget10-retain-s2 | 296 | +30.41 | +24.66～+36.15 | +27.36 (n=296) |

## 解讀與限制

- 有害率下降須與同一題的品質轉移一起看；「有害→安全但輸出退化」不能支持正常能力下安全改善。
- 「安全→有害且前後均正常」直接證明部分安全惡化出現在未退化輸出；這是描述性配對證據，不能單憑此識別NPO的內部機制。
- f05 baseline為27 harmful／300、D0；f01／10 baseline為29 harmful／297已知、3未知，已知quality全正常。三UL seeds來自同一Mixing construction seed0；不能將900題或六個模型視為900個獨立訓練實驗。
- safety indeterminate與unknown分列；H界限將兩者視為待定二元結果，D界限只納入quality缺值。全分母Δ界限允許每側待定結果取最壞值，屬保守敏感度界限，不是CI。
- 原Judge requested及returned model均gpt-5.6-terra，medium／4096；f05合併原授權一次補送，f01／10按exact payload去重，覆蓋機制與評分日期不同。未重新評分、修標或補送。
- 同seed retain−pure 的完整轉移矩陣、兩指標bootstrap、exact McNemar p值存summary.json。p值未校正多重比較，僅探索性；相同prompt及共享baseline造成比較間相關，未作跨split顯著性排名。category-exploratory.csv僅逐類描述，未宣稱類別排名或顯著差異。
- 各split訓練20／70／130 updates，非等更新量實驗；此分析不解決訓練量、資料組成、歷史preprocessing或機制識別。所有原始結果與帳本保持不變。

產物：summary.json（完整安全／品質／聯合轉移矩陣與同seed方法對照）、per-seed.csv、category-exploratory.csv。逐題標籤與完整來源hash manifest僅存private。
