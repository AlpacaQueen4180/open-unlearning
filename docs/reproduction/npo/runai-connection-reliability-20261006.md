# RunAI 連線診斷與有限重試

2026-10-06本機CLI為 `runai-cli/2.25.28 sdk/2.114.4`。歷次遠端命令曾出現cluster-api/status EOF、TLS handshake timeout、WebSocket EOF，以及exit0但stdout為空；這些回傳不能當作遠端任務沒有執行。

## 查核結果

- 本機 `api.cluster_status_timeout_duration` 已設定20s；官方預設3s。沒有因為錯誤就盲目再延長，亦未更動尚未確認用途的其他timeout欄位。
- 控制端點一次DNS／TCP／TLS量測成功：DNS8ms、TCP421ms、TLS110ms，TLS1.3且憑證驗證通過。一次成功不代表間歇故障已消失，也不能直接歸因於VPN或平台。
- EAA、ZeroTier等虛擬介面與有線網路目前啟用。介面存在不代表已證明該流量走哪條路徑；未自行切換VPN、代理、DNS、MTU或憑證檢查。
- `--pod-running-timeout`／`--wait-timeout`是等待pod／workload就緒的timeout，不能當成TLS／WebSocket傳输timeout。CLI沒有在help中提供通用exec TLS重試參數。

官方來源：[config set](https://run-ai-docs.nvidia.com/self-hosted/reference/cli/runai/runai_config_set)、[workspace exec](https://run-ai-docs.nvidia.com/self-hosted/reference/cli/runai/runai_workspace_exec)。

## 已採用的改善

[invoke_readonly_json.ps1](../../../scripts/reproduction/npo/gb200/invoke_readonly_json.ps1)將snapshot gzip壓縮後以短命令傳送，限制URL-encoded程式長度不超過6000；太長直接拒絕，改以已持久化script路徑讀取，降低已觀測的長URL400問題。這個長度是本地保守限制，不宣稱平台公布了相同上限。隨機begin／end framing及JSON解析可檢出stdout空白／截斷，不能只看CLI exit0。只對暫時性transport failure或不完整回傳重試一次；遠端Python錯誤或完整但無效JSON不重試。

另實測官方支援的 `--stdin -- python -`，但目前Windows CLI在PowerShell pipeline直接回報 `failed to exec output. The handle is invalid.`；沒有採用此尚未可用的方式，也不宣稱stdin能修復TLS。CLI版與管線介面的相容性仍待確認。

壓縮短命令helper已完成一次實際遠端唯讀測試，回傳完整JSON且Python版本為3.12.3；原始回傳保存於 [connection-framing-check.json](../../../results/reproduction/npo/large-model-validation-20261006/connection-framing-check.json)。此測試沒有讀取或重啟訓練。排程snapshot使用此helper最多兩次嘗試，外層不得再重試，以免超過一次重試限制。

```powershell
$snapshot = @'
import json
from pathlib import Path
p=Path('/data/npo-gb200-20261004/model-validation-20261006-r1/status.json')
print(json.dumps(json.loads(p.read_text())))
'@
& ./scripts/reproduction/npo/gb200/invoke_readonly_json.ps1 -PythonCode $snapshot
```

此helper只能用於read-only snapshot，不能包覆背景啟動、提交或修改動作。啟動結果不確定時先讀獨立launch／status／PID再決定，不盲目重送。長任務仍在pod內background執行與寫PVC，桌面CLI只在排程時短暫連線；CLI回傳斷線與容器／訓練是否存活分開判定。

若仍頻繁發生TLS／WebSocket EOF，需以錯誤時間、端點與CLI版本讓平台／網路管理者檢查EAA／proxy／gateway與cluster-api路徑；目前沒有定位到特定設備，未宣稱此helper已治好平台連線。不自行升級CLI、變更全域網路或停止workload。
