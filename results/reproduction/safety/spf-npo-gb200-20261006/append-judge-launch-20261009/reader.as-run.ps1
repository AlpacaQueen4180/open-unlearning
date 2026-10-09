$ErrorActionPreference='Stop'
$taskRoot=(Get-Location).Path
$taskPrivate=Join-Path $taskRoot 'work/spf-npo-gb200-20261006/private'
$taskQueue=Join-Path $taskPrivate 'checkpoint313469-append-judge-queue-20261009'
function Read-TaskJson([string]$Path){if(Test-Path -LiteralPath $Path -PathType Leaf){ConvertFrom-Json -InputObject ([IO.File]::ReadAllText($Path))}else{$null}}
function Read-TaskLatest([string]$Directory,[string]$Key){
    $taskFiles=@(Get-ChildItem -LiteralPath (Join-Path $Directory 'state-snapshots') -File -Filter '*.json' -ErrorAction SilentlyContinue|Sort-Object Name)
    if($taskFiles.Count -eq 0){return $null}
    $taskFile=$taskFiles[-1]
    try{$taskRow=Read-TaskJson $taskFile.FullName;return @{sequence=$taskRow.sequence;state=$taskRow.$Key;sha256=(Get-FileHash -LiteralPath $taskFile.FullName -Algorithm SHA256).Hash.ToLower();file_count=$taskFiles.Count}}
    catch{return @{observation='LATEST_IMMUTABLE_FILE_NOT_YET_COMPLETE';error_type=$_.Exception.GetType().Name;sequence_file=$taskFile.Name}}
}
function Read-TaskLinesShared([string]$Path){
    $taskStream=[IO.File]::Open($Path,[IO.FileMode]::Open,[IO.FileAccess]::Read,([IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete))
    $taskReader=[IO.StreamReader]::new($taskStream,[Text.UTF8Encoding]::new($false))
    try{while(-not $taskReader.EndOfStream){$taskReader.ReadLine()}}finally{$taskReader.Dispose();$taskStream.Dispose()}
}
$taskReceipt=Read-TaskJson (Join-Path $taskQueue 'launch-receipt.private.json')
$taskResult=@{observed_at_utc=[DateTime]::UtcNow.ToString('o');queue_latest=(Read-TaskLatest $taskQueue 'queue');launch=$taskReceipt;cases=@();key_contents_read=$false;ciphertext_read=$false;api_calls_from_reader=0;runai_queries=0;new_dialogs=0}
$taskProcessIds=@($taskReceipt.pid)
foreach($taskStep in @(313,469)){
    $taskRun=Join-Path $taskPrivate ('spf-checkpoint-judge-20261008/checkpoint-'+$taskStep+'-saved-key-r2-append')
    $taskLatest=Read-TaskLatest $taskRun 'job'
    $taskJob=Read-TaskJson (Join-Path $taskRun 'job.json')
    if($taskJob){$taskProcessIds+=,$taskJob.pid}elseif($taskLatest.state){$taskProcessIds+=,$taskLatest.state.pid}
    $taskCounts=@{};$taskModels=@{};$taskRows=0
    $taskResultsPath=Join-Path $taskRun 'judge-results.jsonl'
    if(Test-Path -LiteralPath $taskResultsPath){foreach($taskLine in (Read-TaskLinesShared $taskResultsPath)){
        try{$taskRow=ConvertFrom-Json -InputObject $taskLine;$taskRows++;$taskCounts[$taskRow.status]++;$taskModels[$taskRow.actual_model]++}catch{break}
    }}
    $taskEvents=@{};$taskEventsPath=Join-Path $taskRun 'execution-events.private.jsonl'
    if(Test-Path -LiteralPath $taskEventsPath){foreach($taskLine in (Read-TaskLinesShared $taskEventsPath)){
        try{$taskEvent=ConvertFrom-Json -InputObject $taskLine;$taskEvents[$taskEvent.operation]++}catch{break}
    }}
    $taskResult.cases+=,@{checkpoint_step=$taskStep;latest=$taskLatest;terminal_job=$taskJob;result_records=$taskRows;result_statuses=$taskCounts;actual_models=$taskModels;event_operations=$taskEvents;credential_status=(Read-TaskJson (Join-Path $taskRun 'credential-input-status.private.json'));launcher_result=(Read-TaskJson (Join-Path $taskRun 'launcher-result.private.json'));safe_loader_error=(Read-TaskJson (Join-Path $taskRun 'credential-launcher-error.private.json'));audit=(Read-TaskJson (Join-Path $taskPrivate ('checkpoint-'+$taskStep+'-append-state-judge-audit-20261009/audit.private.json')))}
}
$taskFilter=($taskProcessIds|Where-Object {$_}|Sort-Object -Unique|ForEach-Object {'ProcessId='+$_}) -join ' OR '
$taskProcesses=@(Get-CimInstance Win32_Process -Filter $taskFilter|Where-Object {$_.CommandLine -like '*append*'})
$taskResult.own_live_pids=@($taskProcesses|ForEach-Object {$_.ProcessId})
$taskObservation=Join-Path $taskQueue ('observation-'+[DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss')+'.private.json')
$taskBytes=[Text.UTF8Encoding]::new($false).GetBytes(($taskResult|ConvertTo-Json -Depth 20)+"`n")
$taskStream=[IO.File]::Open($taskObservation,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
try{$taskStream.Write($taskBytes,0,$taskBytes.Length);$taskStream.Flush($true)}finally{$taskStream.Dispose()}
$taskResult|ConvertTo-Json -Depth 20 -Compress
