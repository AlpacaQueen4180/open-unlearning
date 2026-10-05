param([switch]$Save)
$ErrorActionPreference = 'Stop'
$taskCli = 'C:\Program Files\Runai\runai.exe'
for ($taskAttempt=0; $taskAttempt -lt 2; $taskAttempt++) {
    $taskOutput = & $taskCli workspace exec machine-unlearning-pvc -p smart-mfg -- python /data/npo-gb200-20261004/repo/scripts/reproduction/npo/gb200/collect_status.py
    if ($LASTEXITCODE -eq 0) { break }
}
if ($LASTEXITCODE -ne 0) { throw 'Cannot fetch RunAI status' }
$taskJson = $taskOutput -join "`n"
$taskState = $taskJson | ConvertFrom-Json
if ($Save) {
    $taskDestination = Join-Path $PSScriptRoot '../../../../results/reproduction/npo/gb200'
    New-Item -ItemType Directory -Force -Path $taskDestination | Out-Null
    [IO.File]::WriteAllText((Join-Path $taskDestination 'status-20261004.json'), $taskJson, [Text.UTF8Encoding]::new($false))
}
$taskState.runs | ForEach-Object {
    [pscustomobject]@{Run=$_.name; Phase=$_.phase; Step=$_.trainer.global_step; FQ=$_.summary.forget_quality; MU=$_.summary.model_utility; TR=$_.summary.forget_truth_ratio}
} | Format-Table -AutoSize
$taskState.runs | Where-Object { $_.phase -in 'TRAINING','EVALUATING' } | ForEach-Object {
    $_.'training.log'
    $_.'evaluating.log'
}
