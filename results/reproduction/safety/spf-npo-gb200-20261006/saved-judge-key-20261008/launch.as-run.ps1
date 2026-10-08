$ErrorActionPreference='Stop'
$env:CUDA_VISIBLE_DEVICES=''
$env:OPENAI_API_KEY=$null
$env:OPENAI_ORG_ID=$null
$env:OPENAI_PROJECT_ID=$null
$taskPrivate=Join-Path $PSScriptRoot 'private'
$taskOld=Join-Path $taskPrivate 'spf-checkpoint-judge-20261008/checkpoint-157-r1'
$taskRun=Join-Path $taskPrivate 'spf-checkpoint-judge-20261008/checkpoint-157-saved-key-r1'
$taskPlanPath=Join-Path $taskRun 'plan.private.json'
$taskPlan=ConvertFrom-Json -InputObject ([IO.File]::ReadAllText($taskPlanPath))
$taskStorageChecks=ConvertFrom-Json -InputObject ([IO.File]::ReadAllText((Join-Path $taskPrivate 'saved-key-actual-user-checks-20261008/checks.private.json')))
$taskGuardChecks=ConvertFrom-Json -InputObject ([IO.File]::ReadAllText((Join-Path $taskPrivate 'saved-key-judge-guards-20261008/checks.private.json')))
if($taskStorageChecks.status -ne 'WINDOWS_CURRENTUSER_DPAPI_AND_PRIVATE_ACL_TEST_PASS' -or -not $taskStorageChecks.actual_authorized_user -or $taskGuardChecks.status -ne 'SAVED_KEY_NEW_PROFILE_AND_EXISTING_BUDGET_GUARDS_PASS'){throw 'Storage and new-profile checks required'}
if([Security.Principal.WindowsIdentity]::GetCurrent().User.Value -ne $taskPlan.credential_windows_sid){throw 'Actual authorized Windows account required'}
foreach($taskName in @('job.json','execution.lock','raw-responses','execution-events.private.jsonl','credential-input-status.private.json','saved-key-launch-receipt.private.json','saved-key-helper.stdout.txt','saved-key-helper.stderr.txt')){if(Test-Path -LiteralPath (Join-Path $taskRun $taskName)){throw 'Saved-key execution already exists'}}
foreach($taskName in @('job.json','execution.lock','raw-responses','execution-events.private.jsonl')){if(Test-Path -LiteralPath (Join-Path $taskOld $taskName)){throw 'Original Judge may have sent requests; do not replace or repeat'}}
if(Test-Path -LiteralPath (Join-Path $taskOld '../allocations/157.json')){throw 'Original checkpoint already allocated'}
$taskOldStatus=ConvertFrom-Json -InputObject ([IO.File]::ReadAllText((Join-Path $taskOld 'credential-input-status.private.json')))
if($taskOldStatus.status -ne 'WAITING_FOR_NEW_KEY_LOCAL_MASKED_DIALOG' -and $taskOldStatus.status -ne 'INPUT_CLOSED_WITHOUT_KEY'){throw 'Original UI not an unsent key dialog'}
$taskProcess=Get-CimInstance Win32_Process -Filter 'ProcessId=63928'
if($taskProcess){
    $taskExpected=Join-Path $taskOld 'launch-dialog.as-run.ps1'
    if(-not $taskProcess.CommandLine.Contains($taskExpected)){throw 'Original PID identity changed'}
    if($taskOldStatus.status -ne 'WAITING_FOR_NEW_KEY_LOCAL_MASKED_DIALOG'){throw 'Original process not waiting'}
    $taskNativeProcess=[Diagnostics.Process]::GetProcessById(63928)
    if(-not $taskNativeProcess.CloseMainWindow()){throw 'Cannot close own waiting dialog safely'}
    if(-not $taskNativeProcess.WaitForExit(5000)){throw 'Original UI did not exit; no new launch'}
}
foreach($taskName in @('job.json','execution.lock','raw-responses','execution-events.private.jsonl')){if(Test-Path -LiteralPath (Join-Path $taskOld $taskName)){throw 'Original execution evidence appeared; no new launch'}}
if(Test-Path -LiteralPath (Join-Path $taskOld '../allocations/157.json')){throw 'Original allocation appeared; no new launch'}
$taskOldStatus=ConvertFrom-Json -InputObject ([IO.File]::ReadAllText((Join-Path $taskOld 'credential-input-status.private.json')))
if($taskOldStatus.status -ne 'INPUT_CLOSED_WITHOUT_KEY'){throw 'Original dialog was not closed without API'}
$taskRetirement=@{status='ORIGINAL_MEMORY_ONLY_UI_RETIRED_WITHOUT_KEY_OR_API';original_pid=63928;original_run=$taskOld;observed_at_utc=[DateTime]::UtcNow.ToString('o');original_api0=$true;user_requested_saved_key=$true}
if(-not (Test-Path -LiteralPath (Join-Path $taskRun 'old-dialog-retirement.private.json'))){throw 'Original API0 retirement evidence required'}
[IO.File]::WriteAllText((Join-Path $taskRun 'old-dialog-retirement-reused-r2.private.json'),($taskRetirement|ConvertTo-Json),[Text.UTF8Encoding]::new($false))
$taskRunner=Join-Path $taskRun 'run_spf_checkpoint_development_judge_saved_key.py'
if((Get-FileHash -LiteralPath $taskRunner -Algorithm SHA256).Hash.ToLower() -ne $taskPlan.runner_sha256 -or $taskPlan.runner_sha256 -ne $taskGuardChecks.source_sha256){throw 'Tested source changed'}
& $taskPlan.python_executable -X utf8 $taskRunner --run $taskRun --source-sha256 $taskPlan.runner_sha256 --plan-sha256 ((Get-FileHash -LiteralPath $taskPlanPath -Algorithm SHA256).Hash.ToLower()) --preflight > (Join-Path $taskRun 'actual-user-r2-preflight.stdout.txt') 2> (Join-Path $taskRun 'actual-user-r2-preflight.stderr.txt')
if($LASTEXITCODE -ne 0){throw 'Actual-user CPU-bound preflight failed; no launch'}
$taskPowerShell='C:/Program Files/PowerShell/7/pwsh.exe'
$taskLauncher=Join-Path $taskRun 'launch-saved-key.as-run.ps1'
$taskProcess=Start-Process -FilePath $taskPowerShell -ArgumentList @('-NoProfile','-STA','-File',('"'+$taskLauncher+'"')) -WindowStyle Normal -PassThru -RedirectStandardOutput (Join-Path $taskRun 'saved-key-helper.stdout.txt') -RedirectStandardError (Join-Path $taskRun 'saved-key-helper.stderr.txt')
$taskReceipt=@{status='SAVED_KEY_JUDGE_LAUNCHER_SUBMITTED';pid=$taskProcess.Id;checkpoint_step=157;new_call_limit=792;authorized_new_cap=2376;started_at_utc=[DateTime]::UtcNow.ToString('o');plan_sha256=(Get-FileHash -LiteralPath $taskPlanPath -Algorithm SHA256).Hash.ToLower();source_sha256=$taskPlan.runner_sha256;launcher_sha256=(Get-FileHash -LiteralPath $taskLauncher -Algorithm SHA256).Hash.ToLower();old_unsent_ui_retired=$true;credential_policy='local_windows_dpapi_current_user_file';actual_api_result_not_checked=$true;command=@($taskPowerShell,'-NoProfile','-STA','-File',$taskLauncher)}
[IO.File]::WriteAllText((Join-Path $taskRun 'saved-key-launch-receipt.private.json'),($taskReceipt|ConvertTo-Json -Depth 5),[Text.UTF8Encoding]::new($false))
$taskReceipt|ConvertTo-Json -Depth 5 -Compress
