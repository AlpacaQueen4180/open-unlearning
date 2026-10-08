$ErrorActionPreference='Stop'
$taskRun=Join-Path $PSScriptRoot 'private/spf-checkpoint-judge-20261008/checkpoint-157-r1'
$taskPlanPath=Join-Path $taskRun 'plan.private.json'
$taskPlan=ConvertFrom-Json -InputObject ([IO.File]::ReadAllText($taskPlanPath))
$taskChecks=ConvertFrom-Json -InputObject ([IO.File]::ReadAllText((Join-Path $PSScriptRoot 'private/checkpoint-judge-mock-20261008/checks.private.json')))
if($taskChecks.status -ne 'NEW_CHECKPOINT_792_LOOP_AND_POST_RESPONSE_FILESYSTEM_FAILSTOP_MOCK_PASS' -or $taskChecks.real_api_calls -ne 0 -or $taskChecks.mock_http_requests.complete -ne 792 -or $taskChecks.mock_http_requests.save_failure -ne 1){throw 'New mock integration not verified'}
foreach($taskName in @('job.json','execution.lock','raw-responses','execution-events.private.jsonl','credential-input-status.private.json','dialog-launch-receipt.private.json','dialog-helper.stdout.txt','dialog-helper.stderr.txt')){if(Test-Path -LiteralPath (Join-Path $taskRun $taskName)){throw 'New dialog/execution already exists'}}
if(Test-Path -LiteralPath (Join-Path $taskRun '../allocations/157.json')){throw 'Checkpoint already allocated'}
foreach($taskPriorId in @(38808,43480,38848,23980,48688)){
 $taskProcess=Get-CimInstance Win32_Process -Filter ('ProcessId='+$taskPriorId)
 if($taskProcess -and $taskProcess.CommandLine -like '*spf-judge-local-20261007*'){throw 'Prior own process still live'}
}
$taskPowerShell='C:/Program Files/PowerShell/7/pwsh.exe'
$taskDialog=Join-Path $taskRun 'start_spf_checkpoint_judge.ps1'
$taskLauncher=Join-Path $taskRun 'launch-dialog.as-run.ps1'
$taskRunner=Join-Path $taskRun 'run_spf_checkpoint_development_judge.py'
if((Get-FileHash -LiteralPath $taskDialog -Algorithm SHA256).Hash.ToLower() -ne $taskPlan.credential_dialog_sha256){throw 'Dialog SHA binding failed'}
if((Get-FileHash -LiteralPath $taskRunner -Algorithm SHA256).Hash.ToLower() -ne $taskChecks.source_sha256){throw 'Tested runner SHA binding failed'}
$taskErrors=@()
foreach($taskFile in @($taskDialog,$taskLauncher)){
 $taskTokens=$null;$taskCurrentErrors=$null
 [System.Management.Automation.Language.Parser]::ParseFile($taskFile,[ref]$taskTokens,[ref]$taskCurrentErrors)|Out-Null
 $taskErrors+=@($taskCurrentErrors)
}
$taskParser=@{parser_error_count=$taskErrors.Count;engine_version=$PSVersionTable.PSVersion.ToString();files=2}
[IO.File]::WriteAllText((Join-Path $taskRun 'actual-powershell-parser.private.json'),($taskParser|ConvertTo-Json),[Text.UTF8Encoding]::new($false))
if($taskErrors.Count -ne 0 -or $PSVersionTable.PSVersion.Major -ne 7){throw 'PowerShell7 parser required; no launch'}
$taskNewProcess=Start-Process -FilePath $taskPowerShell -ArgumentList @('-NoProfile','-STA','-File',('"'+$taskLauncher+'"')) -WindowStyle Normal -PassThru -RedirectStandardOutput (Join-Path $taskRun 'dialog-helper.stdout.txt') -RedirectStandardError (Join-Path $taskRun 'dialog-helper.stderr.txt')
$taskReceipt=@{status='CHECKPOINT157_DIALOG_SUBMITTED_DISPLAY_AND_KEY_PENDING';pid=$taskNewProcess.Id;checkpoint_step=157;new_call_limit=792;authorized_new_cap=2376;new_api_calls_at_launch=0;key_persisted=$false;actual_powershell_parser_pass=$true;started_at_utc=[DateTime]::UtcNow.ToString('o');plan_sha256=(Get-FileHash -LiteralPath $taskPlanPath -Algorithm SHA256).Hash.ToLower();launcher_sha256=(Get-FileHash -LiteralPath $taskLauncher -Algorithm SHA256).Hash.ToLower();command=@($taskPowerShell,'-NoProfile','-STA','-File',$taskLauncher)}
[IO.File]::WriteAllText((Join-Path $taskRun 'dialog-launch-receipt.private.json'),($taskReceipt|ConvertTo-Json -Depth 6),[Text.UTF8Encoding]::new($false))
$taskReceipt|ConvertTo-Json -Depth 6 -Compress
