$ErrorActionPreference='Stop'
$taskRoot=(Get-Location).Path
$taskRun=Join-Path $taskRoot 'work/spf-npo-gb200-20261006/private/spf-safety-search-20261010/model-04-hybrid01-lr5e6-full-r1'
$taskSource='9d7203d95db3a828b4e42ead96a0b1a178f711254a2cd5cac5df43d1fc3dee2e'
$taskPlanSha='1980e38cb55e8cc91accfa70331139901dfd88c482b40694ffbd37022f2fd70a'
$taskLoader=Join-Path $taskRun 'start_spf_safety_search_judge.ps1'
if((Get-FileHash -LiteralPath $taskLoader).Hash.ToLower() -ne '8d5cca098f33740397b345f64804903297178689f5e206e7e6fc5d9bf8e558ad'){throw 'Loader binding failed'}
if((Get-FileHash -LiteralPath (Join-Path $taskRun 'plan.private.json')).Hash.ToLower() -ne $taskPlanSha){throw 'Plan binding failed'}
if((Get-FileHash -LiteralPath (Join-Path $taskRun 'run_spf_safety_search_judge.py')).Hash.ToLower() -ne $taskSource){throw 'Runner binding failed'}
$taskPlan=[IO.File]::ReadAllText((Join-Path $taskRun 'plan.private.json'))|ConvertFrom-Json
$taskClaim=Get-Content -LiteralPath (Join-Path $taskPlan.budget_root 'hourly-claim-20261012-0225.private.json') -Raw|ConvertFrom-Json
if($taskClaim.returncode -ne 0 -or $taskClaim.clock_override -or $taskClaim.state.reserved_models -ne 3 -or $taskClaim.state.complete_models -ne 3){throw 'Original hourly claim and completed prior budget required'}
if($taskPlan.method_family -ne 'spf_anchor_loss_hybrid' -or $taskPlan.safety_loss_coefficient -ne 0.1 -or $taskPlan.full_hybrid_acceptance_sha256 -ne '5457809bceb7539fb13618ee3e3b8c736a2224d02fa838a0fe59a90e68621c6c'){throw 'Explicit hybrid completion binding required'}
if((Get-FileHash -LiteralPath (Join-Path $taskRun 'development-proof.private.json')).Hash.ToLower() -ne 'a30968fc90c9f556e8f3463d0caa40fdc36f71162740b06ebeaf8a1a09f777b1'){throw 'New full development proof required'}

if($taskPlan.model_identity_sha256 -ne '7c3636e230dc3c53ee824572488191ae103f5c5fa4f85af28eb7b79a4ef14962' -or $taskPlan.packet_sha256 -ne '03cabe0ab431ef9ccd5d1631976cfecc973d7698dd906611fcdd87dba8e0bade' -or $taskPlan.new_campaign_model_limit -ne 50 -or $taskPlan.total -ne 792){throw 'Exact new model payload and campaign cap required'}
$taskAuth=Join-Path $taskPlan.budget_root 'authorization.private.json'
if((Get-FileHash -LiteralPath $taskAuth).Hash.ToLower() -ne 'd29a89e81d76edd7bc4571ec6c4d1c42322bd1749a703d94c7cc2ac1d231e283'){throw 'New50 human authorization binding failed'}
if([Security.Principal.WindowsIdentity]::GetCurrent().User.Value -ne $taskPlan.credential_windows_sid){throw 'Authorized DPAPI account required'}
$taskOwnJudges=@(Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pwsh.exe'" | Where-Object {
    $_.CommandLine -like '*run_spf_safety_search_judge.py*' -or
    $_.CommandLine -like '*start_spf_safety_search_judge.ps1*' -or
    $_.CommandLine -like '*run_spf_checkpoint469_unsent395.py*' -or
    $_.CommandLine -like '*start_spf_checkpoint469_unsent395.ps1*' -or
    $_.CommandLine -like '*run_spf_checkpoint_judge_append_state.py*'
})
if($taskOwnJudges.Count -ne 0){throw 'Prior own Judge must have exited; no concurrent model launch'}
foreach($taskName in @('launch-intent.private.json','launch-receipt.private.json','reservation.private.json','job.json','execution.lock','state-snapshots','raw-responses','judge-results.jsonl')){
    if(Test-Path -LiteralPath (Join-Path $taskRun $taskName)){throw 'New run must be unused'}
}
$taskTokens=$null;$taskErrors=$null
[void][Management.Automation.Language.Parser]::ParseFile($taskLoader,[ref]$taskTokens,[ref]$taskErrors)
if($taskErrors.Count -ne 0){throw 'Actual PowerShell7 parser rejected loader'}
[IO.File]::WriteAllBytes((Join-Path $taskRun 'launch.as-run.ps1'),[IO.File]::ReadAllBytes($PSCommandPath))
$taskIntent=@{status='SINGLE_LAUNCH_INTENT_BEFORE_STARTPROCESS';utc=[DateTime]::UtcNow.ToString('o');source_sha256=$taskSource;plan_sha256=$taskPlanSha;model_limit=50;this_model_request_limit=792;reserve_before_key=$true}
$taskIntentRaw=[Text.UTF8Encoding]::new($false).GetBytes(($taskIntent|ConvertTo-Json))
$taskIntentStream=[IO.File]::Open((Join-Path $taskRun 'launch-intent.private.json'),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
try{$taskIntentStream.Write($taskIntentRaw,0,$taskIntentRaw.Length);$taskIntentStream.Flush($true)}finally{$taskIntentStream.Dispose()}
$taskArguments=@('-NoProfile','-File',('"'+$taskLoader+'"'),'-RunDirectory',('"'+$taskRun+'"'),'-SourceSha256',$taskSource,'-PlanSha256',$taskPlanSha)
$taskChild=Start-Process -FilePath 'C:/Program Files/PowerShell/7/pwsh.exe' -ArgumentList $taskArguments -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskRun 'loader.stdout.private.txt') -RedirectStandardError (Join-Path $taskRun 'loader.stderr.private.txt') -PassThru
$taskReceipt=@{status='SUBMITTED_RESULT_NOT_CHECKED';pid=$taskChild.Id;submitted_utc=[DateTime]::UtcNow.ToString('o');source_sha256=$taskSource;plan_sha256=$taskPlanSha;this_model_request_limit=792;new_campaign_model_limit=50;new_key_dialog=$false;actual_parser_errors=0;reserve_before_key=$true;api_endpoint='https://api.openai.com/v1';model='gpt-5.6-terra';max_retries=0}
[IO.File]::WriteAllText((Join-Path $taskRun 'launch-receipt.private.json'),($taskReceipt|ConvertTo-Json),[Text.UTF8Encoding]::new($false))
$taskReceipt|ConvertTo-Json -Compress
