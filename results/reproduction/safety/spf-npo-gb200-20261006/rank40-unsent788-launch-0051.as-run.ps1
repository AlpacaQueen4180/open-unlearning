$ErrorActionPreference='Stop'
$taskRoot=(Get-Location).Path
$taskRun=Join-Path $taskRoot 'work/spf-npo-gb200-20261006/private/spf-safety-search-20261010/model-06-hybrid01-rank40-unsent788-r2'
$taskSource='557159f211ca4aa73f088816f785f60bf534c3dfad49693b05f6bcf4d622e0df'
$taskPlanSha='494832e9c379351b2ed0a04bf99a5bd5e7922e93a339d372fef1f7572d5b00aa'
$taskLoader=Join-Path $taskRun 'start_spf_safety_search_judge.ps1'
if((Get-FileHash -LiteralPath $taskLoader).Hash.ToLower() -ne '8d5cca098f33740397b345f64804903297178689f5e206e7e6fc5d9bf8e558ad'){throw 'Loader binding failed'}
if((Get-FileHash -LiteralPath (Join-Path $taskRun 'plan.private.json')).Hash.ToLower() -ne $taskPlanSha){throw 'Plan binding failed'}
if((Get-FileHash -LiteralPath (Join-Path $taskRun 'run_spf_safety_search_judge.py')).Hash.ToLower() -ne $taskSource){throw 'Runner binding failed'}
$taskPlan=[IO.File]::ReadAllText((Join-Path $taskRun 'plan.private.json'))|ConvertFrom-Json
$taskClaim=Get-Content -LiteralPath (Join-Path $taskPlan.budget_root 'hourly-claim-20261013-0051.private.json') -Raw|ConvertFrom-Json
if($taskClaim.returncode -ne 0 -or $taskClaim.clock_override -or $taskClaim.state.reserved_models -ne 5 -or $taskClaim.state.complete_models -ne 5){throw 'Original hourly claim and completed prior budget required'}
if($taskPlan.spf_projection_rank -ne 40 -or -not $taskPlan.final_spf_candidate){throw 'Only final rank40 closeout allowed'}
if($taskPlan.method_family -ne 'spf_anchor_loss_hybrid' -or $taskPlan.safety_loss_coefficient -ne 0.1 -or $taskPlan.full_hybrid_acceptance_sha256 -ne '3050b474dee9f3e5ace7458ecd0590dd3ba38b392537acc6547b36dadf0dd19f'){throw 'Explicit hybrid completion binding required'}
if((Get-FileHash -LiteralPath (Join-Path $taskRun 'development-proof.private.json')).Hash.ToLower() -ne '3cea2c8df244fa179f072b9403f58489ed468c9969f041eb239ae79768a5a680'){throw 'New full development proof required'}

if($taskPlan.model_identity_sha256 -ne '64d3fd783130bc4936a97cbc31da044dfd5e20487b35f2e71a21f6e20ad30c78' -or $taskPlan.packet_sha256 -ne 'd757a2450319c8802c8f06fb8e897b66e27a5b8ff58f9485cdfcd80752e3f33d' -or $taskPlan.new_campaign_model_limit -ne 50 -or $taskPlan.total -ne 792){throw 'Exact new model payload and campaign cap required'}
if($taskPlan.campaign_slot -ne 6 -or $taskPlan.new_calls -ne 788 -or $taskPlan.verified_prior_successes -ne 3 -or (($taskPlan.reserved_uncertain_indices -join ',') -ne '3') -or (($taskPlan.selected_packet_indices -join ',') -ne ((4..791) -join ','))){throw 'Original slot6 verified788unsent only'}
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
$taskIntent=@{status='SINGLE_LAUNCH_INTENT_BEFORE_STARTPROCESS';utc=[DateTime]::UtcNow.ToString('o');source_sha256=$taskSource;plan_sha256=$taskPlanSha;model_limit=50;this_model_request_limit=788;original_model_allocation=792;reuses_slot=6;reserved_uncertain=1;reserve_before_key=$true}
$taskIntentRaw=[Text.UTF8Encoding]::new($false).GetBytes(($taskIntent|ConvertTo-Json))
$taskIntentStream=[IO.File]::Open((Join-Path $taskRun 'launch-intent.private.json'),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
try{$taskIntentStream.Write($taskIntentRaw,0,$taskIntentRaw.Length);$taskIntentStream.Flush($true)}finally{$taskIntentStream.Dispose()}
$taskArguments=@('-NoProfile','-File',('"'+$taskLoader+'"'),'-RunDirectory',('"'+$taskRun+'"'),'-SourceSha256',$taskSource,'-PlanSha256',$taskPlanSha)
$taskChild=Start-Process -FilePath 'C:/Program Files/PowerShell/7/pwsh.exe' -ArgumentList $taskArguments -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskRun 'loader.stdout.private.txt') -RedirectStandardError (Join-Path $taskRun 'loader.stderr.private.txt') -PassThru
$taskReceipt=@{status='SUBMITTED_RESULT_NOT_CHECKED';pid=$taskChild.Id;submitted_utc=[DateTime]::UtcNow.ToString('o');source_sha256=$taskSource;plan_sha256=$taskPlanSha;this_model_request_limit=788;original_model_allocation=792;reuses_slot=6;reserved_uncertain=1;new_campaign_model_limit=50;new_key_dialog=$false;actual_parser_errors=0;reserve_before_key=$true;api_endpoint='https://api.openai.com/v1';model='gpt-5.6-terra';max_retries=0}
[IO.File]::WriteAllText((Join-Path $taskRun 'launch-receipt.private.json'),($taskReceipt|ConvertTo-Json),[Text.UTF8Encoding]::new($false))
$taskReceipt|ConvertTo-Json -Compress
