param(
    [Parameter(Mandatory=$true)][string]$QueueDirectory,
    [Parameter(Mandatory=$true)][string]$SourceSha256,
    [Parameter(Mandatory=$true)][string]$ConfigSha256,
    [switch]$Preflight
)
$ErrorActionPreference='Stop'
function Get-TaskSha([string]$Path){(Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLower()}
function Write-TaskState($State){
    $taskTmp=Join-Path $QueueDirectory ('status.'+[Guid]::NewGuid().ToString('N')+'.tmp')
    [IO.File]::WriteAllText($taskTmp,($State|ConvertTo-Json -Depth 12),[Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $taskTmp -Destination (Join-Path $QueueDirectory 'status.private.json') -Force
}
if((Get-TaskSha $PSCommandPath) -ne $SourceSha256){throw 'Queue source binding failed'}
$taskConfigPath=Join-Path $QueueDirectory 'config.private.json'
if((Get-TaskSha $taskConfigPath) -ne $ConfigSha256){throw 'Queue config binding failed'}
$taskConfig=ConvertFrom-Json -InputObject ([IO.File]::ReadAllText($taskConfigPath))
if(($taskConfig.cases.checkpoint_step -join ',') -ne '313,469' -or $taskConfig.total_authorized_cap -ne 2376 -or $taskConfig.per_checkpoint_cap -ne 792 -or $taskConfig.retries -ne 0){throw 'Only remaining313/469 under original budget allowed'}
if((Get-TaskSha $taskConfig.authorization_file) -ne '40c913b98bc7539d7a2e3e79b3759a6b44cf303f91d7a475805bfc6f2a80909e'){throw 'Original human paid scope changed'}
if((Get-TaskSha $taskConfig.previous_audit) -ne $taskConfig.previous_audit_sha256){throw 'Prior157 audit changed'}
$taskPrevious=ConvertFrom-Json -InputObject ([IO.File]::ReadAllText($taskConfig.previous_audit))
if($taskPrevious.status -ne 'NEW_CHECKPOINT792_RAW_SCHEMA_PAYLOAD_EVENTS_VERIFIED' -or $taskPrevious.checkpoint_step -ne 157 -or $taskPrevious.verified_successes -ne 792){throw 'Original157 completion must be audited'}
if((Get-TaskSha $taskConfig.previous_job) -ne $taskPrevious.job_sha256){throw 'Prior157 completion changed'}
if((Get-TaskSha $taskConfig.auditor) -ne $taskConfig.auditor_sha256){throw 'Auditor source changed'}
if(Test-Path -LiteralPath (Join-Path $QueueDirectory 'status.private.json')){throw 'Never repeat an existing queue'}
$env:CUDA_VISIBLE_DEVICES=''
$env:OPENAI_API_KEY=$null
$env:OPENAI_ORG_ID=$null
$env:OPENAI_PROJECT_ID=$null
$env:OPENAI_LOG=$null
$env:SPF_JUDGE_NEW_KEY_INPUT=$null
$env:SPF_JUDGE_SAVED_KEY_INPUT=$null
foreach($taskCase in $taskConfig.cases){
    $taskPlanPath=Join-Path $taskCase.run_directory 'plan.private.json'
    if((Get-TaskSha $taskPlanPath) -ne $taskCase.plan_sha256){throw 'Own checkpoint plan changed'}
    $taskPlan=ConvertFrom-Json -InputObject ([IO.File]::ReadAllText($taskPlanPath))
    if($taskPlan.checkpoint_step -ne $taskCase.checkpoint_step -or [Security.Principal.WindowsIdentity]::GetCurrent().User.Value -ne $taskPlan.credential_windows_sid){throw 'Exact checkpoint and actual Windows user required'}
    foreach($taskBound in @(@{name='run_spf_checkpoint_development_judge_saved_key.py';sha=$taskPlan.runner_sha256},@{name='start_spf_checkpoint_judge_saved_key.ps1';sha=$taskPlan.credential_dialog_sha256},@{name='local_spf_judge_key.ps1';sha=$taskPlan.key_helper_sha256})){
        if((Get-TaskSha (Join-Path $taskCase.run_directory $taskBound.name)) -ne $taskBound.sha){throw 'Checkpoint source binding failed'}
    }
    if(-not (Test-Path -LiteralPath $taskPlan.credential_file -PathType Leaf)){throw 'Saved encrypted key required; no new dialog in this queue'}
    . (Join-Path $taskCase.run_directory 'local_spf_judge_key.ps1')
    Assert-SpfJudgeKeyPath -Path $taskPlan.credential_file -PrivateRoot $taskPlan.credential_private_root
    Assert-SpfJudgePrivateAcl -Path $taskPlan.credential_file
    Assert-SpfJudgePrivateAcl -Path (Split-Path -Parent $taskPlan.credential_file)
    $taskCheck=& $taskPlan.python_executable (Join-Path $taskCase.run_directory 'run_spf_checkpoint_development_judge_saved_key.py') --run $taskCase.run_directory --source-sha256 $taskPlan.runner_sha256 --plan-sha256 $taskCase.plan_sha256 --preflight 2>&1
    [IO.File]::WriteAllText((Join-Path $QueueDirectory ('checkpoint-'+$taskCase.checkpoint_step+'-preflight.private.txt')),($taskCheck -join "`n"),[Text.UTF8Encoding]::new($false))
    if($LASTEXITCODE -ne 0){throw 'Checkpoint preflight failed before key access'}
}
if($Preflight){'{"status":"REMAINING313469_QUEUE_PREFLIGHT_PASS","api_calls":0,"key_contents_read":false}';exit 0}
$taskState=@{status='RUNNING';pid=$PID;started_at_utc=[DateTime]::UtcNow.ToString('o');source_sha256=$SourceSha256;config_sha256=$ConfigSha256;completed=@();stages=@();retries=0;authorized_total_cap=2376;prior157_verified=792}
Write-TaskState $taskState
try{
    foreach($taskCase in $taskConfig.cases){
        $taskPlan=ConvertFrom-Json -InputObject ([IO.File]::ReadAllText((Join-Path $taskCase.run_directory 'plan.private.json')))
        $taskRow=@{checkpoint_step=$taskCase.checkpoint_step;status='SUBMITTED_RESULT_NOT_CHECKED';started_at_utc=[DateTime]::UtcNow.ToString('o')}
        $taskState.stages+=,$taskRow;$taskState.current_checkpoint=$taskCase.checkpoint_step
        Write-TaskState $taskState
        $taskArguments=@('-NoProfile','-File',('"'+(Join-Path $taskCase.run_directory 'start_spf_checkpoint_judge_saved_key.ps1')+'"'),'-RunDirectory',('"'+$taskCase.run_directory+'"'),'-SourceSha256',$taskPlan.runner_sha256,'-PlanSha256',$taskCase.plan_sha256)
        $taskProcess=Start-Process -FilePath $taskConfig.pwsh_executable -ArgumentList $taskArguments -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $QueueDirectory ('checkpoint-'+$taskCase.checkpoint_step+'-launcher.stdout.private.txt')) -RedirectStandardError (Join-Path $QueueDirectory ('checkpoint-'+$taskCase.checkpoint_step+'-launcher.stderr.private.txt'))
        $taskRow.launcher_pid=$taskProcess.Id;Write-TaskState $taskState
        $taskProcess.WaitForExit();$taskProcess.Refresh();$taskRow.launcher_exit=$taskProcess.ExitCode
        if($taskProcess.ExitCode -ne 0){throw 'Saved-key launcher failed; no retry'}
        $taskJobPath=Join-Path $taskCase.run_directory 'job.json'
        $taskJob=ConvertFrom-Json -InputObject ([IO.File]::ReadAllText($taskJobPath))
        if($taskJob.status -ne 'CHECKPOINT_JUDGE_COMPLETE_PENDING_RESULT_AUDIT_AND_HUMAN_CLUSTER_GATE' -or $taskJob.attempted -ne 792 -or $taskJob.completed -ne 792 -or $taskJob.statuses.success -ne 792 -or $taskJob.plan_sha256 -ne $taskCase.plan_sha256){throw 'Paid Judge incomplete or failed; stop remaining queue without retry'}
        $taskAudit=& $taskPlan.python_executable -X utf8 $taskConfig.auditor --step $taskCase.checkpoint_step --plan-sha256 $taskCase.plan_sha256 2>&1
        [IO.File]::WriteAllText((Join-Path $QueueDirectory ('checkpoint-'+$taskCase.checkpoint_step+'-audit.stdout.private.txt')),($taskAudit -join "`n"),[Text.UTF8Encoding]::new($false))
        if($LASTEXITCODE -ne 0){throw 'Completed responses audit failed; stop without retry'}
        $taskRow.status='JUDGE792_COMPLETE_AND_AUDITED';$taskRow.finished_at_utc=[DateTime]::UtcNow.ToString('o');$taskRow.job_sha256=Get-TaskSha $taskJobPath
        $taskState.completed+=,$taskCase.checkpoint_step;Write-TaskState $taskState
    }
    $taskState.status='REMAINING313469_JUDGE_COMPLETE_PENDING_HUMAN_CLUSTER_GATE'
}catch{
    $taskState.status='STOPPED_NO_RETRY'
    $taskState.failure=@{error_type=$_.Exception.GetType().Name;exception_message_saved=$false}
}finally{
    $taskState.finished_at_utc=[DateTime]::UtcNow.ToString('o');Write-TaskState $taskState
}
if($taskState.status -eq 'STOPPED_NO_RETRY'){exit 1}
