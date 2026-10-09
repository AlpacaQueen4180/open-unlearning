param(
    [Parameter(Mandatory=$true)][string]$RunDirectory,
    [Parameter(Mandatory=$true)][string]$SourceSha256,
    [Parameter(Mandatory=$true)][string]$PlanSha256
)
$ErrorActionPreference='Stop'
$env:CUDA_VISIBLE_DEVICES=''
$env:OPENAI_API_KEY=$null;$env:OPENAI_ORG_ID=$null;$env:OPENAI_PROJECT_ID=$null;$env:OPENAI_LOG=$null
$env:SPF_JUDGE_NEW_KEY_INPUT=$null;$env:SPF_JUDGE_SAVED_KEY_INPUT=$null
$taskPlanPath=Join-Path $RunDirectory 'plan.private.json'
if((Get-FileHash -LiteralPath $taskPlanPath -Algorithm SHA256).Hash.ToLower() -ne $PlanSha256){throw 'Plan binding failed'}
$taskPlan=ConvertFrom-Json -InputObject ([IO.File]::ReadAllText($taskPlanPath))
$taskRunner=Join-Path $RunDirectory 'run_spf_checkpoint469_unsent395.py'
$taskHelper=Join-Path $RunDirectory 'local_spf_judge_key.ps1'
if((Get-FileHash -LiteralPath $taskRunner -Algorithm SHA256).Hash.ToLower() -ne $SourceSha256){throw 'Append-state source binding failed'}
if((Get-FileHash -LiteralPath $taskHelper -Algorithm SHA256).Hash.ToLower() -ne $taskPlan.key_helper_sha256){throw 'Key helper binding failed'}
if([Security.Principal.WindowsIdentity]::GetCurrent().User.Value -ne $taskPlan.credential_windows_sid){throw 'Actual authorized Windows user required'}
. $taskHelper
$taskArguments=@($taskRunner,'--run',$RunDirectory,'--source-sha256',$SourceSha256,'--plan-sha256',$PlanSha256)
$taskPreflight=& $taskPlan.python_executable @taskArguments --preflight 2>&1
[IO.File]::WriteAllText((Join-Path $RunDirectory 'preflight-before-key.private.txt'),($taskPreflight -join "`n"),[Text.UTF8Encoding]::new($false))
if($LASTEXITCODE -ne 0){throw 'CPU-only preflight failed before key access'}
$taskSecure=$null
try{
    if(-not (Test-Path -LiteralPath $taskPlan.credential_file -PathType Leaf)){throw 'Previously saved encrypted key required'}
    $taskSecure=Read-SpfJudgeKey -Path $taskPlan.credential_file -PrivateRoot $taskPlan.credential_private_root
    [IO.File]::WriteAllText((Join-Path $RunDirectory 'credential-input-status.private.json'),'{"status":"SAVED_KEY_LOADED_TO_CHILD_MEMORY","key_saved":true,"file_encrypted":true,"new_dialog_opened":false}',[Text.UTF8Encoding]::new($false))
    $env:OPENAI_API_KEY=([Net.NetworkCredential]::new('',$taskSecure)).Password
    $env:SPF_JUDGE_SAVED_KEY_INPUT='local_windows_dpapi_current_user_file'
    & $taskPlan.python_executable @taskArguments > (Join-Path $RunDirectory 'runner-output.private.txt')
    $taskExit=$LASTEXITCODE
    [IO.File]::WriteAllText((Join-Path $RunDirectory 'launcher-result.private.json'),('{"runner_exit_code":'+$taskExit+',"key_persisted":true}'),[Text.UTF8Encoding]::new($false))
    exit $taskExit
}catch{
    $taskFailure=@{status='APPEND_STATE_LOADER_STOPPED_NO_RETRY';error_type=$_.Exception.GetType().Name;exception_message_saved=$false}
    [IO.File]::WriteAllText((Join-Path $RunDirectory 'credential-launcher-error.private.json'),($taskFailure|ConvertTo-Json),[Text.UTF8Encoding]::new($false))
    exit 1
}finally{
    $env:OPENAI_API_KEY=$null;$env:SPF_JUDGE_SAVED_KEY_INPUT=$null
    if($null -ne $taskSecure){$taskSecure.Dispose()}
}
