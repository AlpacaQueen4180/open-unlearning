param([Parameter(Mandatory=$true)][string]$JournalDirectory)
$ErrorActionPreference='Stop'
$env:CUDA_VISIBLE_DEVICES=''
foreach($taskVar in @('OPENAI_API_KEY','OPENAI_ORG_ID','OPENAI_PROJECT_ID','OPENAI_LOG','SPF_JUDGE_SAVED_KEY_INPUT')){[Environment]::SetEnvironmentVariable($taskVar,$null,'Process')}
$taskRoot=(Get-Location).Path
$taskRun=Join-Path $taskRoot 'work/spf-npo-gb200-20261006/private/spf-checkpoint-judge-20261008/checkpoint-469-saved-key-r3-unsent395'
$taskAuditor=Join-Path $taskRoot 'scripts/reproduction/safety/audit_spf_checkpoint469_unsent395.py'
$taskSha='0ea4add90afb48343bce696acf45b84dc05f0feb89e17fc3c60baba8a0a32aa1'
$taskAuditDirectory=Join-Path $taskRoot 'work/spf-npo-gb200-20261006/private/checkpoint-469-unsent395-combined791-audit-20261009'
$taskPlanPath=Join-Path $taskRun 'plan.private.json'
$taskState=@{status='WAITING_FOR_ORIGINAL395_LOADER_EXIT';source_sha256=$taskSha;run_directory=$taskRun;started_utc=[DateTime]::UtcNow.ToString('o');api_calls=0;key_contents_read=$false}
function Write-TaskExclusive([string]$Name,$Value){
 $taskBytes=[Text.UTF8Encoding]::new($false).GetBytes(($Value|ConvertTo-Json -Depth 12)+"`n")
 $taskStream=[IO.File]::Open((Join-Path $JournalDirectory $Name),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
 try{$taskStream.Write($taskBytes,0,$taskBytes.Length);$taskStream.Flush($true)}finally{$taskStream.Dispose()}
}
try{
 if((Get-FileHash -LiteralPath $taskAuditor).Hash.ToLower() -ne $taskSha){throw 'Auditor binding failed'}
 if((Get-FileHash -LiteralPath $taskPlanPath).Hash.ToLower() -ne '268bc260861a0780b98ae77f0378e54bb38bb2884d39b434d37efdbf7bf7dac5'){throw 'Original395 plan required'}
 if(Test-Path -LiteralPath $taskAuditDirectory){throw 'Never rerun completed audit'}
 $taskReceipt=[IO.File]::ReadAllText((Join-Path $taskRun 'launch-receipt.private.json'))|ConvertFrom-Json
 if($taskReceipt.pid -ne 61220 -or $taskReceipt.new_call_limit -ne 395){throw 'Original single submitted loader identity required'}
 Write-TaskExclusive 'waiting.private.json' $taskState
 $taskProcess=Get-CimInstance Win32_Process -Filter 'ProcessId=61220'
 if($null -ne $taskProcess){
  if($taskProcess.CommandLine -notlike '*start_spf_checkpoint469_unsent395.ps1*'){throw 'Own original loader identity required'}
  Wait-Process -Id 61220
 }
 $taskJob=[IO.File]::ReadAllText((Join-Path $taskRun 'job.json'))|ConvertFrom-Json
 if($taskJob.status -ne 'CHECKPOINT_JUDGE_COMPLETE_PENDING_RESULT_AUDIT_AND_HUMAN_CLUSTER_GATE' -or $taskJob.attempted -ne 395 -or $taskJob.completed -ne 395 -or $taskJob.statuses.success -ne 395){
  $taskState.status='ORIGINAL395_NOT_COMPLETE_NO_AUDIT_NO_RETRY';$taskState.job_status=$taskJob.status;$taskState.attempted=$taskJob.attempted;$taskState.completed=$taskJob.completed
  Write-TaskExclusive 'result.private.json' $taskState
  exit 1
 }
 if((Get-FileHash -LiteralPath $taskAuditor).Hash.ToLower() -ne $taskSha){throw 'Auditor changed before audit'}
 $taskPython=Join-Path $taskRoot 'work/spf-npo-gb200-20261006/private/local-judge-venv/Scripts/python.exe'
 & $taskPython $taskAuditor > (Join-Path $JournalDirectory 'audit.stdout.private.txt') 2> (Join-Path $JournalDirectory 'audit.stderr.private.txt')
 if($LASTEXITCODE -ne 0){throw 'New395 audit failed; no retry'}
 $taskAudit=[IO.File]::ReadAllText((Join-Path $taskAuditDirectory 'audit.private.json'))|ConvertFrom-Json
 if($taskAudit.verified_successes -ne 791 -or $taskAudit.new_raw_responses_verified -ne 395 -or $taskAudit.missing -ne 1 -or $taskAudit.full792_complete -ne $false){throw '791 plus uncertain1 audit required'}
 $taskState.status='NEW395_AND_PRIOR396_COMBINED791_AUDIT_PASS';$taskState.finished_utc=[DateTime]::UtcNow.ToString('o');$taskState.confirmed=791;$taskState.reserved_uncertain=1
 $taskState.audit_sha256=(Get-FileHash -LiteralPath (Join-Path $taskAuditDirectory 'audit.private.json')).Hash.ToLower()
 Write-TaskExclusive 'result.private.json' $taskState
}catch{
 $taskState.status='AUDIT_FOLLOWUP_STOPPED_NO_RETRY';$taskState.error_type=$_.Exception.GetType().Name;$taskState.exception_message_saved=$false;$taskState.finished_utc=[DateTime]::UtcNow.ToString('o')
 if(-not (Test-Path -LiteralPath (Join-Path $JournalDirectory 'result.private.json'))){Write-TaskExclusive 'result.private.json' $taskState}
 exit 1
}
