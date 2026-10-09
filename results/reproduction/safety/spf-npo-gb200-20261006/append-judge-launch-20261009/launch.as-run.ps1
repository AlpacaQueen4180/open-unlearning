$ErrorActionPreference='Stop'
$taskQueue='C:\Users\Alpaca\Documents\ChatGPT\Machine Unlearning and Safety Alignment\open-unlearning\work\spf-npo-gb200-20261006\private\checkpoint313469-append-judge-queue-20261009'
$taskSource='C:\Users\Alpaca\Documents\ChatGPT\Machine Unlearning and Safety Alignment\open-unlearning\work\spf-npo-gb200-20261006\private\checkpoint313469-append-judge-queue-20261009\run_spf_checkpoint_judge_append_queue.ps1'
$taskConfig='C:\Users\Alpaca\Documents\ChatGPT\Machine Unlearning and Safety Alignment\open-unlearning\work\spf-npo-gb200-20261006\private\checkpoint313469-append-judge-queue-20261009\config.private.json'
$taskSourceSha='6464c9b3aa5869463151a7d20c9cd0b39b89835438015efd8b3502d9343ebd78'
$taskConfigSha='2ebc98f14e3e36ec04db46416724f1b67d1d56d228535d470c93c481e856351f'
if(Test-Path -LiteralPath (Join-Path $taskQueue 'launch-receipt.private.json')){throw 'Never relaunch existing queue'}
& $taskSource -QueueDirectory $taskQueue -SourceSha256 $taskSourceSha -ConfigSha256 $taskConfigSha -Preflight
if($LASTEXITCODE -ne 0){throw 'Preflight failed; no launch'}
$taskArgs=@('-NoProfile','-File',('"'+$taskSource+'"'),'-QueueDirectory',('"'+$taskQueue+'"'),'-SourceSha256',$taskSourceSha,'-ConfigSha256',$taskConfigSha)
$taskProcess=Start-Process -FilePath 'C:/Program Files/PowerShell/7/pwsh.exe' -WorkingDirectory 'C:\Users\Alpaca\Documents\ChatGPT\Machine Unlearning and Safety Alignment\open-unlearning' -ArgumentList $taskArgs -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskQueue 'queue.stdout.private.txt') -RedirectStandardError (Join-Path $taskQueue 'queue.stderr.private.txt')
$taskReceipt=@{status='SUBMITTED_RESULT_NOT_CHECKED';pid=$taskProcess.Id;submitted_at_utc=[DateTime]::UtcNow.ToString('o');source_sha256=$taskSourceSha;config_sha256=$taskConfigSha;steps=@(313,469);new_request_cap=1365;prior313_verified=219;total_authorized_cap=2376;retries=0;credential_policy='local_windows_dpapi_current_user_file'}
$taskBytes=[Text.UTF8Encoding]::new($false).GetBytes(($taskReceipt|ConvertTo-Json -Depth 6)+"`n");$taskStream=[IO.File]::Open((Join-Path $taskQueue 'launch-receipt.private.json'),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read);try{$taskStream.Write($taskBytes,0,$taskBytes.Length);$taskStream.Flush($true)}finally{$taskStream.Dispose()}
$taskReceipt|ConvertTo-Json -Depth 6 -Compress
