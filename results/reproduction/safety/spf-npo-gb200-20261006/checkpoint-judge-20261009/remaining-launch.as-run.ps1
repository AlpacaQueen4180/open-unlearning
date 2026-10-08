$ErrorActionPreference='Stop'
$taskQueue='C:\Users\Alpaca\Documents\ChatGPT\Machine Unlearning and Safety Alignment\open-unlearning\work\spf-npo-gb200-20261006\private\checkpoint313469-judge-queue-20261009'
$taskSource='C:\Users\Alpaca\Documents\ChatGPT\Machine Unlearning and Safety Alignment\open-unlearning\work\spf-npo-gb200-20261006\private\checkpoint313469-judge-queue-20261009\run_spf_checkpoint_judge_remaining_queue.ps1'
$taskConfig='C:\Users\Alpaca\Documents\ChatGPT\Machine Unlearning and Safety Alignment\open-unlearning\work\spf-npo-gb200-20261006\private\checkpoint313469-judge-queue-20261009\config.private.json'
$taskSourceSha='48d3d989c07ff99431d17feb48781270597d96c9af2c31f88425273a64033895'
$taskConfigSha='a9d8b9b25c7c4d4c6f008ea0c24efa924a428b2fb2f6ac9dd2f0d550c95b9ba2'
if(Test-Path -LiteralPath (Join-Path $taskQueue 'launch-receipt.private.json')){throw 'Never relaunch existing queue'}
if((Get-FileHash -LiteralPath $taskSource -Algorithm SHA256).Hash.ToLower() -ne $taskSourceSha){throw 'Source binding failed'}
if((Get-FileHash -LiteralPath $taskConfig -Algorithm SHA256).Hash.ToLower() -ne $taskConfigSha){throw 'Config binding failed'}
$taskProcesses=@(Get-CimInstance Win32_Process -Filter "ProcessId=41604 OR ProcessId=49876" | Where-Object {$_.CommandLine -like '*checkpoint-157-saved-key-r1*'});if($taskProcesses.Count -ne 0){throw 'Prior157 process must have exited'}
& $taskSource -QueueDirectory $taskQueue -SourceSha256 $taskSourceSha -ConfigSha256 $taskConfigSha -Preflight
if($LASTEXITCODE -ne 0){throw 'Preflight failed; no launch'}
$taskArgs=@('-NoProfile','-File',('"'+$taskSource+'"'),'-QueueDirectory',('"'+$taskQueue+'"'),'-SourceSha256',$taskSourceSha,'-ConfigSha256',$taskConfigSha)
$taskProcess=Start-Process -FilePath 'C:/Program Files/PowerShell/7/pwsh.exe' -ArgumentList $taskArgs -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskQueue 'queue.stdout.private.txt') -RedirectStandardError (Join-Path $taskQueue 'queue.stderr.private.txt')
$taskReceipt=@{status='SUBMITTED_RESULT_NOT_CHECKED';pid=$taskProcess.Id;submitted_at_utc=[DateTime]::UtcNow.ToString('o');source_sha256=$taskSourceSha;config_sha256=$taskConfigSha;steps=@(313,469);new_request_cap=1584;total_authorized_cap=2376;retries=0;credential_policy='local_windows_dpapi_current_user_file'}
[IO.File]::WriteAllText((Join-Path $taskQueue 'launch-receipt.private.json'),($taskReceipt|ConvertTo-Json -Depth 6),[Text.UTF8Encoding]::new($false))
$taskReceipt|ConvertTo-Json -Depth 6 -Compress
