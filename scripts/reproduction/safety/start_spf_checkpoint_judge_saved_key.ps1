param(
    [Parameter(Mandatory=$true)][string]$RunDirectory,
    [Parameter(Mandatory=$true)][string]$SourceSha256,
    [Parameter(Mandatory=$true)][string]$PlanSha256
)
$ErrorActionPreference='Stop'
$env:CUDA_VISIBLE_DEVICES=''
$env:OPENAI_API_KEY=$null
$env:OPENAI_ORG_ID=$null
$env:OPENAI_PROJECT_ID=$null
$env:OPENAI_LOG=$null
$env:SPF_JUDGE_NEW_KEY_INPUT=$null
$env:SPF_JUDGE_SAVED_KEY_INPUT=$null
$taskPlanPath=Join-Path $RunDirectory 'plan.private.json'
if((Get-FileHash -LiteralPath $taskPlanPath -Algorithm SHA256).Hash.ToLower() -ne $PlanSha256){throw 'Bound saved-key plan changed'}
$taskPlan=ConvertFrom-Json -InputObject ([IO.File]::ReadAllText($taskPlanPath))
$taskRunner=Join-Path $RunDirectory 'run_spf_checkpoint_development_judge_saved_key.py'
$taskHelper=Join-Path $RunDirectory 'local_spf_judge_key.ps1'
if((Get-FileHash -LiteralPath $taskRunner -Algorithm SHA256).Hash.ToLower() -ne $SourceSha256){throw 'Bound saved-key runner changed'}
if((Get-FileHash -LiteralPath $taskHelper -Algorithm SHA256).Hash.ToLower() -ne $taskPlan.key_helper_sha256){throw 'Bound key helper changed'}
if($taskPlan.credential_policy -ne 'local_windows_dpapi_current_user_file'){throw 'User-authorized file credential policy required'}
if([Security.Principal.WindowsIdentity]::GetCurrent().User.Value -ne $taskPlan.credential_windows_sid){throw 'Use the Windows account authorized to store the dedicated Judge key'}
. $taskHelper
$taskArguments=@($taskRunner,'--run',$RunDirectory,'--source-sha256',$SourceSha256,'--plan-sha256',$PlanSha256)
$taskPreflight=& $taskPlan.python_executable @taskArguments --preflight 2>&1
if($LASTEXITCODE -ne 0){throw 'Judge preflight failed before key access'}
[IO.File]::WriteAllText((Join-Path $RunDirectory 'saved-key-preflight.private.json'),($taskPreflight -join "`n"),[Text.UTF8Encoding]::new($false))
$taskKeyPath=$taskPlan.credential_file
$taskPrivateRoot=$taskPlan.credential_private_root
Assert-SpfJudgeKeyPath -Path $taskKeyPath -PrivateRoot $taskPrivateRoot
$taskForm=$null;$taskSecure=$null
try {
    if(-not (Test-Path -LiteralPath $taskKeyPath -PathType Leaf)){
        Add-Type -AssemblyName System.Windows.Forms
        Add-Type -AssemblyName System.Drawing
        $taskForm=[Windows.Forms.Form]::new()
        $taskForm.Text='SPF Judge — 儲存專用 key 並啟動'
        $taskForm.Size=[Drawing.Size]::new(640,330)
        $taskForm.StartPosition='CenterScreen'
        $taskForm.FormBorderStyle='FixedDialog'
        $taskForm.MaximizeBox=$false
        $taskForm.ShowInTaskbar=$true
        $taskForm.TopMost=$true
        $taskLabel=[Windows.Forms.Label]::new()
        $taskLabel.Location=[Drawing.Point]::new(20,20)
        $taskLabel.Size=[Drawing.Size]::new(590,140)
        $taskLabel.Text="請輸入專用 OpenAI API key，這次會加密儲存供後續 Judge 使用。`nWindows 帳戶加密；檔案只允許目前帳戶存取。`n`n本批 checkpoint $($taskPlan.checkpoint_step)：最多792次；三批總上限2376、零重試。`nOpenAI /v1；gpt-5.6-terra、medium、4096 tokens。`n已完成與不確定的舊請求不重送。"
        $taskBox=[Windows.Forms.TextBox]::new()
        $taskBox.Location=[Drawing.Point]::new(20,170)
        $taskBox.Size=[Drawing.Size]::new(590,28)
        $taskBox.UseSystemPasswordChar=$true
        $taskButton=[Windows.Forms.Button]::new()
        $taskButton.Location=[Drawing.Point]::new(370,225)
        $taskButton.Size=[Drawing.Size]::new(240,35)
        $taskButton.Text='加密儲存並啟動已授權 Judge'
        $taskButton.Add_Click({
            if([string]::IsNullOrWhiteSpace($taskBox.Text)){return}
            $taskForm.Tag=ConvertTo-SecureString -String $taskBox.Text.Trim() -AsPlainText -Force
            $taskBox.Clear()
            $taskForm.DialogResult=[Windows.Forms.DialogResult]::OK
            $taskForm.Close()
        })
        $taskForm.Controls.AddRange(@($taskLabel,$taskBox,$taskButton))
        $taskForm.AcceptButton=$taskButton
        $taskForm.Add_Shown({
            $taskForm.Activate();$taskForm.BringToFront();$taskBox.Focus()
            $taskStatus=@{status='WAITING_FOR_KEY_TO_SAVE_ENCRYPTED_FILE';api_calls_executed=0;shown_event_received=$true;form_visible=$taskForm.Visible;topmost=$taskForm.TopMost;show_in_taskbar=$taskForm.ShowInTaskbar;key_saved=$false}
            [IO.File]::WriteAllText((Join-Path $RunDirectory 'credential-input-status.private.json'),($taskStatus|ConvertTo-Json),[Text.UTF8Encoding]::new($false))
        })
        if($taskForm.ShowDialog() -ne [Windows.Forms.DialogResult]::OK){
            [IO.File]::WriteAllText((Join-Path $RunDirectory 'credential-input-status.private.json'),'{"status":"INPUT_CLOSED_WITHOUT_KEY","api_calls_executed":0,"key_saved":false}',[Text.UTF8Encoding]::new($false))
            exit 0
        }
        $taskSecure=$taskForm.Tag
        Save-SpfJudgeKey -Path $taskKeyPath -PrivateRoot $taskPrivateRoot -Key $taskSecure
        $taskSecure.Dispose();$taskSecure=$null;$taskForm.Tag=$null
    }
    $taskSecure=Read-SpfJudgeKey -Path $taskKeyPath -PrivateRoot $taskPrivateRoot
    [IO.File]::WriteAllText((Join-Path $RunDirectory 'credential-input-status.private.json'),'{"status":"SAVED_KEY_LOADED_TO_CHILD_MEMORY","key_saved":true,"file_encrypted":true,"credential_source":"local_current_user_dpapi_file"}',[Text.UTF8Encoding]::new($false))
    $env:OPENAI_API_KEY=([Net.NetworkCredential]::new('',$taskSecure)).Password
    $env:SPF_JUDGE_SAVED_KEY_INPUT='local_windows_dpapi_current_user_file'
    & $taskPlan.python_executable @taskArguments > (Join-Path $RunDirectory 'runner-output.private.txt')
    $taskRunnerExit=$LASTEXITCODE
    [IO.File]::WriteAllText((Join-Path $RunDirectory 'launcher-result.private.json'),('{"runner_exit_code":'+$taskRunnerExit+',"key_persisted":true,"file_encrypted":true}'),[Text.UTF8Encoding]::new($false))
} catch {
    $taskFailure=@{status='SAVED_KEY_LAUNCHER_STOPPED_NO_RETRY';error_type=$_.Exception.GetType().Name;exception_message_saved=$false}
    [IO.File]::WriteAllText((Join-Path $RunDirectory 'credential-launcher-error.private.json'),($taskFailure|ConvertTo-Json),[Text.UTF8Encoding]::new($false))
    exit 1
} finally {
    $env:OPENAI_API_KEY=$null
    $env:SPF_JUDGE_SAVED_KEY_INPUT=$null
    if($null -ne $taskSecure){$taskSecure.Dispose()}
    if($null -ne $taskForm){$taskForm.Dispose()}
}
