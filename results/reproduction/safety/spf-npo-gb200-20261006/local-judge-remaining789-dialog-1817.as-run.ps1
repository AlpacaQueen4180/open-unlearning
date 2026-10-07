param(
    [Parameter(Mandatory=$true)][string]$RunDirectory,
    [Parameter(Mandatory=$true)][string]$PythonExecutable,
    [Parameter(Mandatory=$true)][string]$Runner,
    [Parameter(Mandatory=$true)][string]$SourceSha256,
    [Parameter(Mandatory=$true)][string]$PlanSha256
)
$ErrorActionPreference = 'Stop'
$env:CUDA_VISIBLE_DEVICES = ''
$env:OPENAI_API_KEY = $null
$env:OPENAI_ORG_ID = $null
$env:OPENAI_PROJECT_ID = $null
$env:SPF_JUDGE_NEW_KEY_INPUT = $null
$arguments = @($Runner, '--run', $RunDirectory, '--source-sha256', $SourceSha256, '--plan-sha256', $PlanSha256)
$preflight = & $PythonExecutable @arguments --preflight 2>&1
if ($LASTEXITCODE -ne 0) { throw 'Local Judge preflight failed before credential input.' }
[System.IO.File]::WriteAllText((Join-Path $RunDirectory 'dialog-preflight.private.json'), ($preflight -join "`n"), [System.Text.UTF8Encoding]::new($false))
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$form = New-Object System.Windows.Forms.Form
$form.Text = 'SPF Judge — 新專用 API key'
$form.Size = New-Object System.Drawing.Size(610, 310)
$form.StartPosition = 'CenterScreen'
$form.FormBorderStyle = 'FixedDialog'
$form.MaximizeBox = $false
$form.ShowInTaskbar = $true
$form.TopMost = $true
$label = New-Object System.Windows.Forms.Label
$label.Location = New-Object System.Drawing.Point(20, 20)
$label.Size = New-Object System.Drawing.Size(560, 125)
$label.Text = "請貼上新建立的 Judge 專用 OpenAI API key。`n`n本機將把 SPF safety 650 筆與 conversation 142 筆 prompts/responses`n送到 https://api.openai.com/v1，使用 gpt-5.6-terra、medium、`n前 3 筆已成功。只接續 789 筆，每筆最多 4096 output tokens、零重試。`nkey 只留在程序記憶體，不寫入檔案、聊天或 Git。"
$box = New-Object System.Windows.Forms.TextBox
$box.Location = New-Object System.Drawing.Point(20, 155)
$box.Size = New-Object System.Drawing.Size(560, 28)
$box.UseSystemPasswordChar = $true
$button = New-Object System.Windows.Forms.Button
$button.Location = New-Object System.Drawing.Point(350, 205)
$button.Size = New-Object System.Drawing.Size(230, 34)
$button.Text = '用同一專用 key 接續未完成評分'
$button.Add_Click({
    if ([string]::IsNullOrWhiteSpace($box.Text)) { return }
    $form.DialogResult = [System.Windows.Forms.DialogResult]::OK
    $form.Close()
})
$form.Controls.AddRange(@($label, $box, $button))
$form.AcceptButton = $button
[System.IO.File]::WriteAllText((Join-Path $RunDirectory 'credential-input-status.private.json'), '{"status":"CREATING_LOCAL_MASKED_DIALOG","api_calls_executed":0}', [System.Text.UTF8Encoding]::new($false))
$form.Add_Shown({
    $form.Activate()
    $form.BringToFront()
    $box.Focus()
    $taskVisible = @{status='WAITING_FOR_NEW_KEY_LOCAL_MASKED_DIALOG';api_calls_executed=0;shown_event_received=$true;form_visible=$form.Visible;show_in_taskbar=$form.ShowInTaskbar;topmost=$form.TopMost}
    [System.IO.File]::WriteAllText((Join-Path $RunDirectory 'credential-input-status.private.json'), ($taskVisible | ConvertTo-Json), [System.Text.UTF8Encoding]::new($false))
})
try {
    $result = $form.ShowDialog()
    if ($result -ne [System.Windows.Forms.DialogResult]::OK) {
        [System.IO.File]::WriteAllText((Join-Path $RunDirectory 'credential-input-status.private.json'), '{"status":"INPUT_CLOSED_WITHOUT_KEY","api_calls_executed":0}', [System.Text.UTF8Encoding]::new($false))
        exit 0
    }
    $env:OPENAI_API_KEY = $box.Text.Trim()
    $box.Clear()
    $env:SPF_JUDGE_NEW_KEY_INPUT = 'local_masked_dialog'
    [System.IO.File]::WriteAllText((Join-Path $RunDirectory 'credential-input-status.private.json'), '{"status":"NEW_KEY_SUPPLIED_TO_CHILD_MEMORY_ONLY"}', [System.Text.UTF8Encoding]::new($false))
    & $PythonExecutable @arguments > (Join-Path $RunDirectory 'runner-output.private.txt')
    $runnerExit = $LASTEXITCODE
    [System.IO.File]::WriteAllText((Join-Path $RunDirectory 'launcher-result.private.json'), ('{"runner_exit_code":' + $runnerExit + ',"key_persisted":false}'), [System.Text.UTF8Encoding]::new($false))
} finally {
    $env:OPENAI_API_KEY = $null
    $env:SPF_JUDGE_NEW_KEY_INPUT = $null
    $box.Clear()
    $form.Dispose()
}
