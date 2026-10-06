<#
Runs one read-only Python snapshot as a compressed short command and validates framed JSON.
At most one retry is permitted for transport failure. Do not use for launches
or other mutations: a lost response does not prove a remote action failed.
#>
[CmdletBinding(DefaultParameterSetName='Code')]
param(
    [Parameter(Mandatory,ParameterSetName='Code')][string]$PythonCode,
    [Parameter(Mandatory,ParameterSetName='File')][string]$PythonFile,
    [string]$Runai = 'C:\Program Files\Runai\runai.exe',
    [string]$Workload = 'machine-unlearning-pvc',
    [string]$Project = 'smart-mfg',
    [string]$Pod = '',
    [ValidateRange(1,2)][int]$MaxAttempts = 2,
    [string]$SavePath = ''
)
$ErrorActionPreference = 'Stop'
if ($PSCmdlet.ParameterSetName -eq 'File') {
    $PythonCode = [IO.File]::ReadAllText((Resolve-Path -LiteralPath $PythonFile).Path)
}
$taskMarker = [Guid]::NewGuid().ToString('N')
$taskBegin = 'RUNAI_JSON_BEGIN_' + $taskMarker
$taskEnd = 'RUNAI_JSON_END_' + $taskMarker
$taskBytes = [Text.Encoding]::UTF8.GetBytes($PythonCode)
$taskMemory = [IO.MemoryStream]::new()
$taskGzip = [IO.Compression.GZipStream]::new($taskMemory,[IO.Compression.CompressionMode]::Compress,$true)
$taskGzip.Write($taskBytes,0,$taskBytes.Length)
$taskGzip.Dispose()
$taskPayload = [Convert]::ToBase64String($taskMemory.ToArray())
$taskMemory.Dispose()
$taskProgram = "import base64,gzip; print('$taskBegin', flush=True); exec(compile(gzip.decompress(base64.b64decode('$taskPayload')),'<readonly_snapshot>','exec')); print('$taskEnd', flush=True)"
if ([Uri]::EscapeDataString($taskProgram).Length -gt 6000) {
    throw 'Snapshot program exceeds the short-command limit. Upload a persistent read-only script separately and invoke it by path.'
}
$taskArgs = @('workspace','exec',$Workload,'-p',$Project)
if ($Pod) { $taskArgs += @('--pod',$Pod) }
$taskArgs += @('--','python','-c',$taskProgram)
$taskTransportPattern = '(?i)TLS handshake timeout|unexpected EOF|\bEOF\b|websocket:|connection reset|i/o timeout|context deadline exceeded|Service Unavailable|\b502\b|\b504\b'
for ($taskAttempt = 1; $taskAttempt -le $MaxAttempts; $taskAttempt++) {
    # Windows PowerShell can turn native stderr into an ErrorRecord. Capture it
    # without aborting before the bounded transport-retry decision below.
    $taskPreviousErrorAction = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $taskRaw = & $Runai @taskArgs 2>&1
        $taskExit = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $taskPreviousErrorAction
    }
    $taskText = ($taskRaw | ForEach-Object { $_.ToString() }) -join "`n"
    $taskStart = $taskText.IndexOf($taskBegin, [StringComparison]::Ordinal)
    # A process snapshot can include this command and its marker in ps output.
    # The emitted closing frame is the final occurrence, not the embedded one.
    $taskStop = $taskText.LastIndexOf($taskEnd, [StringComparison]::Ordinal)
    $taskComplete = $taskExit -eq 0 -and $taskStart -ge 0 -and $taskStop -gt $taskStart
    if ($taskComplete) {
        $taskJson = $taskText.Substring($taskStart + $taskBegin.Length, $taskStop - $taskStart - $taskBegin.Length).Trim()
        try { $taskValue = ConvertFrom-Json -InputObject $taskJson -ErrorAction Stop }
        catch { throw 'Remote output had complete framing but invalid JSON; no automatic retry.' }
        if ($null -eq $taskValue) { throw 'Remote output was JSON null; no automatic retry.' }
        if ($SavePath) {
            $taskDestination = [IO.Path]::GetFullPath($SavePath)
            [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($taskDestination)) | Out-Null
            [IO.File]::WriteAllText($taskDestination, $taskJson, [Text.UTF8Encoding]::new($false))
        }
        return $taskValue
    }
    if ($SavePath) {
        # Preserve failed transport/CLI output privately without exposing it in chat.
        $taskFailurePath = [IO.Path]::GetFullPath($SavePath + ".failed-attempt-$taskAttempt.txt")
        [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($taskFailurePath)) | Out-Null
        [IO.File]::WriteAllText($taskFailurePath, $taskText, [Text.UTF8Encoding]::new($false))
    }
    if ($taskText -match 'Traceback \(most recent call last\)') {
        throw "Remote Python failed (CLI exit $taskExit); no transport retry."
    }
    $taskRetryable = $taskText -match $taskTransportPattern -or ($taskExit -eq 0 -and -not $taskComplete)
    if (-not $taskRetryable -or $taskAttempt -eq $MaxAttempts) {
        throw "RunAI snapshot response incomplete or failed (CLI exit $taskExit, attempts $taskAttempt). No remote mutation was requested."
    }
    Write-Verbose 'Transient transport failure or incomplete response; retrying the read-only snapshot once.'
}
