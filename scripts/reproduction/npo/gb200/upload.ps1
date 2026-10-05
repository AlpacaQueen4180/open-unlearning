param(
    [string]$Runai = 'C:\Program Files\Runai\runai.exe',
    [string]$Workload = 'machine-unlearning-pvc',
    [string]$Project = 'smart-mfg'
)
$ErrorActionPreference = 'Stop'
foreach ($taskFile in Get-ChildItem $PSScriptRoot -File | Where-Object { $_.Extension -in '.py', '.sh' }) {
    $taskBytes = [IO.File]::ReadAllBytes($taskFile.FullName)
    $taskMemory = [IO.MemoryStream]::new()
    $taskGzip = [IO.Compression.GZipStream]::new($taskMemory, [IO.Compression.CompressionMode]::Compress, $true)
    $taskGzip.Write($taskBytes, 0, $taskBytes.Length)
    $taskGzip.Dispose()
    $taskPayload = [Convert]::ToBase64String($taskMemory.ToArray())
    $taskMemory.Dispose()
    $taskHash = (Get-FileHash -LiteralPath $taskFile.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    $taskCode = "import pathlib,base64,gzip,hashlib; d=pathlib.Path('/data/npo-gb200-20261004/repo/scripts/reproduction/npo/gb200'); d.mkdir(parents=True,exist_ok=True); b=gzip.decompress(base64.b64decode('$taskPayload')); assert hashlib.sha256(b).hexdigest()=='$taskHash'; p=d/'$($taskFile.Name)'; p.write_bytes(b); print(p.name,hashlib.sha256(p.read_bytes()).hexdigest())"
    for ($taskAttempt=0; $taskAttempt -lt 2; $taskAttempt++) {
        & $Runai workspace exec $Workload -p $Project -- python -c $taskCode
        if ($LASTEXITCODE -eq 0) { break }
        Start-Sleep -Seconds 2
    }
    if ($LASTEXITCODE -ne 0) { throw "Upload failed for $($taskFile.Name)" }
}
