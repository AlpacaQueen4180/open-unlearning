# Windows CurrentUser DPAPI storage for the dedicated local SPF Judge key.
# Functions return SecureString only; no key values or ciphertext are logged.
Set-StrictMode -Version Latest
function Assert-SpfJudgeKeyPath {
    param([string]$Path,[string]$PrivateRoot)
    if(-not $IsWindows){throw 'Windows CurrentUser credential storage required'}
    $taskRoot=[IO.Path]::GetFullPath($PrivateRoot)
    $taskExpected=[IO.Path]::GetFullPath((Join-Path $taskRoot 'local-judge-credentials/openai-judge-key.dpapi'))
    if([IO.Path]::GetFullPath($Path) -ne $taskExpected){throw 'Dedicated private credential path required'}
    foreach($taskItemPath in @($taskRoot,(Split-Path -Parent $taskExpected),$taskExpected)){
        if(Test-Path -LiteralPath $taskItemPath){
            $taskItem=Get-Item -LiteralPath $taskItemPath -Force
            if(($taskItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0){throw 'Credential reparse points are not allowed'}
        }
    }
}
function Set-SpfJudgePrivateAcl {
    param([string]$Path,[bool]$Directory)
    $taskSid=[Security.Principal.WindowsIdentity]::GetCurrent().User
    if($Directory){
        $taskAcl=[Security.AccessControl.DirectorySecurity]::new()
        $taskRule=[Security.AccessControl.FileSystemAccessRule]::new($taskSid,'FullControl','ContainerInherit, ObjectInherit','None','Allow')
    }else{
        $taskAcl=[Security.AccessControl.FileSecurity]::new()
        $taskRule=[Security.AccessControl.FileSystemAccessRule]::new($taskSid,'FullControl','Allow')
    }
    $taskAcl.SetOwner($taskSid)
    $taskAcl.SetAccessRuleProtection($true,$false)
    $taskAcl.AddAccessRule($taskRule)
    Set-Acl -LiteralPath $Path -AclObject $taskAcl
}
function Assert-SpfJudgePrivateAcl {
    param([string]$Path)
    $taskSid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    $taskAcl=Get-Acl -LiteralPath $Path
    $taskRules=@($taskAcl.GetAccessRules($true,$true,[Security.Principal.SecurityIdentifier]))
    if(-not $taskAcl.AreAccessRulesProtected -or $taskAcl.GetOwner([Security.Principal.SecurityIdentifier]).Value -ne $taskSid -or $taskRules.Count -ne 1){throw 'Current-user-only protected credential ACL required'}
    if($taskRules[0].IdentityReference.Value -ne $taskSid -or $taskRules[0].AccessControlType -ne 'Allow' -or $taskRules[0].FileSystemRights -ne 'FullControl'){throw 'Unexpected credential access rule'}
}
function Save-SpfJudgeKey {
    param([string]$Path,[string]$PrivateRoot,[Security.SecureString]$Key)
    Assert-SpfJudgeKeyPath -Path $Path -PrivateRoot $PrivateRoot
    if(-not $Key -or $Key.Length -eq 0){throw 'A nonempty dedicated key is required'}
    if(Test-Path -LiteralPath $Path){throw 'Existing saved credential will not be overwritten'}
    $taskDirectory=Split-Path -Parent $Path
    if(-not (Test-Path -LiteralPath $taskDirectory)){[IO.Directory]::CreateDirectory($taskDirectory)|Out-Null}
    Set-SpfJudgePrivateAcl -Path $taskDirectory -Directory $true
    Assert-SpfJudgePrivateAcl -Path $taskDirectory
    $taskCipher=ConvertFrom-SecureString -SecureString $Key -ErrorAction Stop
    $taskBytes=[Text.UTF8Encoding]::new($false).GetBytes($taskCipher)
    $taskStream=[IO.File]::Open($Path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
    try{$taskStream.Write($taskBytes,0,$taskBytes.Length);$taskStream.Flush($true)}finally{$taskStream.Dispose()}
    Set-SpfJudgePrivateAcl -Path $Path -Directory $false
    Assert-SpfJudgePrivateAcl -Path $Path
    $taskCipher=$null;$taskBytes=$null
}
function Read-SpfJudgeKey {
    param([string]$Path,[string]$PrivateRoot)
    Assert-SpfJudgeKeyPath -Path $Path -PrivateRoot $PrivateRoot
    if(-not (Test-Path -LiteralPath $Path -PathType Leaf)){throw 'Dedicated saved key is not present'}
    Assert-SpfJudgePrivateAcl -Path (Split-Path -Parent $Path)
    Assert-SpfJudgePrivateAcl -Path $Path
    try{$taskSecure=ConvertTo-SecureString -String ([IO.File]::ReadAllText($Path)) -ErrorAction Stop}
    catch{throw 'Saved key cannot be decrypted by this Windows account'}
    if($taskSecure.Length -eq 0){$taskSecure.Dispose();throw 'Saved key is empty'}
    return $taskSecure
}
