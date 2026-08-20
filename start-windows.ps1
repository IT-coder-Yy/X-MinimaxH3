[CmdletBinding()]
param(
    [string]$Distro = "",
    [string]$WslStateDir = "",
    [ValidateRange(1, 65535)]
    [int]$Port = 8090
)

$ErrorActionPreference = "Stop"
if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw "WSL was not found. Install WSL2 before starting X-MinimaxH3."
}
$installed = @(
    wsl.exe --list --quiet 2>$null |
        ForEach-Object { ($_ -replace "`0", "").Trim() } |
        Where-Object { $_ }
)
$selectedDistro = $Distro.Trim()
if (-not $selectedDistro) {
    $selectedDistro = @($installed | Where-Object { $_ -match '^Ubuntu' })[0]
}
if (-not $selectedDistro -or $installed -notcontains $selectedDistro) {
    throw "No Ubuntu WSL distribution was found. Run install-windows.ps1 first."
}
$windowsRoot = (Resolve-Path $PSScriptRoot).Path
if ($windowsRoot.Contains("'")) {
    throw "The project path cannot contain an apostrophe: $windowsRoot"
}
$linuxRoot = (
    wsl.exe -d $selectedDistro -- bash -lc "wslpath -a '$windowsRoot'" |
        Out-String
).Trim()
if (-not $linuxRoot) { throw "Could not convert the project path to WSL." }
$quotedRoot = "'" + $linuxRoot + "'"
$stateArguments = @()
if ($WslStateDir) {
    if ($WslStateDir.Contains("'")) {
        throw "The WSL state path cannot contain an apostrophe: $WslStateDir"
    }
    $stateArguments = @("--state-dir", $WslStateDir)
}
$quotedStateArguments = ($stateArguments | ForEach-Object { "'" + $_ + "'" }) -join " "
Write-Host "Starting X-MinimaxH3 at http://127.0.0.1:$Port" -ForegroundColor Cyan
wsl.exe -d $selectedDistro -- bash -lc "cd $quotedRoot && H3_SERVE_PORT=$Port ./scripts/windows-wsl.sh $quotedStateArguments start"
exit $LASTEXITCODE
