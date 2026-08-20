[CmdletBinding()]
param(
    [string]$Distro = "",
    [string]$WslStateDir = "",
    [ValidateSet("full", "core", "fl2va", "ref2va", "upscaler")]
    [string]$Profile = "full",
    [ValidateSet("auto", "modelscope", "hf-mirror", "huggingface")]
    [string]$Source = "auto",
    [switch]$AcceptModelLicense,
    [switch]$SkipModels,
    [switch]$RepairModels,
    [switch]$SkipSystemPackages,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw "WSL was not found. Run this in an Administrator PowerShell first: wsl --install"
}

$installed = @(
    wsl.exe --list --quiet 2>$null |
        ForEach-Object { ($_ -replace "`0", "").Trim() } |
        Where-Object { $_ }
)
$selectedDistro = $Distro.Trim()
if (-not $selectedDistro) {
    $selectedDistro = @($installed | Where-Object { $_ -match '^Ubuntu' })[0]
    if (-not $selectedDistro) { $selectedDistro = "Ubuntu-22.04" }
}
if ($installed -notcontains $selectedDistro) {
    Write-Host "Installing $selectedDistro. Windows may request a restart; run this script again afterward." -ForegroundColor Yellow
    wsl.exe --install -d $selectedDistro
    exit $LASTEXITCODE
}

$windowsRoot = (Resolve-Path $PSScriptRoot).Path
if ($windowsRoot.Contains("'")) {
    throw "The project path cannot contain an apostrophe: $windowsRoot"
}
$linuxRoot = (
    wsl.exe -d $selectedDistro -- bash -lc "wslpath -a '$windowsRoot'" |
        Out-String
).Trim()
if (-not $linuxRoot) {
    throw "Could not convert the project path to a WSL path: $windowsRoot"
}

$quotedRoot = "'" + $linuxRoot + "'"
$stateArguments = @()
if ($WslStateDir) {
    if ($WslStateDir.Contains("'")) {
        throw "The WSL state path cannot contain an apostrophe: $WslStateDir"
    }
    $stateArguments = @("--state-dir", $WslStateDir)
}
$arguments = @("--profile", $Profile, "--source", $Source)
if ($AcceptModelLicense) { $arguments += "--accept-model-license" }
if ($SkipModels) { $arguments += "--skip-models" }
if ($RepairModels) { $arguments += "--repair-models" }
if ($SkipSystemPackages) { $arguments += "--skip-system-packages" }
if ($DryRun) { $arguments += "--dry-run" }
$quotedArguments = ($arguments | ForEach-Object { "'" + $_ + "'" }) -join " "
$quotedStateArguments = ($stateArguments | ForEach-Object { "'" + $_ + "'" }) -join " "

Write-Host "Installing X-MinimaxH3 inside WSL2/$selectedDistro. Runtime and models use WSL-native storage; project files remain in the current Windows folder." -ForegroundColor Cyan
wsl.exe -d $selectedDistro -- bash -lc "cd $quotedRoot && ./scripts/windows-wsl.sh $quotedStateArguments install $quotedArguments"
if ($LASTEXITCODE -ne 0) {
    throw "Installation failed. WSL exit code: $LASTEXITCODE"
}
Write-Host "Installation complete. Run .\start-windows.ps1 to start the service." -ForegroundColor Green
