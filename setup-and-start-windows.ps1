[CmdletBinding()]
param(
    [string]$Distro = "",
    [string]$WslStateDir = "",
    [ValidateSet("full", "core", "fl2va", "ref2va", "upscaler")]
    [string]$Profile = "full",
    [ValidateSet("auto", "modelscope", "hf-mirror", "huggingface")]
    [string]$Source = "auto",
    [ValidateRange(1, 65535)]
    [int]$Port = 8090,
    [switch]$AcceptModelLicense,
    [switch]$SkipModels,
    [switch]$RepairModels,
    [switch]$SkipSystemPackages,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$installArguments = @{
    Distro = $Distro
    WslStateDir = $WslStateDir
    Profile = $Profile
    Source = $Source
    AcceptModelLicense = $AcceptModelLicense
    SkipModels = $SkipModels
    RepairModels = $RepairModels
    SkipSystemPackages = $SkipSystemPackages
    DryRun = $DryRun
}
& "$PSScriptRoot\install-windows.ps1" @installArguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($DryRun) { exit 0 }
& "$PSScriptRoot\start-windows.ps1" -Distro $Distro -WslStateDir $WslStateDir -Port $Port
exit $LASTEXITCODE
