[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$TaskLabel,

    [Parameter(Mandatory = $true)]
    [string]$ScriptPath,

    [string[]]$ScriptArguments = @(),

    [string]$WorkingDirectory = (Split-Path -Parent $PSScriptRoot),

    [string]$LogDirectory = (Join-Path (Split-Path -Parent $PSScriptRoot) "logs\task_scheduler"),

    [string]$OddsSource = "oddsapi",

    [string]$OddsApiBookmaker = "fanduel"
)

$ErrorActionPreference = "Stop"

function Resolve-TaskPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathValue,

        [Parameter(Mandatory = $true)]
        [string]$BaseDirectory
    )

    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $BaseDirectory $PathValue))
}

$resolvedWorkingDirectory = [System.IO.Path]::GetFullPath($WorkingDirectory)
$resolvedScriptPath = Resolve-TaskPath -PathValue $ScriptPath -BaseDirectory $resolvedWorkingDirectory
$resolvedLogDirectory = Resolve-TaskPath -PathValue $LogDirectory -BaseDirectory $resolvedWorkingDirectory
$taskSlug = (($TaskLabel -replace "[^A-Za-z0-9._-]", "_").Trim("_"))
if ([string]::IsNullOrWhiteSpace($taskSlug)) {
    $taskSlug = "scheduled_task"
}
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $resolvedLogDirectory "$taskSlug`_$timestamp.log"
$scriptExitCode = 0

New-Item -ItemType Directory -Path $resolvedLogDirectory -Force | Out-Null
Start-Transcript -Path $logPath -Force | Out-Null

try {
    Set-Location $resolvedWorkingDirectory
    $env:ODDS_SOURCE = $OddsSource
    $env:ODDS_API_BOOKMAKER = $OddsApiBookmaker

    Write-Host "[$(Get-Date -Format s)] Task: $TaskLabel"
    Write-Host "Working directory: $resolvedWorkingDirectory"
    Write-Host "Script: $resolvedScriptPath"
    if ($ScriptArguments.Count -gt 0) {
        Write-Host ("Arguments: " + ($ScriptArguments -join " "))
    }

    & $resolvedScriptPath @ScriptArguments

    if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
        $scriptExitCode = [int]$LASTEXITCODE
        throw "Task script exited with code $scriptExitCode."
    }
}
catch {
    if ($scriptExitCode -eq 0) {
        $scriptExitCode = 1
    }
    Write-Error $_
}
finally {
    try {
        Stop-Transcript | Out-Null
    }
    catch {
    }
}

if ($scriptExitCode -ne 0) {
    exit $scriptExitCode
}
