[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$TaskPrefix = "ufc-bot-fight-week-watch",

    [string]$ManifestPath = "",

    [string]$At = "10:00 PM",

    [switch]$RunLevelHighest
)

$ErrorActionPreference = "Stop"

function Resolve-RepoPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathValue
    )

    if ([string]::IsNullOrWhiteSpace($PathValue)) {
        throw "PathValue cannot be empty."
    }
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $RootDir $PathValue))
}

function Quote-TaskArgument {
    param(
        [AllowNull()]
        [string]$Value
    )

    if ($null -eq $Value) {
        return '""'
    }
    return '"' + $Value.Replace('"', '""') + '"'
}

function New-DailyTaskTrigger {
    param(
        [Parameter(Mandatory = $true)]
        [string]$At
    )

    return New-ScheduledTaskTrigger -Daily -DaysInterval 1 -At ([datetime]::Parse($At))
}

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir

if ([string]::IsNullOrWhiteSpace($ManifestPath)) {
    $currentEventPath = Resolve-RepoPath -PathValue "events\current_event.txt"
    if (-not (Test-Path $currentEventPath)) {
        throw "Current event pointer not found: $currentEventPath"
    }
    $ManifestPath = (Get-Content $currentEventPath -Raw).Trim()
}

$resolvedManifestPath = Resolve-RepoPath -PathValue $ManifestPath
if (-not (Test-Path $resolvedManifestPath)) {
    throw "Manifest not found: $resolvedManifestPath"
}

$manifest = Get-Content $resolvedManifestPath -Raw | ConvertFrom-Json
$eventId = [string]$manifest.event_id
$eventName = [string]$manifest.event_name
$wrapperPath = Resolve-RepoPath -PathValue "scripts\invoke_scheduled_card_task.ps1"
$runScriptPath = Resolve-RepoPath -PathValue "scripts\run_fight_week_watch.ps1"
$powershellPath = (Get-Command powershell.exe).Source
$logDirectory = Resolve-RepoPath -PathValue "logs\task_scheduler"
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
$userId = if ($env:USERDOMAIN) { "$($env:USERDOMAIN)\$($env:USERNAME)" } else { $env:USERNAME }
$principal = New-ScheduledTaskPrincipal -UserId $userId -LogonType Interactive -RunLevel ($(if ($RunLevelHighest) { "Highest" } else { "Limited" }))

$taskDefinitions = @(
    @{
        Name = "$TaskPrefix-night"
        At = $At
        TaskLabel = "fight_week_watch_night"
        Description = "Refresh fight-week gym and news alerts for $eventId."
    }
)

Write-Host "Installing fight-week watch scheduled tasks for current event target:"
Write-Host "Event: $eventName"
Write-Host "Event ID: $eventId"
Write-Host "Manifest: $resolvedManifestPath"
Write-Host "Log directory: $logDirectory"
Write-Host "Nightly run time: $At"
Write-Host

foreach ($definition in $taskDefinitions) {
    $argumentParts = @(
        "-WindowStyle", "Hidden",
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", (Quote-TaskArgument $wrapperPath),
        "-TaskLabel", (Quote-TaskArgument $definition.TaskLabel),
        "-ScriptPath", (Quote-TaskArgument $runScriptPath),
        "-WorkingDirectory", (Quote-TaskArgument $RootDir),
        "-LogDirectory", (Quote-TaskArgument $logDirectory),
        "-OddsSource", (Quote-TaskArgument "oddsapi"),
        "-OddsApiBookmaker", (Quote-TaskArgument "fanduel"),
        "-ScriptArguments", (Quote-TaskArgument "--quiet")
    )

    $action = New-ScheduledTaskAction -Execute $powershellPath -Argument ($argumentParts -join " ")
    $trigger = New-DailyTaskTrigger -At $definition.At

    if ($PSCmdlet.ShouldProcess($definition.Name, "Register scheduled task")) {
        Register-ScheduledTask `
            -TaskName $definition.Name `
            -Action $action `
            -Trigger $trigger `
            -Settings $settings `
            -Principal $principal `
            -Description $definition.Description `
            -Force | Out-Null

        $taskInfo = Get-ScheduledTaskInfo -TaskName $definition.Name
        $nextRunText = if ($taskInfo.NextRunTime -and $taskInfo.NextRunTime -gt [datetime]::MinValue) {
            $taskInfo.NextRunTime.ToString("yyyy-MM-dd HH:mm")
        }
        else {
            "n/a"
        }
        Write-Host ("Installed {0} ({1}) -> next run {2}" -f $definition.Name, $definition.At, $nextRunText)
    }
}

Write-Host
Write-Host "Task install complete. Update events\current_event.txt when the next card changes."
