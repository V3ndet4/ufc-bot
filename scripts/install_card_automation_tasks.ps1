[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$TaskPrefix = "ufc-bot-next-card",

    [string]$ManifestPath = "",

    [string]$OddsSource = "oddsapi",

    [string]$OddsApiBookmaker = "fanduel",

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

function New-WeeklyTaskTrigger {
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")]
        [string]$DayOfWeek,

        [Parameter(Mandatory = $true)]
        [string]$At
    )

    return New-ScheduledTaskTrigger -Weekly -WeeksInterval 1 -DaysOfWeek $DayOfWeek -At ([datetime]::Parse($At))
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
$startTimeText = [string]$manifest.start_time
$startTime = if ($startTimeText) { [datetimeoffset]::Parse($startTimeText) } else { $null }
$wrapperPath = Resolve-RepoPath -PathValue "scripts\invoke_scheduled_card_task.ps1"
$runScriptPath = Resolve-RepoPath -PathValue "scripts\run_next_card.ps1"
$gradeScriptPath = Resolve-RepoPath -PathValue "scripts\grade_next_card.ps1"
$powershellPath = (Get-Command powershell.exe).Source
$logDirectory = Resolve-RepoPath -PathValue "logs\task_scheduler"
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
$userId = if ($env:USERDOMAIN) { "$($env:USERDOMAIN)\$($env:USERNAME)" } else { $env:USERNAME }
$principal = New-ScheduledTaskPrincipal -UserId $userId -LogonType Interactive -RunLevel ($(if ($RunLevelHighest) { "Highest" } else { "Limited" }))

$taskDefinitions = @(
    @{
        Name = "$TaskPrefix-full-wed-1800"
        DayOfWeek = "Wednesday"
        At = "6:00 PM"
        ScriptPath = $runScriptPath
        ScriptArguments = @()
        TaskLabel = "next_card_full_refresh"
        Description = "Full UFC next-card refresh for $eventId."
    },
    @{
        Name = "$TaskPrefix-capture-thu-1200"
        DayOfWeek = "Thursday"
        At = "12:00 PM"
        ScriptPath = $runScriptPath
        ScriptArguments = @("--skip-stats")
        TaskLabel = "next_card_capture_thu_1200"
        Description = "Light UFC next-card capture for $eventId."
    },
    @{
        Name = "$TaskPrefix-capture-thu-2000"
        DayOfWeek = "Thursday"
        At = "8:00 PM"
        ScriptPath = $runScriptPath
        ScriptArguments = @("--skip-stats")
        TaskLabel = "next_card_capture_thu_2000"
        Description = "Light UFC next-card capture for $eventId."
    },
    @{
        Name = "$TaskPrefix-capture-fri-1000"
        DayOfWeek = "Friday"
        At = "10:00 AM"
        ScriptPath = $runScriptPath
        ScriptArguments = @("--skip-stats")
        TaskLabel = "next_card_capture_fri_1000"
        Description = "Light UFC next-card capture for $eventId."
    },
    @{
        Name = "$TaskPrefix-capture-fri-1600"
        DayOfWeek = "Friday"
        At = "4:00 PM"
        ScriptPath = $runScriptPath
        ScriptArguments = @("--skip-stats")
        TaskLabel = "next_card_capture_fri_1600"
        Description = "Light UFC next-card capture for $eventId."
    },
    @{
        Name = "$TaskPrefix-capture-fri-2200"
        DayOfWeek = "Friday"
        At = "10:00 PM"
        ScriptPath = $runScriptPath
        ScriptArguments = @("--skip-stats")
        TaskLabel = "next_card_capture_fri_2200"
        Description = "Light UFC next-card capture for $eventId."
    },
    @{
        Name = "$TaskPrefix-capture-sat-1000"
        DayOfWeek = "Saturday"
        At = "10:00 AM"
        ScriptPath = $runScriptPath
        ScriptArguments = @("--skip-stats")
        TaskLabel = "next_card_capture_sat_1000"
        Description = "Light UFC next-card capture for $eventId."
    },
    @{
        Name = "$TaskPrefix-capture-sat-1300"
        DayOfWeek = "Saturday"
        At = "1:00 PM"
        ScriptPath = $runScriptPath
        ScriptArguments = @("--skip-stats")
        TaskLabel = "next_card_capture_sat_1300"
        Description = "Light UFC next-card capture for $eventId."
    },
    @{
        Name = "$TaskPrefix-capture-sat-1600"
        DayOfWeek = "Saturday"
        At = "4:00 PM"
        ScriptPath = $runScriptPath
        ScriptArguments = @("--skip-stats")
        TaskLabel = "next_card_capture_sat_1600"
        Description = "Light UFC next-card capture for $eventId."
    },
    @{
        Name = "$TaskPrefix-capture-sat-1830"
        DayOfWeek = "Saturday"
        At = "6:30 PM"
        ScriptPath = $runScriptPath
        ScriptArguments = @("--skip-stats")
        TaskLabel = "next_card_capture_sat_1830"
        Description = "Late UFC next-card capture for $eventId."
    },
    @{
        Name = "$TaskPrefix-grade-sun-0900"
        DayOfWeek = "Sunday"
        At = "9:00 AM"
        ScriptPath = $gradeScriptPath
        ScriptArguments = @()
        TaskLabel = "next_card_grade_sun_0900"
        Description = "Grade the completed UFC card for $eventId."
    }
)

Write-Host "Installing recurring scheduled tasks for current event target:"
Write-Host "Event: $eventName"
Write-Host "Event ID: $eventId"
if ($startTime) {
    Write-Host "Start time: $($startTime.ToString('yyyy-MM-dd HH:mm zzz'))"
}
Write-Host "Manifest: $resolvedManifestPath"
Write-Host "Log directory: $logDirectory"
Write-Host "Odds source: $OddsSource"
Write-Host "Bookmaker: $OddsApiBookmaker"
Write-Host

foreach ($definition in $taskDefinitions) {
    $argumentParts = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", (Quote-TaskArgument $wrapperPath),
        "-TaskLabel", (Quote-TaskArgument $definition.TaskLabel),
        "-ScriptPath", (Quote-TaskArgument $definition.ScriptPath),
        "-WorkingDirectory", (Quote-TaskArgument $RootDir),
        "-LogDirectory", (Quote-TaskArgument $logDirectory),
        "-OddsSource", (Quote-TaskArgument $OddsSource),
        "-OddsApiBookmaker", (Quote-TaskArgument $OddsApiBookmaker)
    )
    foreach ($scriptArgument in $definition.ScriptArguments) {
        $argumentParts += @("-ScriptArguments", (Quote-TaskArgument $scriptArgument))
    }

    $action = New-ScheduledTaskAction -Execute $powershellPath -Argument ($argumentParts -join " ")
    $trigger = New-WeeklyTaskTrigger -DayOfWeek $definition.DayOfWeek -At $definition.At

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
        Write-Host ("Installed {0} ({1} {2}) -> next run {3}" -f $definition.Name, $definition.DayOfWeek, $definition.At, $nextRunText)
    }
}

Write-Host
Write-Host "Task install complete. Update events\current_event.txt when the next card changes."
