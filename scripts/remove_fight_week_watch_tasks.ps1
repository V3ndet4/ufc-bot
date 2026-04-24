[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$TaskPrefix = "ufc-bot-fight-week-watch"
)

$ErrorActionPreference = "Stop"

$tasks = Get-ScheduledTask | Where-Object { $_.TaskName -like "$TaskPrefix*" } | Sort-Object TaskName

if (-not $tasks) {
    Write-Host "No scheduled tasks found with prefix '$TaskPrefix'."
    return
}

foreach ($task in $tasks) {
    if ($PSCmdlet.ShouldProcess($task.TaskName, "Unregister scheduled task")) {
        Unregister-ScheduledTask -TaskName $task.TaskName -Confirm:$false
        Write-Host "Removed $($task.TaskName)"
    }
}
