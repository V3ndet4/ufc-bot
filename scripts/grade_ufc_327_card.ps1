$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir
. "$PSScriptRoot\_resolve_python.ps1"

$ManifestPath = "events\ufc_327_prochazka_ulberg.json"
$Manifest = Get-Content $ManifestPath -Raw | ConvertFrom-Json
$Slug = $Manifest.slug
$EventId = $Manifest.event_id
$CardDir = Join-Path "cards" $Slug

$ResultsFile = if ($env:RESULTS_FILE) { $env:RESULTS_FILE } else { Join-Path $CardDir "data\results.csv" }
$DbFile = if ($env:DB_FILE) { $env:DB_FILE } else { "data\ufc_betting.db" }
$GradedFile = if ($env:GRADED_FILE) { $env:GRADED_FILE } else { Join-Path $CardDir "reports\graded_picks.csv" }
$LearningFile = if ($env:LEARNING_FILE) { $env:LEARNING_FILE } else { Join-Path $CardDir "reports\learning_report.csv" }
$LearningSummaryFile = if ($env:LEARNING_SUMMARY_FILE) { $env:LEARNING_SUMMARY_FILE } else { Join-Path $CardDir "reports\learning_summary.csv" }
$FilterPerformanceFile = if ($env:FILTER_PERFORMANCE_FILE) { $env:FILTER_PERFORMANCE_FILE } else { Join-Path $CardDir "reports\filter_performance.csv" }

if (-not (Test-Path $ResultsFile)) {
    throw "Results file not found: $ResultsFile"
}

Invoke-PythonChecked -Arguments @(
    "scripts\grade_tracked_picks.py",
    "--results", $ResultsFile,
    "--db", $DbFile,
    "--event-id", $EventId,
    "--output", $GradedFile
)

Invoke-PythonChecked -Arguments @(
    "scripts\export_learning_report.py",
    "--db", $DbFile,
    "--event-id", $EventId,
    "--output", $LearningFile
)

Invoke-PythonChecked -Arguments @(
    "scripts\export_learning_summary.py",
    "--db", $DbFile,
    "--event-id", $EventId,
    "--output", $LearningSummaryFile
)

Invoke-PythonChecked -Arguments @(
    "scripts\export_filter_performance.py",
    "--db", $DbFile,
    "--event-id", $EventId,
    "--output", $FilterPerformanceFile
)

Write-Host "Saved graded picks to $GradedFile"
Write-Host "Saved learning report to $LearningFile"
Write-Host "Saved learning summary to $LearningSummaryFile"
Write-Host "Saved filter performance report to $FilterPerformanceFile"
