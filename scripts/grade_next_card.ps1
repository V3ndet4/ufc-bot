$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir
. "$PSScriptRoot\_resolve_python.ps1"

$ManifestPath = (Get-Content "events\current_event.txt" -Raw).Trim()
$Manifest = Get-Content $ManifestPath -Raw | ConvertFrom-Json
$EventId = $Manifest.event_id
$Slug = $Manifest.slug
$CardDir = Join-Path "cards" $Slug

$ResultsFile = if ($env:RESULTS_FILE) { $env:RESULTS_FILE } else { Join-Path $CardDir "data\results.csv" }
$DbFile = if ($env:DB_FILE) { $env:DB_FILE } else { "data\ufc_betting.db" }
$GradedFile = if ($env:GRADED_FILE) { $env:GRADED_FILE } else { Join-Path $CardDir "reports\graded_picks.csv" }
$LearningFile = if ($env:LEARNING_FILE) { $env:LEARNING_FILE } else { Join-Path $CardDir "reports\learning_report.csv" }
$LearningSummaryFile = if ($env:LEARNING_SUMMARY_FILE) { $env:LEARNING_SUMMARY_FILE } else { Join-Path $CardDir "reports\learning_summary.csv" }
$LearningPostmortemFile = if ($env:LEARNING_POSTMORTEM_FILE) { $env:LEARNING_POSTMORTEM_FILE } else { Join-Path $CardDir "reports\learning_postmortem.csv" }
$LearningPostmortemSummaryFile = if ($env:LEARNING_POSTMORTEM_SUMMARY_FILE) { $env:LEARNING_POSTMORTEM_SUMMARY_FILE } else { Join-Path $CardDir "reports\learning_postmortem_summary.csv" }
$FilterPerformanceFile = if ($env:FILTER_PERFORMANCE_FILE) { $env:FILTER_PERFORMANCE_FILE } else { Join-Path $CardDir "reports\filter_performance.csv" }
$SelectiveModelFile = if ($env:SELECTIVE_MODEL_FILE) { $env:SELECTIVE_MODEL_FILE } else { "models\selective_clv_model.pkl" }
$AutoFetchResults = $true
if ($env:AUTO_FETCH_RESULTS -eq "0") { $AutoFetchResults = $false }
$RefreshResultsFile = $env:REFRESH_RESULTS_FILE -eq "1"

if ($AutoFetchResults -and ( -not (Test-Path $ResultsFile) -or $RefreshResultsFile)) {
    try {
        Invoke-PythonChecked -Arguments @(
            "scripts\fetch_event_results.py",
            "--manifest", $ManifestPath,
            "--output", $ResultsFile,
            "--db", $DbFile
        )
        if (-not (Test-Path $ResultsFile)) {
            throw "fetch_event_results.py completed without creating $ResultsFile"
        }
        Write-Host "Saved auto-fetched results to $ResultsFile"
    }
    catch {
        if (-not (Test-Path $ResultsFile)) {
            throw "Auto results fetch failed and no results file exists at $ResultsFile. $($_.Exception.Message)"
        }
        Write-Host "Auto results fetch failed; reusing existing results file: $ResultsFile"
        Write-Host "Fetch error: $($_.Exception.Message)"
    }
}

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
    "--output", $LearningFile,
    "--postmortem-output", $LearningPostmortemFile,
    "--postmortem-summary-output", $LearningPostmortemSummaryFile
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

try {
    Invoke-PythonChecked -Arguments @(
        "scripts\train_selective_clv_model.py",
        "--db", $DbFile,
        "--output", $SelectiveModelFile,
        "--quiet"
    )
    Write-Host "Refreshed selective CLV model at $SelectiveModelFile"
}
catch {
    Write-Host "Selective CLV model refresh skipped: $($_.Exception.Message)"
}

Write-Host "Saved graded picks to $GradedFile"
Write-Host "Saved learning report to $LearningFile"
Write-Host "Saved learning postmortem to $LearningPostmortemFile"
Write-Host "Saved learning postmortem summary to $LearningPostmortemSummaryFile"
Write-Host "Saved learning summary to $LearningSummaryFile"
Write-Host "Saved filter performance report to $FilterPerformanceFile"
