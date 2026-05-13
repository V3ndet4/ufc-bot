$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir
. "$PSScriptRoot\_resolve_python.ps1"

$ManifestPath = (Get-Content "events\current_event.txt" -Raw).Trim()
$Manifest = Get-Content $ManifestPath -Raw | ConvertFrom-Json
$OddsApiBookmaker = if ($env:ODDS_API_BOOKMAKER) { $env:ODDS_API_BOOKMAKER } else { "fanduel" }
$DefaultOddsPath = Join-Path (Join-Path (Join-Path "cards" $Manifest.slug) "data") "oddsapi_odds.csv"
$DefaultPropOddsPath = Join-Path (Join-Path (Join-Path "cards" $Manifest.slug) "data") "modeled_market_odds.csv"
$HasExplicitOdds = $args -contains "--odds"
$HasExplicitProps = $args -contains "--include-props"

if (-not $HasExplicitOdds -and -not (Test-Path $DefaultOddsPath)) {
    Write-Host "No odds CSV found: $DefaultOddsPath"
    Write-Host "run_core_card.ps1 requires live moneyline odds before it can build a core board."
    Write-Host ""
    Write-Host "For the no-odds card view, run:"
    Write-Host ".\.venv-win\Scripts\python.exe scripts\print_card_preview.py --manifest $ManifestPath"
    exit 0
}

$CoreArgs = @(
    "scripts\run_core_scan.py",
    "--manifest", $ManifestPath,
    "--book", $OddsApiBookmaker
)

if (-not $HasExplicitProps -and (Test-Path $DefaultPropOddsPath)) {
    $CoreArgs += "--include-props"
}

Invoke-PythonChecked -Arguments ($CoreArgs + $args)
