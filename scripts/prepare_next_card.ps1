$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir
. "$PSScriptRoot\_resolve_python.ps1"

$ManifestPath = (Get-Content "events\current_event.txt" -Raw).Trim()
Invoke-PythonChecked -Arguments @("scripts\prepare_event.py", "--manifest", $ManifestPath)
