$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir
. "$PSScriptRoot\_resolve_python.ps1"

Invoke-PythonChecked -Arguments (@("scripts\set_current_event.py") + $args)
