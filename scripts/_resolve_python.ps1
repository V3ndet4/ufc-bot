$PythonBin = $env:PYTHON_BIN
if (-not $PythonBin) {
    if ($env:VIRTUAL_ENV -and (Test-Path (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"))) {
        $PythonBin = Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"
    }
    elseif (Test-Path ".\.venv-win\Scripts\python.exe") {
        $PythonBin = ".\.venv-win\Scripts\python.exe"
    }
    else {
        $PythonBin = "python"
    }
}

function Invoke-PythonChecked {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & $PythonBin @Arguments
    if ($LASTEXITCODE -ne 0) {
        $renderedArgs = ($Arguments | ForEach-Object {
            if ($_ -match '\s') { '"{0}"' -f $_ } else { $_ }
        }) -join ' '
        throw "Python command failed with exit code ${LASTEXITCODE}: $PythonBin $renderedArgs"
    }
}
