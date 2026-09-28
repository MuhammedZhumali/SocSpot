$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Create the project .venv and install requirements-ml.txt first; see docs/model-and-search.md.'
}
Set-Location -LiteralPath $projectRoot
& $pythonPath -m uvicorn socspot.app:app --host 127.0.0.1 --port 8767
