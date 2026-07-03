$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:PYTHONPATH = "src"

$LocalPython = Join-Path $Root ".venv\Scripts\python.exe"
$WorkspacePython = Join-Path (Split-Path -Parent $Root) "cupcast-mini-llm\.venv\Scripts\python.exe"

if (Test-Path $LocalPython) {
    & $LocalPython scripts/run_api_dev.py --host 127.0.0.1 --port 8000
} elseif (Test-Path $WorkspacePython) {
    & $WorkspacePython scripts/run_api_dev.py --host 127.0.0.1 --port 8000
} else {
    & "python" scripts/run_api_dev.py --host 127.0.0.1 --port 8000
}
