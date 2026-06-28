$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

& "$ProjectRoot\.venv\Scripts\python.exe" -m streamlit run "$ProjectRoot\app\streamlit_app.py" `
  --server.address localhost `
  --server.port 8501 `
  --server.headless true `
  --logger.level error
