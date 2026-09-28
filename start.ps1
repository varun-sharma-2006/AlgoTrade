# Starts the Algo Trade Simulator: FastAPI backend on :8000 and Vite frontend on :5173.
# Installs dependencies on first run. Usage:  powershell -ExecutionPolicy Bypass -File .\start.ps1
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$venvPython = Join-Path $root "backend\.venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating Python virtual environment..."
    python -m venv (Join-Path $root "backend\.venv")
    & $venvPython -m pip install -r (Join-Path $root "backend\requirements.txt")
}

if (-not (Test-Path (Join-Path $root "node_modules"))) {
    Write-Host "Installing frontend dependencies..."
    Push-Location $root; npm install; Pop-Location
}

$backendEnv = Join-Path $root "backend\.env"
if (-not (Test-Path $backendEnv)) {
    "USE_IN_MEMORY_DB=true`nENABLE_DEV_ENDPOINTS=true" | Set-Content $backendEnv
}
$clientEnv = Join-Path $root "client\.env"
if (-not (Test-Path $clientEnv)) {
    "VITE_ENABLE_LOGIN_BYPASS=true" | Set-Content $clientEnv
}

# Free the ports in case a previous run is still holding them.
foreach ($port in 8000, 5173) {
    Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
}

Start-Process powershell -WorkingDirectory $root -ArgumentList "-NoExit", "-Command", "& '$venvPython' -m uvicorn backend.main:app --port 8000"
Start-Process powershell -WorkingDirectory $root -ArgumentList "-NoExit", "-Command", "npm run dev"

Write-Host "Backend:  http://localhost:8000/docs"
Write-Host "Frontend: http://localhost:5173"
