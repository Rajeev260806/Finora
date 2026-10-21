@echo off
echo ========================================================
echo Starting Finora Quantitative AI React + FastAPI System
echo ========================================================
cd /d "%~dp0"

if not exist "frontend\dist\index.html" (
    echo Building React production bundle...
    cd frontend
    call npm.cmd run build
    cd ..
)

echo Launching Fullstack Uvicorn Server on http://127.0.0.1:8000 ...
start "" http://127.0.0.1:8000
".\venv\Scripts\python.exe" -m uvicorn server:app --app-dir frontend --host 127.0.0.1 --port 8000 --reload
pause
