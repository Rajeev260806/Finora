@echo off
echo ========================================================
echo Starting Finora Quantitative AI Model Dashboard...
echo ========================================================
cd /d "%~dp0"

echo Launching Uvicorn Server on http://127.0.0.1:8000 ...
start "" http://127.0.0.1:8000
".\venv\Scripts\python.exe" -m uvicorn server:app --app-dir frontend --host 127.0.0.1 --port 8000 --reload
pause
