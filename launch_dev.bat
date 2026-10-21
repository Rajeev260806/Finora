@echo off
echo ========================================================
echo Starting Finora in Dual Development Mode (HMR + API)
echo ========================================================
cd /d "%~dp0"

echo [1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "Finora Backend API" cmd /k ".\venv\Scripts\python.exe -m uvicorn server:app --app-dir frontend --host 127.0.0.1 --port 8000 --reload"

echo [2/2] Starting React Vite Dev Server on http://localhost:5173 ...
cd frontend
start "Finora React Frontend" cmd /k "npm.cmd run dev"

timeout /t 3 /nobreak >nul
start "" http://localhost:5173
echo Both servers launched!
