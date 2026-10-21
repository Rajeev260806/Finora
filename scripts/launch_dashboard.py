"""
Finora Dashboard Launcher
Starts the Uvicorn server and automatically opens the browser dashboard.
"""

import os
import sys
import webbrowser
import uvicorn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

if __name__ == "__main__":
    url = "http://127.0.0.1:8000"
    print(f"Opening Finora Dashboard at: {url}")
    webbrowser.open(url)
    
    # Run uvicorn
    uvicorn.run("server:app", app_dir=FRONTEND_DIR, host="127.0.0.1", port=8000, reload=True)
