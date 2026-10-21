"""
Finora Root Server Entrypoint: Runs the FastAPI server that serves both
the backend quantitative API and the production React frontend build.
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from frontend.server import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting Finora Fullstack Server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
