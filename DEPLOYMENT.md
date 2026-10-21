# Finora Fullstack Deployment & Hosting Guide

This guide details how to run, build, and deploy the Finora Quantitative AI Dashboard (React Frontend + FastAPI Backend).

---

## Architecture Overview

- **Frontend**: React 19 + Vite + Chart.js + HTML5 Canvas Force-Directed Network Simulation.
- **Backend**: FastAPI + Uvicorn + NumPy + Pandas running quantitative portfolio backtests & serving cached model evaluations.
- **Deployment Strategy**:
  1. **Unified Hosting (Recommended)**: The built React SPA in `frontend/dist` is directly served by the FastAPI backend. A single service / container / server hosts both on the same port!
  2. **Decoupled Hosting**: Host the React frontend on Vercel / Netlify and the FastAPI backend on Render / Railway / Fly.io.

---

## 1. Local Development

### Option A: Dual-Server Hot Reload (Recommended for Development)
Run both the Vite dev server (with HMR) and FastAPI backend concurrently:
- **Windows**: Double click [`launch_dev.bat`](file:///d:/Projects/ML_Package/launch_dev.bat)
- **Manual Commands**:
  ```bash
  # Terminal 1: Start FastAPI Backend
  python -m uvicorn server:app --app-dir frontend --host 127.0.0.1 --port 8000 --reload

  # Terminal 2: Start React Frontend
  cd frontend
  npm run dev
  ```
  The Vite dev server runs on `http://localhost:5173` and automatically proxies `/api` calls to `http://127.0.0.1:8000`.

### Option B: Unified Production Preview
Test the production React build served by FastAPI:
- **Windows**: Double click [`launch_frontend.bat`](file:///d:/Projects/ML_Package/launch_frontend.bat)
- Access at `http://127.0.0.1:8000`

---

## 2. Unified Cloud Hosting (Single Service)

Because the FastAPI server is configured to serve the React production bundle (`frontend/dist`), you can deploy both frontend and backend as a single service.

### Deploying to Render
1. Connect your GitHub repository to [Render](https://render.com).
2. Create a new **Web Service**.
3. Choose **Docker** as the environment (or use [`render.yaml`](file:///d:/Projects/ML_Package/render.yaml)).
4. Render will automatically build the React app and deploy the container on your assigned URL.

### Deploying to Railway
1. Create a new project on [Railway](https://railway.app).
2. Connect your GitHub repository.
3. Railway automatically detects [`Dockerfile`](file:///d:/Projects/ML_Package/Dockerfile) and provisions the container.

### Deploying with Docker / Docker Compose
Build and run the unified container locally or on any VPS (AWS EC2, DigitalOcean, Hetzner):
```bash
docker compose up --build
```
Access at `http://localhost:8000`.

---

## 3. Decoupled Hosting (Vercel/Netlify + Cloud Backend)

If you prefer hosting the React frontend on a CDN like Vercel or Netlify:

1. **Deploy Backend**:
   - Deploy `server.py` to Render, Railway, or Fly.io.
   - Note your backend URL: e.g. `https://finora-api.onrender.com`.

2. **Deploy Frontend (Vercel / Netlify)**:
   - Root Directory: `frontend`
   - Build Command: `npm run build`
   - Output Directory: `dist`
   - Environment Variable:
     - `VITE_API_URL`: `https://finora-api.onrender.com`
   - The frontend API client automatically prefixes API requests with `VITE_API_URL` and CORS is already enabled on the backend.

---

## 4. Verification Checklist

- [x] React 19 application built with Vite
- [x] Executive Cockpit with interactive KPI cards and master line charts
- [x] Model Ablations & 5-seed cross-validation table
- [x] Interactive GNN stock correlation network with canvas physics and threshold slider
- [x] Dynamic sector allocation streamgraph and time scrubber
- [x] Transaction friction decay curve and cost robustness summary
- [x] What-If portfolio simulator with live `/api/simulate` integration and offline fallback
- [x] Scientific methodology report with mathematical formulations
- [x] CSV and JSON data export
- [x] Multi-stage `Dockerfile` and `docker-compose.yml`
- [x] Unified SPA routing and static asset serving in `server.py`
