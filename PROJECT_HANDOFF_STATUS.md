# Finora AI Quantitative Engine — Project Status & Handoff Summary

**Project**: Finora Quantitative AI Portfolio Management & Model Intelligence  
**Current Date**: October 21, 2026  
**Status**: **100% Complete, Built, and Verified in Browser.**

---

## 1. Summary of Completed Work

### A. Minimalist Fintech Landing Page (`LandingPage.jsx`)
- **Project Title**: **Finora** displayed with modern clean typography and Champagne Gold (`#fbbf24`) text gradient and warm amber glow.
- **Crisp 2-Line Project Description**:
  > *Graph-Neural-Network Powered Portfolio Intelligence & Reinforcement Learning.*  
  > *Dynamic relational asset modeling engineered for out-of-sample risk and alpha optimization.*
- **Color Theme**: **Champagne Gold & Warm Amber** applied universally across the Landing Page and the entire application:
  - Champagne Gold: `#fbbf24` / `#fde68a`
  - Warm Amber: `#f59e0b` / `#d97706`
  - Deep Bronze: `#b45309` / `#78350f`
- **Transparent Vector Clipart Background**:
  - Implemented entirely in scalable vector SVG (`landing-bg-svg`) behind the content (`pointer-events: none; z-index: 1`).
  - Low opacity (10%–20%), hollow outlined shapes with minimal fills, seamlessly blending into the dark obsidian navy background (`#07080a` / `#0d0f14`).
  - **4 Thematic Clipart Groups**:
    1. *Stocks & Market Trends* (Top-Right): Outlined candlesticks, coordinate axes, ascending price spline, ticker badges.
    2. *Relationships & Network Graph* (Top-Left): Interconnected asset nodes, correlation link edges, orbital relationship rings.
    3. *Cash & Assets* (Bottom-Left): Hollow layered banknotes, dollar token shields, coin stack rings, liquidity flow vector.
    4. *Neural Architecture & GNN* (Bottom-Right): Input feature layer, relational graph hidden nodes, portfolio allocation output nodes.
- **Navigation & Call to Action**:
  - Primary button: **"Launch Dashboard"** with Champagne Gold gradient border and hover animation.
  - Clicking redirects smoothly to the Finora Quantitative AI dashboard.
  - Top navigation bar in Dashboard includes a **"Home"** button to return to the landing page at any time.

### B. React Frontend Migration & Full Architecture
- Successfully migrated legacy vanilla HTML/CSS/JavaScript frontend to a modular **React 19 + Vite** single-page application located in `frontend/`.
- Full production bundle compiled into `frontend/dist/` (`npm.cmd run build` finishes in ~800ms with zero errors).
- Design system preserved: Dark neon cyber-glass aesthetic (`Outfit` & `Inter` fonts, glassmorphism cards, subtle glowing cyan/emerald/purple accents, responsive grid).

### B. Tabs & Views
The application contains 6 core views accessible via the top navigation bar (`TabsNav.jsx`):
1. **Executive Cockpit (`ExecutiveCockpitTab.jsx`)**:
   - Executive summary cards (Annualized Alpha, Max Drawdown, Information Ratio, Turnover).
   - Comparative equity trajectory chart (PPO+GAT vs Baselines).
   - Rigorous out-of-sample benchmark table with leak-free test performance.
2. **Model Ablations & Rigor (`ModelAblationsTab.jsx`)**:
   - 5-seed cross-validation radar chart and ablation comparison table (GAT vs GCN vs No Embeddings vs Freeform Softmax).
3. **GNN Correlation Network (`NetworkGraphTab.jsx`)**:
   - Fixed interactive canvas graph displaying 20 asset nodes clustered into 5 market sectors (Technology, Financials, Energy, Utilities, Real Estate).
   - Dynamic Pearson correlation threshold scrubber (`0.10` to `0.80`).
   - Node click inspector with asset stats (Beta, Volatility, Sector, Weight).
   - 20×20 correlation matrix heatmap table below the network graph.
4. **Allocations & Exposure (`AllocationsTab.jsx`)**:
   - Dynamic asset allocation streamgraph across all 20 assets.
   - Interactive date scrubber updating the sector exposure breakdown and asset weights in real-time.
5. **Transaction Friction (`TransactionFrictionTab.jsx`)**:
   - Cost decay curves demonstrating model robustness across execution slippage regimes (5 bps, 10 bps, 20 bps).
6. **What-If Backtester (`SimulatorTab.jsx`)**:
   - Real-time quantitative portfolio simulator.
   - User inputs: Initial capital ($), execution slippage (1–50 bps), rebalancing frequency (daily, weekly, monthly), and residual policy dampener α (0.00–0.50).
   - **All 11 strategy modes supported**:
     - *Machine Learning & GNN*: `ppo_gat_enhanced`, `ppo_gat_residual`, `ppo_gcn_residual`, `ppo_no_embeddings`, `ppo_freeform`
     - *Quantitative Baselines*: `momentum_residual`, `equal_weight_residual`, `mean_variance`, `momentum`, `buy_and_hold`, `equal_weight`

### C. Specific Fixes & Changes Implemented
1. **Removed Scientific Report Tab**:
   - Completely removed the legacy scientific report tab from navigation and view renderers.
2. **What-If Backtester Strategy Expansion**:
   - Expanded strategy selector and client/server simulation engine from 2 modes to all 11 ML and baseline strategies.
3. **GNN Graph Rendering & Physics Fixes**:
   - *Fixed Black Screen*: Correctly loads graph data and renders all 20 nodes with proper coordinates, labels, and sector colored badges.
   - *Fixed Infinite Horizontal Width Expansion*: Set `<canvas>` to `position: absolute; inset: 0; width: 100%; height: 100%;` and enforced `min-width: 0; overflow: hidden;` on the parent grid cards to eliminate layout loops.
   - *Fixed Node Overlap & High-Frequency Vibration*:
     - Arranged sector nodes into 60px orbital rings around sector centroids rather than stacking them at identical coordinates.
     - Added hard non-overlap collision repulsion (`minDist = 48px`).
     - Normalized spring vectors to eliminate quadratic distance explosions.
     - Implemented velocity clamping (`2.0 px/frame`), velocity damping (`0.76`), and zero-velocity cutoff below `0.02 px/frame`.
4. **Unified Backend & Hosting Architecture**:
   - Configured `server.py` (and `frontend/server.py`) to serve the compiled React SPA directly with fallback routing.
   - Preserved API endpoints: `GET /api/summary`, `GET /api/data`, `GET /api/graph`, and `POST /api/simulate`.
   - Added containerization files: `Dockerfile`, `docker-compose.yml`, `render.yaml`, and `DEPLOYMENT.md`.

---

## 2. Project Directory Structure

```
d:\Projects\ML_Package\
├── frontend/
│   ├── dist/                          # Production compiled React SPA
│   │   ├── index.html
│   │   └── assets/
│   │       ├── index-*.css
│   │       └── index-*.js
│   ├── public/                        # Static assets (data, favicons, svgs)
│   ├── src/
│   │   ├── api/
│   │   │   └── finoraApi.js           # API client for backend endpoints
│   │   ├── components/
│   │   │   ├── Header.jsx             # Top brand header, mode toggles, CSV/JSON export
│   │   │   ├── TabsNav.jsx            # 6-tab navigation strip
│   │   │   ├── Footer.jsx             # Footer with quantitative notices
│   │   │   └── tabs/
│   │   │       ├── ExecutiveCockpitTab.jsx
│   │   │       ├── ModelAblationsTab.jsx
│   │   │       ├── NetworkGraphTab.jsx
│   │   │       ├── AllocationsTab.jsx
│   │   │       ├── TransactionFrictionTab.jsx
│   │   │       └── SimulatorTab.jsx
│   │   ├── utils/
│   │   │   └── chartConfig.js         # Chart.js registered plugins and default styling
│   │   ├── App.jsx                    # Root React component
│   │   ├── main.jsx                   # React 19 entrypoint
│   │   └── index.css                  # Core CSS design system
│   ├── package.json                   # Dependencies (React 19, Chart.js, Lucide-react, Vite)
│   ├── vite.config.js                 # Vite dev proxy configuration
│   └── server.py                      # FastAPI server
├── server.py                          # Root server entrypoint
├── launch_frontend.bat                # 1-Click launcher for fullstack production server (port 8000)
├── launch_dev.bat                     # 1-Click launcher for dual hot-reload dev mode (port 5173 + 8000)
├── Dockerfile                         # Production multi-stage Docker build
├── docker-compose.yml                 # Docker compose deployment
├── render.yaml                        # Render.com deployment manifest
├── DEPLOYMENT.md                      # Cloud hosting instructions
├── PROJECT_HANDOFF_STATUS.md          # This handoff document
└── PROJECT_ACTION_PLAN.md             # Action plan and mathematical evaluation notes
```

---

## 3. How to Run the Application

### Option A: 1-Click Fullstack Production Server (Recommended)
1. Double-click `launch_frontend.bat` **OR** run in terminal:
   ```powershell
   cd d:\Projects\ML_Package
   .\venv\Scripts\python.exe -m uvicorn server:app --app-dir frontend --host 127.0.0.1 --port 8000
   ```
2. Open your browser to:
   ```
   http://127.0.0.1:8000
   ```
   *FastAPI serves the built React application and all REST API endpoints together.*

### Option B: Dual Development Server (Hot-Reloading)
1. Double-click `launch_dev.bat` **OR** run:
   - **Terminal 1 (Backend API)**:
     ```powershell
     .\venv\Scripts\python.exe -m uvicorn server:app --app-dir frontend --host 127.0.0.1 --port 8000 --reload
     ```
   - **Terminal 2 (React Vite HMR)**:
     ```powershell
     cd frontend
     npm.cmd run dev
     ```
2. Open `http://localhost:5173`. Any changes to React files hot-reload immediately.

---

## 4. Notes for Continuing Work in a New Session / Account

1. **Build Commands**:
   - Always run `npm.cmd` on Windows (e.g. `npm.cmd run build` or `npm.cmd run dev`) because PowerShell execution policy restricts `.ps1` scripts.
2. **Virtual Environment**:
   - Python virtualenv is at `d:\Projects\ML_Package\venv\Scripts\python.exe`.
   - Packages installed: `fastapi`, `uvicorn`, `httpx`, `pandas`, `numpy`, `torch_geometric`.
3. **Frontend Bundle**:
   - `frontend/dist/` is fresh, cleanly built, and matches all current React source code in `frontend/src/`.
4. **All Figma prototype experiments have been removed**:
   - No rocket landing page or Figma mock layouts remain.
   - The application is back to the clean Finora Quantitative AI dashboard as requested.
