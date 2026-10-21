import React from 'react';
import { ArrowRight, Activity, Share2, Layers, DollarSign } from 'lucide-react';

export default function LandingPage({ onEnterDashboard }) {
  return (
    <div className="landing-wrapper">
      {/* =========================================================================
          BACKGROUND: Minimalist, Hollow Vector Clipart (Stocks, Relationships, Graphs, Cash)
          Palette: Champagne Gold (#fbbf24), Warm Amber (#f59e0b), and Light Champagne (#fde68a)
          Opacity: 10%–20%, Hollow outlines, transparent background, behind content
          ========================================================================= */}
      <svg
        className="landing-bg-svg"
        viewBox="0 0 1600 1000"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        preserveAspectRatio="xMidYMid slice"
      >
        <defs>
          <linearGradient id="goldAmberGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#fbbf24" />
            <stop offset="100%" stopColor="#d97706" />
          </linearGradient>
          <linearGradient id="amberGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#f59e0b" />
            <stop offset="100%" stopColor="#fbbf24" />
          </linearGradient>
          <pattern id="dotGrid" x="0" y="0" width="40" height="40" patternUnits="userSpaceOnUse">
            <circle cx="2" cy="2" r="1" fill="#f59e0b" fillOpacity="0.14" />
          </pattern>
        </defs>

        {/* Global subtle coordinate grid */}
        <rect width="100%" height="100%" fill="url(#dotGrid)" />

        {/* =======================================================
            CLIPART GROUP 1: Stock Market Candlesticks & Price Charts (Top-Right)
            ======================================================= */}
        <g className="clipart-stocks" opacity="0.18" strokeWidth="1.5">
          {/* Coordinate axis */}
          <line x1="1080" y1="280" x2="1520" y2="280" stroke="#f59e0b" strokeDasharray="3 4" />
          <line x1="1080" y1="80" x2="1080" y2="280" stroke="#f59e0b" strokeDasharray="3 4" />

          {/* Hollow Candlestick 1 */}
          <line x1="1120" y1="130" x2="1120" y2="250" stroke="#fbbf24" />
          <rect x="1112" y="160" width="16" height="55" rx="2" stroke="#fbbf24" fill="none" />

          {/* Hollow Candlestick 2 */}
          <line x1="1170" y1="110" x2="1170" y2="230" stroke="#f59e0b" />
          <rect x="1162" y="130" width="16" height="60" rx="2" stroke="#f59e0b" fill="rgba(245,158,11,0.06)" />

          {/* Hollow Candlestick 3 */}
          <line x1="1220" y1="90" x2="1220" y2="210" stroke="#fde68a" />
          <rect x="1212" y="110" width="16" height="45" rx="2" stroke="#fde68a" fill="none" />

          {/* Hollow Candlestick 4 */}
          <line x1="1270" y1="140" x2="1270" y2="270" stroke="#fbbf24" />
          <rect x="1262" y="180" width="16" height="60" rx="2" stroke="#fbbf24" fill="none" />

          {/* Hollow Candlestick 5 */}
          <line x1="1320" y1="70" x2="1320" y2="200" stroke="#f59e0b" />
          <rect x="1312" y="90" width="16" height="70" rx="2" stroke="#f59e0b" fill="rgba(251,191,36,0.08)" />

          {/* Hollow Candlestick 6 */}
          <line x1="1370" y1="60" x2="1370" y2="180" stroke="#d97706" />
          <rect x="1362" y="80" width="16" height="50" rx="2" stroke="#d97706" fill="none" />

          {/* Ascending Trend Spline Curve */}
          <path
            d="M 1100 240 Q 1180 200 1230 160 T 1340 110 T 1480 75"
            stroke="#fbbf24"
            strokeWidth="2"
            fill="none"
          />

          {/* Data point markers on trend curve */}
          <circle cx="1100" cy="240" r="3.5" stroke="#fbbf24" fill="#07080a" />
          <circle cx="1230" cy="160" r="3.5" stroke="#f59e0b" fill="#07080a" />
          <circle cx="1340" cy="110" r="3.5" stroke="#fde68a" fill="#07080a" />
          <circle cx="1480" cy="75" r="4.5" stroke="#fbbf24" fill="#fbbf24" fillOpacity="0.3" />

          {/* Stock Ticker Badges */}
          <rect x="1390" y="140" width="56" height="20" rx="4" stroke="#f59e0b" strokeDasharray="2 2" fill="none" />
          <text x="1401" y="154" fill="#fbbf24" fontSize="9" fontFamily="monospace" letterSpacing="1">AAPL</text>

          <rect x="1455" y="170" width="56" height="20" rx="4" stroke="#d97706" strokeDasharray="2 2" fill="none" />
          <text x="1466" y="184" fill="#f59e0b" fontSize="9" fontFamily="monospace" letterSpacing="1">NVDA</text>
        </g>

        {/* =======================================================
            CLIPART GROUP 2: Graph Relationships & Interconnected Nodes (Top-Left)
            ======================================================= */}
        <g className="clipart-relationships" opacity="0.18" strokeWidth="1.4">
          {/* Interconnecting Relationship Edges */}
          <line x1="160" y1="180" x2="280" y2="120" stroke="#fbbf24" strokeDasharray="4 4" />
          <line x1="160" y1="180" x2="230" y2="290" stroke="#f59e0b" />
          <line x1="280" y1="120" x2="390" y2="170" stroke="#d97706" />
          <line x1="280" y1="120" x2="230" y2="290" stroke="#fbbf24" />
          <line x1="230" y1="290" x2="390" y2="170" stroke="#f59e0b" strokeDasharray="2 3" />
          <line x1="390" y1="170" x2="490" y2="260" stroke="#fde68a" />
          <line x1="230" y1="290" x2="340" y2="380" stroke="#fbbf24" />
          <line x1="340" y1="380" x2="490" y2="260" stroke="#f59e0b" />

          {/* Graph Nodes (Hollow circles with center dot) */}
          <circle cx="160" cy="180" r="18" stroke="#fbbf24" fill="#07080a" />
          <circle cx="160" cy="180" r="4" fill="#fbbf24" />

          <circle cx="280" cy="120" r="24" stroke="#f59e0b" fill="#07080a" />
          <circle cx="280" cy="120" r="14" stroke="#f59e0b" strokeDasharray="2 2" fill="none" />
          <circle cx="280" cy="120" r="5" fill="#f59e0b" />

          <circle cx="230" cy="290" r="20" stroke="#d97706" fill="#07080a" />
          <circle cx="230" cy="290" r="4" fill="#d97706" />

          <circle cx="390" cy="170" r="22" stroke="#fbbf24" fill="#07080a" />
          <circle cx="390" cy="170" r="5" fill="#fbbf24" />

          <circle cx="490" cy="260" r="16" stroke="#f59e0b" fill="#07080a" />
          <circle cx="490" cy="260" r="3.5" fill="#f59e0b" />

          <circle cx="340" cy="380" r="18" stroke="#fde68a" fill="#07080a" />
          <circle cx="340" cy="380" r="4" fill="#fde68a" />

          {/* Orbital Relationship Rings */}
          <circle cx="280" cy="120" r="45" stroke="#fbbf24" strokeWidth="0.8" strokeDasharray="3 5" fill="none" />
          <circle cx="390" cy="170" r="42" stroke="#f59e0b" strokeWidth="0.8" strokeDasharray="4 6" fill="none" />
        </g>

        {/* =======================================================
            CLIPART GROUP 3: Cash, Assets & Currency Symbols (Bottom-Left)
            ======================================================= */}
        <g className="clipart-cash" opacity="0.16" strokeWidth="1.5">
          {/* Layered Banknote 1 (Back) */}
          <rect x="130" y="740" width="170" height="95" rx="6" stroke="#f59e0b" strokeDasharray="4 3" fill="none" transform="rotate(-6 130 740)" />

          {/* Layered Banknote 2 (Front) */}
          <rect x="160" y="760" width="180" height="100" rx="6" stroke="#fbbf24" fill="#07080a" />
          <rect x="172" y="772" width="156" height="76" rx="4" stroke="#fbbf24" strokeWidth="1" strokeDasharray="2 3" fill="none" />

          {/* Banknote Center Shield & Dollar Motif */}
          <circle cx="250" cy="810" r="22" stroke="#fbbf24" fill="none" />
          <path
            d="M 250 798 V 822 M 255 803 C 255 799 245 799 245 807 C 245 815 255 813 255 821 C 255 825 245 825 245 821"
            stroke="#fbbf24"
            strokeWidth="2"
            strokeLinecap="round"
          />

          {/* Outlined Coin Stack */}
          <ellipse cx="390" cy="850" rx="30" ry="12" stroke="#d97706" fill="none" />
          <ellipse cx="390" cy="840" rx="30" ry="12" stroke="#d97706" fill="none" />
          <ellipse cx="390" cy="830" rx="30" ry="12" stroke="#d97706" fill="#07080a" />
          <ellipse cx="390" cy="820" rx="30" ry="12" stroke="#f59e0b" fill="#07080a" />
          <path d="M 390 813 V 827 M 394 816 C 394 814 386 814 386 818 C 386 822 394 822 394 826 C 394 828 386 828 386 826" stroke="#f59e0b" strokeWidth="1.5" strokeLinecap="round" />

          {/* Liquidity Flow Arrow */}
          <path
            d="M 350 770 Q 420 740 470 760"
            stroke="#fbbf24"
            strokeWidth="1.6"
            strokeDasharray="4 3"
            fill="none"
          />
          <polygon points="475,762 465,754 468,766" fill="#fbbf24" />
        </g>

        {/* =======================================================
            CLIPART GROUP 4: Neural Network & Algorithmic Graphs (Bottom-Right)
            ======================================================= */}
        <g className="clipart-gnn" opacity="0.18" strokeWidth="1.3">
          {/* Layer 1 Nodes (Input Features) */}
          <line x1="1180" y1="720" x2="1300" y2="690" stroke="#fbbf24" />
          <line x1="1180" y1="720" x2="1300" y2="760" stroke="#f59e0b" strokeDasharray="3 3" />
          <line x1="1180" y1="720" x2="1300" y2="830" stroke="#d97706" />

          <line x1="1180" y1="790" x2="1300" y2="690" stroke="#d97706" strokeDasharray="2 3" />
          <line x1="1180" y1="790" x2="1300" y2="760" stroke="#fbbf24" />
          <line x1="1180" y1="790" x2="1300" y2="830" stroke="#f59e0b" />

          <line x1="1180" y1="860" x2="1300" y2="760" stroke="#fde68a" />
          <line x1="1180" y1="860" x2="1300" y2="830" stroke="#fbbf24" strokeDasharray="3 3" />

          {/* Layer 2 to Output Layer */}
          <line x1="1300" y1="690" x2="1420" y2="740" stroke="#fbbf24" />
          <line x1="1300" y1="760" x2="1420" y2="740" stroke="#f59e0b" />
          <line x1="1300" y1="760" x2="1420" y2="810" stroke="#d97706" />
          <line x1="1300" y1="830" x2="1420" y2="810" stroke="#fbbf24" />

          {/* Layer 1 Nodes */}
          <circle cx="1180" cy="720" r="10" stroke="#fbbf24" fill="#07080a" />
          <circle cx="1180" cy="790" r="10" stroke="#f59e0b" fill="#07080a" />
          <circle cx="1180" cy="860" r="10" stroke="#d97706" fill="#07080a" />

          {/* Layer 2 Nodes (Relational Hidden) */}
          <circle cx="1300" cy="690" r="12" stroke="#fbbf24" fill="#07080a" />
          <circle cx="1300" cy="690" r="4" fill="#fbbf24" />

          <circle cx="1300" cy="760" r="14" stroke="#f59e0b" fill="#07080a" />
          <circle cx="1300" cy="760" r="5" fill="#f59e0b" />

          <circle cx="1300" cy="830" r="12" stroke="#d97706" fill="#07080a" />
          <circle cx="1300" cy="830" r="4" fill="#d97706" />

          {/* Output Nodes (Allocation Vector) */}
          <circle cx="1420" cy="740" r="16" stroke="#fbbf24" fill="#07080a" />
          <circle cx="1420" cy="740" r="7" stroke="#fbbf24" strokeDasharray="2 2" fill="none" />

          <circle cx="1420" cy="810" r="16" stroke="#f59e0b" fill="#07080a" />
          <circle cx="1420" cy="810" r="7" stroke="#f59e0b" strokeDasharray="2 2" fill="none" />
        </g>

        {/* =======================================================
            CENTER BACKGROUND: Concentric Geometry & Radar Ring
            ======================================================= */}
        <g opacity="0.09" strokeWidth="1">
          <circle cx="800" cy="500" r="320" stroke="#f59e0b" strokeDasharray="4 8" />
          <circle cx="800" cy="500" r="220" stroke="#fbbf24" strokeDasharray="2 6" />
          <circle cx="800" cy="500" r="120" stroke="#d97706" strokeDasharray="3 4" />
          <line x1="450" y1="500" x2="1150" y2="500" stroke="#f59e0b" strokeDasharray="2 6" />
          <line x1="800" y1="160" x2="800" y2="840" stroke="#fbbf24" strokeDasharray="2 6" />
        </g>
      </svg>

      {/* =========================================================================
          FOREGROUND CONTENT: Minimalist, Crisp, Institutional FinTech Header & Hero
          ========================================================================= */}
      <header className="landing-nav">
        <div className="landing-brand">
          <div className="landing-logo-badge">
            <Share2 size={20} className="landing-logo-icon" />
          </div>
          <span className="landing-brand-text">Finora</span>
        </div>

        <div className="landing-nav-right">
          <div className="landing-engine-tag">
            <span className="landing-pulse-dot"></span>
            <span>GNN + Deep RL Live Model</span>
          </div>
          <button
            type="button"
            className="landing-secondary-cta"
            onClick={onEnterDashboard}
          >
            Terminal View
          </button>
        </div>
      </header>

      <main className="landing-hero-center">
        {/* Subtle pill tag above title */}
        <div className="landing-pill-badge">
          <Activity size={13} />
          <span>Graph Neural Network Portfolio Engine</span>
        </div>

        {/* Project Title */}
        <h1 className="landing-title">
          Finora
        </h1>

        {/* 2-line crisp description */}
        <p className="landing-description">
          Graph-Neural-Network Powered Portfolio Intelligence & Reinforcement Learning.<br />
          Dynamic relational asset modeling engineered for out-of-sample risk and alpha optimization.
        </p>

        {/* Call to action button to redirect to dashboard */}
        <div className="landing-cta-container">
          <button
            type="button"
            className="landing-primary-btn"
            onClick={onEnterDashboard}
          >
            <span>Launch Dashboard</span>
            <ArrowRight size={18} className="landing-btn-arrow" />
          </button>
        </div>

        {/* Feature highlight badges */}
        <div className="landing-feature-strip">
          <div className="landing-feature-item">
            <Share2 size={14} className="feature-icon" />
            <span>Relational GAT Embeddings</span>
          </div>
          <div className="landing-feature-dot">•</div>
          <div className="landing-feature-item">
            <Layers size={14} className="feature-icon" />
            <span>Leak-Free Holdout Evaluation</span>
          </div>
          <div className="landing-feature-dot">•</div>
          <div className="landing-feature-item">
            <DollarSign size={14} className="feature-icon" />
            <span>Transaction Friction Aware (10 bps)</span>
          </div>
        </div>
      </main>

      <footer className="landing-footer">
        <span>Finora Quantitative Intelligence © 2026</span>
        <span className="footer-sep">|</span>
        <span>Institutional Asset Allocation Platform</span>
      </footer>
    </div>
  );
}
