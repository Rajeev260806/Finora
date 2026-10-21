import React from 'react';
import { LayoutGrid, Layers, Share2, PieChart, DollarSign, Play } from 'lucide-react';

const TABS = [
  { id: 'tab-cockpit', label: 'Executive Cockpit', icon: LayoutGrid },
  { id: 'tab-ablations', label: 'Model Ablations & Rigor', icon: Layers },
  { id: 'tab-network', label: 'GNN Correlation Graph', icon: Share2 },
  { id: 'tab-weights', label: 'Allocations & Exposure', icon: PieChart },
  { id: 'tab-friction', label: 'Transaction Friction', icon: DollarSign },
  { id: 'tab-simulator', label: 'What-If Backtester', icon: Play },
];

export default function TabsNav({ activeTab, setActiveTab }) {
  return (
    <nav className="tab-strip">
      {TABS.map((tab) => {
        const Icon = tab.icon;
        const isActive = activeTab === tab.id;
        return (
          <button
            key={tab.id}
            type="button"
            className={`tab-btn ${isActive ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            <Icon size={17} />
            {tab.label}
          </button>
        );
      })}
    </nav>
  );
}
