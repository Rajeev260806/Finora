/**
 * Finora API Client
 * Supports both unified deployment (relative /api) and decoupled deployment (VITE_API_URL).
 * Also falls back seamlessly to /data/model_outputs.json if offline or static preview.
 */

const API_BASE = import.meta.env.VITE_API_URL || '';

export async function fetchAllData() {
  // 1. Try backend API
  try {
    const res = await fetch(`${API_BASE}/api/data`);
    if (res.ok) {
      const data = await res.json();
      console.log('[Finora API] Successfully loaded model data from server.');
      return data;
    }
  } catch (err) {
    console.warn('[Finora API] Backend API unavailable, falling back to static asset.', err);
  }

  // 2. Fallback to public static JSON
  try {
    const fallbackRes = await fetch('/data/model_outputs.json');
    if (fallbackRes.ok) {
      const data = await fallbackRes.json();
      console.log('[Finora API] Loaded dataset from static public bundle.');
      return data;
    }
  } catch (err) {
    console.error('[Finora API] Failed to load static fallback dataset.', err);
  }

  throw new Error('Unable to fetch Finora model dataset from API or static asset.');
}

export async function runSimulation(payload) {
  // 1. Try backend API simulation endpoint
  try {
    const res = await fetch(`${API_BASE}/api/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (res.ok) {
      return await res.json();
    }
  } catch (err) {
    console.warn('[Finora API] Backend simulation endpoint failed, using client-side fallback simulation engine.', err);
  }

  // 2. Client-side simulation fallback if server is not available
  return null;
}
