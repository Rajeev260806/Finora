"""
PortfolioEnv — Phase 1 (Person B)
===================================

A Gymnasium-compatible environment for portfolio allocation, built on
top of Person A's cleaned dataset (data/clean/full_dataset.parquet).

DESIGN DECISIONS (documented per the checkpoint requirements)
---------------------------------------------------------------
1. Observation space -> Dict, not a flat vector:
       {
         "returns_window": (n_assets, window)  # past `window` days of returns per asset
         "weights":        (n_assets + 1,)     # current allocation incl. cash, from t-1
       }
   Reasoning: a GNN agent (later phase) needs per-asset feature vectors
   as node features. A Dict space keeps "returns_window" naturally
   shaped as (n_assets, window) so it can be fed to a GNN without any
   changes to this environment. A flat vector would need to be
   reshaped downstream every time and hides asset identity.

2. Action space -> long-only WITH cash (not fully-invested):
       raw logits of shape (n_assets + 1,) -> softmax -> weights that
       are non-negative and sum to 1, where the last slot is cash.
   Reasoning: forcing full investment (no cash) means an early/random
   policy is always fully exposed to market risk, which makes
   validation-episode portfolio value paths noisier/uglier than
   necessary and doesn't reflect realistic allocators. Cash gives the
   agent (and a random baseline) a safe fallback.

3. Reward (v1) -> portfolio log return net of transaction cost ONLY:
       reward = log(1 + port_return - cost)
   Deliberately minimal. A drawdown penalty is a natural v2 addition,
   but it should be added only AFTER this simpler version is validated
   -- see the "Tips for Solo Work" principle of starting minimal.

4. Transaction costs -> linear in turnover (basis points), no slippage
   yet. Slippage is a documented future refinement (Step 6 in the
   guide), not implemented in v1 to avoid over-engineering before the
   basic loop is proven correct.

5. No look-ahead bias: at step t, the reward uses returns[t] (today's
   realized return, which is legitimate — the agent commits its
   allocation and then "today" plays out). The NEXT observation's
   window ends at t (exclusive of t+1), so it only ever contains
   returns that have already been realized by the time it's shown.

ASSUMPTIONS / LIMITATIONS (for the Phase 1 handoff doc)
---------------------------------------------------------------
- Daily data only; no intraday dynamics.
- Transaction cost model is linear and constant (no slippage, no
  market-impact curve).
- No borrow cost modeled (no shorting is possible anyway — long-only).
- No interim cash flows; assumes Person A's data is already adjusted
  for dividends/splits (yfinance auto_adjust=True).
- Observation window is fixed at `window` days (configurable at
  construction, not adaptive).
- Cash always earns exactly 0 return (no risk-free rate modeled).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame,
        window: int = 20,
        transaction_cost_bps: float = 10.0,
        initial_cash: float = 1.0,
        return_col: str | None = None,
        seed: int | None = None,
    ):
        """
        Parameters
        ----------
        data : pd.DataFrame
            MultiIndex (Date, Ticker) DataFrame from Person A's pipeline
            (i.e. data/clean/full_dataset.parquet loaded with pandas).
        window : int
            Number of past days of returns the agent sees at each step.
        transaction_cost_bps : float
            Linear transaction cost in basis points (10 = 0.10%),
            applied to turnover each step.
        initial_cash : float
            Starting portfolio value in abstract units (1.0 = 100%).
        return_col : str, optional
            Explicit name of the returns column in `data`. If omitted,
            the env tries to auto-detect a sensible column name.
        seed : int, optional
            RNG seed for reproducibility.
        """
        super().__init__()

        if window < 1:
            raise ValueError("window must be >= 1")

        self.window = window
        self.tc_rate = transaction_cost_bps / 10_000.0
        self.initial_cash = initial_cash

        # ---- reshape the panel data into a (n_dates, n_assets) returns matrix ----
        self.tickers = sorted(data.index.get_level_values(-1).unique())
        self.n_assets = len(self.tickers)
        if self.n_assets < 1:
            raise ValueError("No tickers found in data index.")

        self.return_col = return_col or self._detect_return_col(data)

        wide = data[self.return_col].unstack(level=-1)  # index=Date, columns=Ticker
        wide = wide.reindex(columns=self.tickers)
        n_missing = int(wide.isnull().sum().sum())
        if n_missing:
            # Person A's pipeline should have handled NaNs; this is a
            # safety net, not a substitute for clean upstream data.
            wide = wide.fillna(0.0)

        self.returns = wide.values.astype(np.float32)  # (n_dates, n_assets)
        self.dates = wide.index
        self.n_steps = self.returns.shape[0]

        if self.n_steps <= self.window + 1:
            raise ValueError(
                f"Not enough dates ({self.n_steps}) for window={self.window}. "
                "Reduce window or provide more data."
            )

        # ---- spaces ----
        # Action: n_assets + 1 (cash) raw logits -> softmax'd internally
        # in _process_action. Using unconstrained logits (rather than
        # asking the policy to output valid weights directly) means any
        # RL algorithm with a Box action space works out of the box.
        self.action_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.n_assets + 1,), dtype=np.float32
        )

        self.observation_space = spaces.Dict(
            {
                "returns_window": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(self.n_assets, self.window), dtype=np.float32,
                ),
                "weights": spaces.Box(
                    low=0.0, high=1.0, shape=(self.n_assets + 1,), dtype=np.float32
                ),
            }
        )

        self._np_rng = np.random.default_rng(seed)
        self.current_idx = self.window
        self.portfolio_value = self.initial_cash
        self.prev_weights = np.zeros(self.n_assets + 1, dtype=np.float32)
        self.prev_weights[-1] = 1.0
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    @staticmethod
    def _detect_return_col(data: pd.DataFrame) -> str:
        candidates = [
            "Returns Daily", "Returns_Daily",
            "daily_return", "daily_returns", "return", "Return",
            "log_return", "log_returns", "ret", "Ret", "Log Return",
        ]
        for c in candidates:
            if c in data.columns:
                return c
        raise ValueError(
            "Could not auto-detect a returns column. "
            f"Available columns: {list(data.columns)}. "
            "Pass return_col explicitly, e.g. PortfolioEnv(data, return_col='daily_return')."
        )

    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_rng = np.random.default_rng(seed)

        self.current_idx = self.window  # first index with a full window behind it
        self.portfolio_value = self.initial_cash
        # start fully in cash: weights = [0, ..., 0, 1.0]
        self.prev_weights = np.zeros(self.n_assets + 1, dtype=np.float32)
        self.prev_weights[-1] = 1.0
        self.history = []

        obs = self._get_observation(self.current_idx)
        info = {"portfolio_value": self.portfolio_value}
        return obs, info

    def _get_observation(self, idx: int) -> dict:
        window_slice = self.returns[idx - self.window: idx]  # (window, n_assets)
        returns_window = window_slice.T.astype(np.float32)  # (n_assets, window)
        return {
            "returns_window": returns_window,
            "weights": self.prev_weights.copy(),
        }

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        z = x - np.max(x)
        e = np.exp(z)
        return e / e.sum()

    def _process_action(self, action: np.ndarray) -> np.ndarray:
        """Convert raw agent output into valid long-only weights (incl. cash)."""
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        expected = self.n_assets + 1
        if action.shape[0] != expected:
            raise ValueError(f"Expected action of length {expected}, got {action.shape[0]}")
        return self._softmax(action)

    def _apply_transaction_costs(self, weights: np.ndarray) -> tuple[float, float]:
        """Returns (turnover, cost) for moving from prev_weights to weights."""
        turnover = float(np.sum(np.abs(weights - self.prev_weights)))
        cost = self.tc_rate * turnover
        return turnover, cost

    def _calculate_reward(self, port_return: float, cost: float) -> float:
        net_return = port_return - cost
        # guard against log(<=-1) if costs ever exceed -100% (shouldn't
        # happen with long-only weights, but keeps reward finite)
        if net_return <= -1.0:
            return -10.0
        return float(np.log1p(net_return))

    def step(self, action):
        weights = self._process_action(action)          # (n_assets+1,), last = cash
        today_returns = self.returns[self.current_idx]   # (n_assets,)
        asset_weights = weights[:-1]

        port_return = float(np.dot(asset_weights, today_returns))  # cash contributes 0
        turnover, cost = self._apply_transaction_costs(weights)
        reward = self._calculate_reward(port_return, cost)

        self.portfolio_value *= (1.0 + port_return - cost)
        self.prev_weights = weights
        step_date = self.dates[self.current_idx]
        self.current_idx += 1

        terminated = self.current_idx >= self.n_steps
        truncated = False
        next_idx = min(self.current_idx, self.n_steps - 1)
        obs = self._get_observation(next_idx)

        info = {
            "date": step_date,
            "portfolio_value": self.portfolio_value,
            "turnover": turnover,
            "transaction_cost": cost,
            "raw_return": port_return,
            "cash_weight": float(weights[-1]),
        }
        self.history.append(info)

        return obs, reward, terminated, truncated, info

    def render(self):
        idx = min(self.current_idx, self.n_steps - 1)
        print(
            f"Date: {self.dates[idx].date()} | "
            f"Portfolio value: {self.portfolio_value:.4f} | "
            f"Cash weight: {self.prev_weights[-1]:.2%}"
        )

    def get_history_df(self) -> pd.DataFrame:
        """Episode log for Person C — value/turnover/return/cash per step."""
        return pd.DataFrame(self.history).set_index("date")