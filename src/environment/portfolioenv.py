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
        use_embeddings: bool = False,
        embedding_dim: int = 32,
        embedding_dir: str | None = None,
        lambda_vol: float = 0.0,
        lambda_conc: float = 0.0,
        lambda_txn: float = 1.0,
        residual_alpha: float = 0.0,
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
        use_embeddings : bool, default=False
            If True, observation space returns GNN node embeddings
            instead of return windows. Used for Phase 3 testing and Phase 4 RL.
        embedding_dim : int, default=32
            Dimensionality of GNN embeddings per asset node.
        embedding_dir : str, optional
            Directory path containing pre-computed GNN embedding files (.npy).
            If None, checks 'src/gnn/embeddings' automatically. If missing, synthetic placeholder embeddings are generated.
        lambda_vol : float, default=0.0
            Penalty weight for annualized 20-day rolling portfolio volatility (Phase 2 model).
        lambda_conc : float, default=0.0
            Penalty weight for asset concentration (HHI across risky assets).
        lambda_txn : float, default=1.0
            Multiplier for transaction cost deduction in reward calculation.
        residual_alpha : float, default=0.0
            If > 0.0, implements residual policy around Equal Weight:
            w = softmax(log(1/N) + residual_alpha * tanh(action))
            as required by PROJECT_ACTION_PLAN.md Phase 3.1.
        seed : int, optional
            RNG seed for reproducibility.
        """
        super().__init__()

        if window < 1:
            raise ValueError("window must be >= 1")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be >= 1")

        self.window = window
        self.tc_rate = transaction_cost_bps / 10_000.0
        self.initial_cash = initial_cash
        self.use_embeddings = use_embeddings
        self.embedding_dim = embedding_dim
        self.embedding_dir = embedding_dir
        self.lambda_vol = lambda_vol
        self.lambda_conc = lambda_conc
        self.lambda_txn = lambda_txn
        self.residual_alpha = residual_alpha
        self.seed_val = seed

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
        # in _process_action. Using unconstrained logits means any
        # RL algorithm with a Box action space works out of the box.
        self.action_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.n_assets + 1,), dtype=np.float32
        )

        if self.use_embeddings:
            self.observation_space = spaces.Dict(
                {
                    "gnn_embeddings": spaces.Box(
                        low=-np.inf, high=np.inf,
                        shape=(self.n_assets, self.embedding_dim), dtype=np.float32,
                    ),
                    "weights": spaces.Box(
                        low=0.0, high=1.0, shape=(self.n_assets + 1,), dtype=np.float32
                    ),
                }
            )
        else:
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
        self._embedding_cache: dict[str, np.ndarray] = {}
        self._date_to_idx: dict[str, int] = {}
        self._sorted_graph_dates: list[str] = []
        self._preloaded_embeddings: dict[int, np.ndarray] = {}
        self._init_embedding_date_map()
        if self.use_embeddings:
            self._preload_embeddings()

    def _preload_embeddings(self):
        """Preloads embeddings into RAM for instantaneous O(1) step access during training."""
        import os
        dirs_to_check = []
        if self.embedding_dir:
            dirs_to_check.append(self.embedding_dir)
        dirs_to_check.extend([
            os.path.join("src", "gnn", "embeddings"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gnn", "embeddings"),
        ])
        for target_dir in dirs_to_check:
            if target_dir and os.path.exists(target_dir):
                files = [f for f in os.listdir(target_dir) if f.endswith(".npy")]
                for fname in files:
                    try:
                        # Extract integer index
                        idx_part = "".join(c for c in fname.replace(".npy", "") if c.isdigit())
                        if idx_part:
                            idx = int(idx_part)
                            emb = np.load(os.path.join(target_dir, fname)).astype(np.float32)
                            if emb.shape == (self.n_assets, self.embedding_dim):
                                self._preloaded_embeddings[idx] = emb
                    except Exception:
                        pass
                if len(self._preloaded_embeddings) > 0:
                    break

    def _init_embedding_date_map(self):
        """Builds lookup mapping from calendar date strings to graph snapshot indices."""
        import os
        import pandas as pd
        csv_candidates = [
            os.path.join("graphs", "graph_features.csv"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "graphs", "graph_features.csv"),
        ]
        for csv_path in csv_candidates:
            if os.path.exists(csv_path):
                try:
                    feat_df = pd.read_csv(csv_path)
                    if "date" in feat_df.columns:
                        for i, d in enumerate(feat_df["date"]):
                            self._date_to_idx[str(d)[:10]] = i
                        self._sorted_graph_dates = sorted(self._date_to_idx.keys())
                        break
                except Exception:
                    pass

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
            self.seed_val = seed
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
        if self.use_embeddings:
            emb = self._load_or_generate_embeddings(idx)
            return {
                "gnn_embeddings": emb,
                "weights": self.prev_weights.copy(),
            }
        else:
            window_slice = self.returns[idx - self.window: idx]  # (window, n_assets)
            returns_window = window_slice.T.astype(np.float32)  # (n_assets, window)
            return {
                "returns_window": returns_window,
                "weights": self.prev_weights.copy(),
            }

    def _load_or_generate_embeddings(self, idx: int) -> np.ndarray:
        """Loads GNN embeddings from file if available; otherwise generates placeholder embeddings."""
        import os

        # Determine date and embedding file index
        target_idx = idx
        if 0 <= idx < len(self.dates):
            date_str = str(self.dates[idx])[:10]
            if self._date_to_idx:
                if date_str in self._date_to_idx:
                    target_idx = self._date_to_idx[date_str]
                elif self._sorted_graph_dates and date_str < self._sorted_graph_dates[0]:
                    target_idx = 0
                elif self._sorted_graph_dates and date_str > self._sorted_graph_dates[-1]:
                    target_idx = self._date_to_idx[self._sorted_graph_dates[-1]]

        cache_key = f"idx_{idx}_target_{target_idx}"
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key].copy()

        if target_idx in self._preloaded_embeddings:
            res = self._preloaded_embeddings[target_idx]
            self._embedding_cache[cache_key] = res
            return res.copy()

        dirs_to_check = []
        if self.embedding_dir:
            dirs_to_check.append(self.embedding_dir)
        dirs_to_check.extend([
            os.path.join("src", "gnn", "embeddings"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gnn", "embeddings"),
            os.path.join("graphs", "saved_embeddings"),
        ])

        patterns = [
            f"embeddings_graph_{target_idx:04d}.npy",
            f"embeddings_graph_{idx:04d}.npy",
            f"embedding_{target_idx:04d}.npy",
            f"embedding_{idx:04d}.npy",
            f"{target_idx:04d}.npy",
            f"{idx:04d}.npy",
        ]

        for target_dir in dirs_to_check:
            if target_dir and os.path.exists(target_dir):
                for filename in patterns:
                    filepath = os.path.join(target_dir, filename)
                    if os.path.exists(filepath):
                        emb = np.load(filepath).astype(np.float32)
                        res = None
                        if emb.shape == (self.n_assets, self.embedding_dim):
                            res = emb
                        elif emb.ndim == 2 and emb.shape[0] == self.n_assets:
                            dim = emb.shape[1]
                            if dim > self.embedding_dim:
                                res = emb[:, :self.embedding_dim].astype(np.float32)
                            elif dim < self.embedding_dim:
                                padded = np.zeros((self.n_assets, self.embedding_dim), dtype=np.float32)
                                padded[:, :dim] = emb
                                res = padded
                        if res is not None:
                            self._embedding_cache[cache_key] = res
                            return res.copy()

        # Deterministic synthetic placeholder embeddings for reproducibility
        seed = (self.seed_val or 42) + idx
        rng = np.random.default_rng(seed)
        res = (rng.standard_normal((self.n_assets, self.embedding_dim)) * 0.1).astype(np.float32)
        self._embedding_cache[cache_key] = res
        return res.copy()

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        z = x - np.max(x)
        e = np.exp(z)
        return e / e.sum()

    def _process_action(self, action: np.ndarray) -> np.ndarray:
        """
        Convert raw agent output into valid long-only weights (incl. cash).
        Supports standard free-form softmax (residual_alpha=0) and residual policy
        around Equal Weight (residual_alpha > 0):
            w = softmax( log(w_base) + residual_alpha * tanh(action) )
        as specified in PROJECT_ACTION_PLAN.md Phase 3.1.
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        expected = self.n_assets + 1
        if action.shape[0] != expected:
            raise ValueError(f"Expected action of length {expected}, got {action.shape[0]}")

        if self.residual_alpha > 0.0:
            # Base weights: Equal weight across stocks (1/N), 0 for cash
            base_weights = np.zeros(expected, dtype=np.float32)
            base_weights[:self.n_assets] = 1.0 / self.n_assets
            base_weights[-1] = 1e-4  # tiny epsilon for cash
            base_logits = np.log(base_weights)
            # Residual perturbation
            logits = base_logits + self.residual_alpha * np.tanh(action)
            return self._softmax(logits)
        else:
            return self._softmax(action)

    def _apply_transaction_costs(self, weights: np.ndarray) -> tuple[float, float]:
        """Returns (turnover, cost) for moving from prev_weights to weights."""
        turnover = float(np.sum(np.abs(weights - self.prev_weights)))
        cost = self.tc_rate * turnover
        return turnover, cost

    def _calculate_reward(
        self,
        port_return: float,
        cost: float,
        asset_weights: np.ndarray,
        window_returns: np.ndarray,
    ) -> float:
        """Calculates risk-adjusted, cost-aware modular reward (supporting Phase 2 spec)."""
        net_return = port_return - (self.lambda_txn * cost)

        if net_return <= -1.0:
            log_net_return = -10.0
        else:
            log_net_return = float(np.log1p(net_return))

        # Volatility penalty (annualized 20-day rolling portfolio volatility)
        vol_penalty = 0.0
        if self.lambda_vol > 0.0 and window_returns.size > 0 and np.sum(asset_weights) > 0.0:
            # window_returns is (window, n_assets)
            port_win_returns = window_returns @ asset_weights  # (window,)
            if len(port_win_returns) > 1:
                daily_vol = float(np.std(port_win_returns, ddof=1))
                ann_vol = daily_vol * np.sqrt(252.0)
                vol_penalty = self.lambda_vol * ann_vol

        # Concentration penalty (Herfindahl-Hirschman Index across risky assets)
        conc_penalty = 0.0
        if self.lambda_conc > 0.0:
            hhi = float(np.sum(asset_weights ** 2))
            conc_penalty = self.lambda_conc * hhi

        return log_net_return - vol_penalty - conc_penalty

    def step(self, action):
        weights = self._process_action(action)          # (n_assets+1,), last = cash
        today_returns = self.returns[self.current_idx]   # (n_assets,)
        asset_weights = weights[:-1]

        port_return = float(np.dot(asset_weights, today_returns))  # cash contributes 0
        turnover, cost = self._apply_transaction_costs(weights)

        # Get window slice for reward volatility penalty if needed
        window_slice = self.returns[max(0, self.current_idx - self.window): self.current_idx]
        reward = self._calculate_reward(port_return, cost, asset_weights, window_slice)

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
            "net_return": port_return - cost,
            "cash_weight": float(weights[-1]),
            "reward": reward,
        }
        self.history.append(info)

        return obs, reward, terminated, truncated, info

    def render(self):
        idx = min(self.current_idx, self.n_steps - 1)
        print(
            f"Date: {self.dates[idx].date()} | "
            f"Portfolio value: {self.portfolio_value:.4f} | "
            f"Cash weight: {self.prev_weights[-1]:.2%} | "
            f"Obs mode: {'GNN Embeddings' if self.use_embeddings else 'Returns Window'}"
        )

    def get_history_df(self) -> pd.DataFrame:
        """Episode log for Person C — value/turnover/return/cash per step."""
        return pd.DataFrame(self.history).set_index("date")