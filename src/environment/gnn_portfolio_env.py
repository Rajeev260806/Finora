from __future__ import annotations

import os
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.environment.portfolioenv import PortfolioEnv


class GNNPortfolioEnv(PortfolioEnv):

    def __init__(
        self,
        data: pd.DataFrame,
        embedding_dir: str,
        window: int = 20,
        transaction_cost_bps: float = 10.0,
        initial_cash: float = 1.0,
        return_col: str | None = None,
        seed: int | None = None,
    ):
        super().__init__(
            data=data,
            window=window,
            transaction_cost_bps=transaction_cost_bps,
            initial_cash=initial_cash,
            return_col=return_col,
            seed=seed,
        )

        self.embedding_dir = embedding_dir

        self.embedding_files = sorted(
            f
            for f in os.listdir(embedding_dir)
            if f.startswith("embedding_") and f.endswith(".npy")
        )

        if not self.embedding_files:
            raise FileNotFoundError(
                f"No embedding files found in {embedding_dir}"
            )

        first_embedding = np.load(
            os.path.join(
                embedding_dir,
                self.embedding_files[0]
            )
        ).astype(np.float32)

        if first_embedding.shape[0] != self.n_assets:
            raise ValueError(
                f"Embedding has {first_embedding.shape[0]} stocks, "
                f"but environment has {self.n_assets}."
            )

        self.embedding_dim = first_embedding.shape[1]

        self.observation_space = spaces.Dict(
            {
                "gnn_embeddings": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.n_assets, self.embedding_dim),
                    dtype=np.float32,
                ),
                "weights": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.n_assets + 1,),
                    dtype=np.float32,
                ),
            }
        )

    def _load_embedding(self, idx: int) -> np.ndarray:
        idx = min(idx, len(self.embedding_files) - 1)

        path = os.path.join(
            self.embedding_dir,
            self.embedding_files[idx]
        )

        embedding = np.load(path).astype(np.float32)

        if embedding.shape != (
            self.n_assets,
            self.embedding_dim,
        ):
            raise ValueError(
                f"Invalid embedding shape at index {idx}: "
                f"{embedding.shape}"
            )

        return embedding

    def _get_observation(self, idx: int) -> dict:
        return {
            "gnn_embeddings": self._load_embedding(idx),
            "weights": self.prev_weights.copy(),
        }