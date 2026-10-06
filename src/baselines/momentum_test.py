import os
import pandas as pd


LOOKBACK = 252
SKIP = 21


def calculate_momentum(returns, end_idx):
    start_idx = end_idx - LOOKBACK
    skip_start = end_idx - SKIP

    if start_idx < 0:
        return None

    momentum_returns = returns.iloc[start_idx:skip_start]

    scores = (1 + momentum_returns).prod() - 1

    return scores.sort_values(ascending=False)


if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    data = pd.read_parquet(
        os.path.join(BASE_DIR, "data", "clean", "full_dataset.parquet")
    )

    returns = data["Returns Daily"].unstack(
        level="Ticker"
    )

    returns = returns.sort_index(axis=1)
    returns = returns.fillna(0.0)

    end_idx = 300

    scores = calculate_momentum(
        returns,
        end_idx
    )

    print("\nMomentum Scores")
    print("----------------")

    print(scores.round(4))

    print("\nTop 5 Momentum Stocks")
    print("---------------------")

    print(scores.head(5).round(4))