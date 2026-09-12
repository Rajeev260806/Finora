import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def calculate_correlation(data):
    returns = data["Returns Daily"].unstack(level="Ticker")
    returns = returns.sort_index(axis=1)

    correlation = returns.corr()

    return correlation


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python correlation_analysis.py <path to full_dataset.parquet>")
    data = pd.read_parquet(sys.argv[1])

    correlation = calculate_correlation(data)

    print("\nCorrelation Matrix")
    print("------------------")
    print(correlation.round(2))

    correlation.to_csv("correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(10, 8))

    image = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)

    ax.set_xticks(range(len(correlation.columns)))
    ax.set_yticks(range(len(correlation.columns)))

    ax.set_xticklabels(correlation.columns, rotation=90)
    ax.set_yticklabels(correlation.columns)

    fig.colorbar(image, ax=ax)

    ax.set_title("Stock Return Correlation Matrix")

    fig.tight_layout()

    fig.savefig(
        "correlation_matrix.png",
        dpi=150
    )

    print("\nSaved:")
    print("correlation_matrix.csv")
    print("correlation_matrix.png")