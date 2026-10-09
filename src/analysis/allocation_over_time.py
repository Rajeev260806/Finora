import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_FILE = (
    PROJECT_ROOT / "results" / "baselines" /
    "mean_variance" / "weights.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "allocation_over_time.png"

# Load actual portfolio weights
df = pd.read_csv(WEIGHTS_FILE)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# Extract stock and cash allocation columns
asset_columns = [
    col for col in df.columns
    if col != "date"
]

# Create stacked area chart
plt.figure(figsize=(16, 8))

plt.stackplot(
    df["date"],
    *[df[col] for col in asset_columns],
    labels=asset_columns,
    alpha=0.85
)

plt.title(
    "Mean-Variance Portfolio Allocation Over Time",
    fontsize=16
)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Portfolio Weight", fontsize=12)
plt.ylim(0, 1.05)
plt.legend(
    loc="upper left",
    bbox_to_anchor=(1.01, 1),
    fontsize=9
)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Allocation chart created successfully!")
print(f"Output: {OUTPUT_FILE}")
print(f"Trading days: {len(df)}")
print(f"Assets plotted: {len(asset_columns)}")
