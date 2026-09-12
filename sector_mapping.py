import pandas as pd

sector_map = {
    "AAPL": "Technology",
    "ADBE": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "V": "Financials",

    "AEP": "Utilities",
    "DUK": "Utilities",
    "NEE": "Utilities",
    "SO": "Utilities",

    "AMT": "Real Estate",
    "O": "Real Estate",
    "PLD": "Real Estate",
    "SPG": "Real Estate",

    "BAC": "Financials",
    "GS": "Financials",
    "JPM": "Financials",

    "COP": "Energy",
    "CVX": "Energy",
    "SLB": "Energy",
    "XOM": "Energy"
}

clusters = [
    ["AAPL", "ADBE", "MSFT", "NVDA", "V"],
    ["AEP", "DUK", "NEE", "SO"],
    ["AMT", "O", "PLD", "SPG"],
    ["BAC", "GS", "JPM"],
    ["COP", "CVX", "SLB", "XOM"]
]

print("\nSector vs Correlation Clusters")
print("-----------------------------")

for i, cluster in enumerate(clusters, 1):

    print(f"\nCluster {i}:")

    for ticker in cluster:
        print(
            f"{ticker} -> {sector_map[ticker]}"
        )