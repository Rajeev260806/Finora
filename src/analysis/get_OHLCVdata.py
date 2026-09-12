import yfinance as yf
import pandas as pd
import os

# all tickers sector information is available in project_ideas.md file
tickers = [
    "AAPL","MSFT","NVDA","ADBE",
    "JPM","BAC","GS","V",
    "XOM","CVX","COP","SLB",
    "NEE","DUK","SO","AEP",
    "PLD","AMT","O","SPG"
]

os.makedirs("data/raw", exist_ok=True)

for ticker in tickers:
    try:
        data = yf.download(ticker,start="2015-01-01",end="2025-01-01",auto_adjust=True)
        if data.empty:
            print("No data available for",ticker)
            continue
        filename = ticker+".csv"
        filepath = os.path.join("data/raw",filename)
        df = pd.DataFrame(data)
        df = df.droplevel("Ticker",axis=1)
        df.to_csv(filepath)
        print(ticker,"data's has been created as a seperate csv file")
    except Exception as e:
        print("Something went wrong with",ticker,"'s data")
        print("Error: ",e)
    
