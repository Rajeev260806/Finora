import yfinance as yf

data = yf.download(["AAPL","JPM","O"],start="2015-01-01",end="2025-01-01",auto_adjust=True)
print(data)
print(data.shape)