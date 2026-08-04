import yfinance as yf
import pandas as pd

stock = "RELIANCE.NS"

df = yf.download(
    stock,
    start="2020-01-01",
    end="2026-08-01",
    auto_adjust=False
)

if df.empty:
    print("No data downloaded")
    exit()


if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)


df = df.reset_index()

df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]


df.to_csv(
    "data/real_stock_data.csv",
    index=False
)

print("Real Stock Data Saved Successfully!")
print(df.head())