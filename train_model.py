import yfinance as yf
import pandas as pd

stock = "AAPL"

df = yf.download(stock, start="2020-01-01", end="2026-01-01")

# MultiIndex columns ko normal columns banao
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Date ko column banao
df = df.reset_index()

# Save CSV
df.to_csv("data/AAPL.csv", index=False)

print(df.head())
print(df.columns)
print("Dataset saved successfully!")