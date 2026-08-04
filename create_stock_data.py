import pandas as pd
import numpy as np
from datetime import date, timedelta

np.random.seed(42)

days = 300

dates = [
    date(2025, 1, 1) + timedelta(days=i)
    for i in range(days)
]

close = np.cumsum(np.random.normal(0, 2, days)) + 250

open_price = close + np.random.normal(0, 1, days)

high = np.maximum(open_price, close) + np.random.uniform(0, 5, days)

low = np.minimum(open_price, close) - np.random.uniform(0, 5, days)

volume = np.random.randint(10000, 100000, days)


df = pd.DataFrame({
    "Date": dates,
    "Open": open_price,
    "High": high,
    "Low": low,
    "Close": close,
    "Volume": volume
})


df.to_csv("data/stock_data.csv", index=False)

print("stock_data.csv created successfully!")