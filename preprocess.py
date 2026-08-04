import pandas as pd

# Load dataset
df = pd.read_csv("data/AAPL.csv")

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"])

# Sort by Date
df = df.sort_values("Date")

# Create Moving Averages
df["SMA_20"] = df["Close"].rolling(window=20).mean()
df["SMA_50"] = df["Close"].rolling(window=50).mean()

# Create Daily Return
df["Daily_Return"] = df["Close"].pct_change()

# Remove missing values
df = df.dropna()

# Save processed dataset
df.to_csv("data/AAPL_processed.csv", index=False)

print(df.head())
print("\n✅ Processed dataset saved successfully!")

