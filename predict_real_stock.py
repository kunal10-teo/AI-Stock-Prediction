import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_csv("data/real_stock_data.csv")

# Close prices
close_prices = df["Close"].values.reshape(-1, 1)

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Prepare last 60 days
last_60 = scaled_data[-60:]
X_test = np.array([last_60])

# Load trained model
model = load_model("models/real_lstm_model.keras")

# Predict
prediction = model.predict(X_test)

# Convert back to original price
predicted_price = scaler.inverse_transform(prediction)

print("=" * 40)
print("Predicted Next Day Closing Price")
print(f"₹ {predicted_price[0][0]:.2f}")
print("=" * 40)

# Plot last 60 days
plt.figure(figsize=(10, 5))
plt.plot(close_prices[-60:], label="Last 60 Days Close")
plt.scatter(60, predicted_price[0][0], label="Predicted Next Day")
plt.legend()
plt.title("Next Day Stock Price Prediction")
plt.show()