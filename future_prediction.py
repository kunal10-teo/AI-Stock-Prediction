import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


# Load data
df = pd.read_csv("data/stock_data.csv")


# Select Close price
data = df[['Close']]


# Scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)


# Last 60 days data
last_60_days = scaled_data[-60:]

# Reshape for LSTM
X_test = np.array([last_60_days])


# Load trained model
model = load_model("models/lstm_model.keras")


# Prediction
prediction = model.predict(X_test)


# Convert back original price
predicted_price = scaler.inverse_transform(prediction)


print("Predicted Next Day Stock Price:")
print(predicted_price[0][0])