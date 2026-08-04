import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# Load data
df = pd.read_csv("data/real_stock_data.csv")


print(df.head())
print(df.columns)


# Remove empty rows
df = df.dropna()


# Convert Close to numeric
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")


df = df.dropna()


# Select Close price
data = df[["Close"]]


print("Close Data:")
print(data.head())
print(data.shape)


# Scaling
scaler = MinMaxScaler()
print(df.head(10))
print(df.columns)
print(df.shape)
print(data.head())

scaled_data = scaler.fit_transform(data)


# Create sequences
X = []
y = []

for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i])
    y.append(scaled_data[i])


X = np.array(X)
y = np.array(y)


print("Training Data Shape:", X.shape)


# LSTM Model

model = Sequential()

model.add(
    LSTM(
        50,
        return_sequences=True,
        input_shape=(X.shape[1],1)
    )
)

model.add(Dropout(0.2))

model.add(
    LSTM(
        50,
        return_sequences=False
    )
)

model.add(Dropout(0.2))


model.add(Dense(1))


# Compile

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)


# Train

model.fit(
    X,
    y,
    epochs=20,
    batch_size=32
)


# Save model

model.save("models/real_lstm_model.keras")


print("Real Stock LSTM Model Saved Successfully!")