import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load dataset
df = pd.read_csv("data/AAPL_processed.csv")

# Close price
data = df["Close"].values.reshape(-1, 1)

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

sequence_length = 60

X = []
y = []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i])

X = np.array(X)
y = np.array(y)

# Build model
model = Sequential()

model.add(LSTM(50, return_sequences=True,
               input_shape=(X.shape[1], 1)))

model.add(LSTM(50))

model.add(Dense(25))
model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

# Train model
model.fit(
    X,
    y,
    epochs=10,
    batch_size=32
)

# Save model
model.save("models/lstm_model.keras")

print("LSTM Model Saved Successfully!")