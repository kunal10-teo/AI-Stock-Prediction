import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import load_model


# Load dataset
df = pd.read_csv("data/stock_data.csv")

close_price = df[['Close']]


# Scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(close_price)


# Create test sequence
x_test = []
y_test = []

for i in range(60, len(scaled_data)):
    x_test.append(scaled_data[i-60:i])
    y_test.append(scaled_data[i])


x_test = np.array(x_test)
y_test = np.array(y_test)


# Load LSTM model
model = load_model("models/lstm_model.keras")

# Prediction
predicted = model.predict(x_test)


# Convert back original price
predicted_price = scaler.inverse_transform(predicted)
actual_price = scaler.inverse_transform(y_test)


# Metrics
mae = mean_absolute_error(actual_price, predicted_price)
rmse = np.sqrt(mean_squared_error(actual_price, predicted_price))


print("MAE:", mae)
print("RMSE:", rmse)


# Graph
plt.figure(figsize=(12,6))

plt.plot(actual_price, label="Actual Price")
plt.plot(predicted_price, label="Predicted Price")

plt.title("Stock Price Prediction Using LSTM")
plt.xlabel("Days")
plt.ylabel("Price")

plt.legend()
plt.show()