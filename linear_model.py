import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load processed data
df = pd.read_csv("data/AAPL_processed.csv")

# Features
X = df[["Open", "High", "Low", "Volume", "SMA_20", "SMA_50"]]

# Target
y = df["Close"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Accuracy
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# Save model
joblib.dump(model, "models/linear_model.pkl")

print("Model saved successfully!")