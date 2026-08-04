import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Load processed dataset
df = pd.read_csv("data/AAPL_processed.csv")

# Features
X = df[["Open", "High", "Low", "Volume", "SMA_20", "SMA_50"]]

# Target
y = df["Close"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Evaluation
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

# Save model
joblib.dump(model, "models/random_forest.pkl")

print("✅ Random Forest model saved successfully!")