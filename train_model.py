import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load dataset
df = pd.read_csv("AmesHousing.csv")
df.columns = df.columns.str.replace(" ", "").str.strip()

# Select features and target
selected_features = [
    "LotArea", "YearBuilt", "OverallQual", "OverallCond",
    "GrLivArea", "GarageCars", "GarageArea", "FullBath", "TotRmsAbvGrd"
]
target = "SalePrice"

# Drop missing values
df = df[selected_features + [target]].dropna()

# LOG TRANSFORM the target
df[target] = np.log1p(df[target])  # log(1 + SalePrice)

# Remove outliers (optional, helps improve quality)
df = df[(df["LotArea"] < 100000) & (df["GrLivArea"] < 5000) & (df["GarageArea"] < 1500)]

# Split into X, y
X = df[selected_features]
y = df[target]

# Train-test split (to check performance)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))  # inverse of log1p
print(f"✅ MAE (in actual $): ${mae:,.2f}")

# Save everything
joblib.dump(model, "xgb_model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(selected_features, "features.joblib")
print("📦 Model and scalers saved.")
