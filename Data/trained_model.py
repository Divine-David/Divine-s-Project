# Import libraries
import pandas as pd
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("Data/REAL ESTATE.csv")

# -------- STEP 1: ESTIMATE TIME TO PROFIT -------- #
base_rent_per_sqft = 100

def calculate_rent_multiplier(row):
    multiplier = 1.0
    multiplier += 0.1 * row['bedrooms']
    multiplier += 0.1 * row['bathrooms']
    multiplier += 0.05 * row['stories']
    multiplier += 0.05 * row['parking']
    multiplier += 0.1 * row['guestroom']
    multiplier += 0.1 * row['basement']
    multiplier += 0.15 * row['airconditioning']
    multiplier += 0.1 * row['prefarea']
    multiplier += 0.1 * row['furnishingstatus']
    multiplier += 0.05 * row['hotwaterheating']
    return multiplier

df["rent_multiplier"] = df.apply(calculate_rent_multiplier, axis=1)
df["estimated_annual_rent"] = base_rent_per_sqft * df["area"] * df["rent_multiplier"]
df["time_to_profit_months"] = (df["price"] / df["estimated_annual_rent"] * 12).clip(upper=360)

# -------- STEP 2: CLASSIFICATION MODEL -------- #
median_price = df["price"].median()
df["profitable"] = (df["price"] > median_price).astype(int)

X_cls = df.drop(columns=["price", "profitable", "estimated_annual_rent", "time_to_profit_months", "rent_multiplier"])
y_cls = df["profitable"]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
scaler_cls = StandardScaler()
Xc_train_scaled = scaler_cls.fit_transform(Xc_train)
Xc_test_scaled = scaler_cls.transform(Xc_test)

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(Xc_train_scaled, yc_train)
yc_pred = classifier.predict(Xc_test_scaled)

print("\n--- Classification Report ---")
print("Accuracy:", accuracy_score(yc_test, yc_pred))
print(classification_report(yc_test, yc_pred))

# -------- STEP 3: REGRESSION MODEL -------- #
X_reg = df.drop(columns=["price", "profitable", "estimated_annual_rent", "time_to_profit_months", "rent_multiplier"])
y_reg = df["time_to_profit_months"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
scaler_reg = StandardScaler()
Xr_train_scaled = scaler_reg.fit_transform(Xr_train)
Xr_test_scaled = scaler_reg.transform(Xr_test)

regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(Xr_train_scaled, yr_train)
yr_pred = regressor.predict(Xr_test_scaled)

print("\n--- Regression Report ---")
print("MSE:", mean_squared_error(yr_test, yr_pred))
print("R² Score:", r2_score(yr_test, yr_pred))

# -------- STEP 4: SAVE MODELS AND METADATA -------- #

joblib.dump(classifier, "Model/classifier_model.pkl")
joblib.dump(scaler_cls, "Model/classifier_scaler.pkl")

joblib.dump(regressor, "Model/regressor_model.pkl")
joblib.dump(scaler_reg, "Model/regressor_scaler.pkl")

model_info = {
    "classification_model": "classifier_model.pkl",
    "classification_scaler": "classifier_scaler.pkl",
    "regression_model": "regressor_model.pkl",
    "regression_scaler": "regressor_scaler.pkl",
    "regression_target": "time_to_profit_months"
}

with open("Model/model_info.json", "w") as f:
    json.dump(model_info, f, indent=4)

print("\n✅ All models and metadata saved in 'Model/' folder.")
