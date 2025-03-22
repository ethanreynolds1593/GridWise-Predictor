import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

# Set random seed for reproducibility
np.random.seed(100)

# Define directories
SCRIPT_DIR = os.getcwd()
DATA_DIR = "data/finalized"
MODEL_DIR = "models"

# Load processed datasets
df_hourly = pd.read_csv(os.path.join(DATA_DIR, "finalized_hourly_data.csv"), parse_dates=["Start date"], low_memory=False)
df_daily = pd.read_csv(os.path.join(DATA_DIR, "finalized_daily_data.csv"), parse_dates=["Start date"], low_memory=False)
df_weekly = pd.read_csv(os.path.join(DATA_DIR, "finalized_weekly_data.csv"), parse_dates=["Start date"], low_memory=False)

# Set index to datetime
df_hourly.set_index("Start date", inplace=True)
df_daily.set_index("Start date", inplace=True)
df_weekly.set_index("Start date", inplace=True)

# Compute average price
def compute_average_price(df, columns):
    df["Avg_Price_EUR_MWh"] = df[columns].mean(axis=1)
    return df

price_cols = ["Germany/Luxembourg [/MWh] Original resolutions", "Belgium [/MWh] Original resolutions", "France [/MWh] Original resolutions"]
df_hourly = compute_average_price(df_hourly, price_cols)
df_daily = compute_average_price(df_daily, price_cols)
df_weekly = compute_average_price(df_weekly, price_cols)

# Compute price movement direction
def compute_price_movement(prices, threshold=0.05):
    price_change = prices.pct_change().fillna(0)
    return np.where(price_change > threshold, 1, np.where(price_change < -threshold, 2, 0))

df_hourly["Price_Movement"] = compute_price_movement(df_hourly["Avg_Price_EUR_MWh"])
df_daily["Price_Movement"] = compute_price_movement(df_daily["Avg_Price_EUR_MWh"])
df_weekly["Price_Movement"] = compute_price_movement(df_weekly["Avg_Price_EUR_MWh"])

# Feature engineering
def create_features(df, window_sizes):
    df[f"Rolling_Mean_{window_sizes[0]}"] = df["Avg_Price_EUR_MWh"].rolling(window=window_sizes[0]).mean()
    df[f"Price_Change_{window_sizes[1]}"] = df["Avg_Price_EUR_MWh"].pct_change(window_sizes[1]) * 100
    df[f"Lag_{window_sizes[2]}"] = df["Avg_Price_EUR_MWh"].shift(window_sizes[2])
    df.fillna(0, inplace=True)
    return df

df_hourly = create_features(df_hourly, [24, 1, 1])
df_daily = create_features(df_daily, [7, 1, 1])
df_weekly = create_features(df_weekly, [4, 1, 1])

# Select features
features_hourly = ["Rolling_Mean_24", "Price_Change_1", "Lag_1"]
features_daily = ["Rolling_Mean_7"]
features_weekly = ["Rolling_Mean_4"]

def preprocess_data(df, features, label):
    X, y = df[features], df[label]
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)
    return X, y

X_hourly, y_hourly = preprocess_data(df_hourly, features_hourly, "Price_Movement")
X_daily, y_daily = preprocess_data(df_daily, features_daily, "Price_Movement")
X_weekly, y_weekly = preprocess_data(df_weekly, features_weekly, "Price_Movement")

# Normalize features
scaler = MinMaxScaler()
X_hourly_scaled = scaler.fit_transform(X_hourly)
X_daily_scaled = scaler.fit_transform(X_daily)
X_weekly_scaled = scaler.fit_transform(X_weekly)

# Split dataset
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_hourly_scaled, y_hourly, test_size=0.2, random_state=42, stratify=y_hourly)
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_daily_scaled, y_daily, test_size=0.2, random_state=42, stratify=y_daily)
X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(X_weekly_scaled, y_weekly, test_size=0.2, random_state=42, stratify=y_weekly)

# Handle class imbalance
smote = SMOTE(random_state=42)
X_train_h, y_train_h = smote.fit_resample(X_train_h, y_train_h)

# Train models
lgb_classifier = lgb.LGBMClassifier()
xgb_classifier = xgb.XGBClassifier()
lgb_classifier.fit(X_train_h, y_train_h)
xgb_classifier.fit(X_train_h, y_train_h)

# Predictions
y_pred_lgb = lgb_classifier.predict(X_test_h)
y_pred_xgb = xgb_classifier.predict(X_test_h)

# Ensemble method
y_pred_ensemble = np.round((y_pred_lgb + y_pred_xgb) / 2)

# Evaluation
def evaluate_predictions(y_true, y_pred):
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Mean Absolute Error: {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    actual_volatility = np.std(y_true)
    predicted_volatility = np.std(y_pred)
    volatility_capture = 1 - abs(actual_volatility - predicted_volatility) / actual_volatility if actual_volatility != 0 else 0
    print(f"Volatility Capture Score: {volatility_capture:.4f}")
    extreme_moves = (y_true.abs() > 0.15).sum()
    extreme_correct = ((y_true.abs() > 0.15) & (np.abs(y_pred) > 0.15)).sum()
    extreme_accuracy = extreme_correct / extreme_moves if extreme_moves > 0 else 0
    print(f"Extreme Price Movement Accuracy: {extreme_accuracy:.4f}")

evaluate_predictions(y_test_h, y_pred_ensemble)

# Save models
joblib.dump(lgb_classifier, os.path.join(MODEL_DIR, "lgb_price_model.pkl"))
joblib.dump(xgb_classifier, os.path.join(MODEL_DIR, "xgb_price_model.pkl"))

print("✅ Model training and evaluation completed!")

# Accuracy: 0.9969
# Mean Absolute Error: 0.0042
# Root Mean Squared Error: 0.0793
# Volatility Capture Score: 0.9974
# Extreme Price Movement Accuracy: 0.9919
