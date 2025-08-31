
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error , r2_score , make_scorer
from sklearn.preprocessing import StandardScaler

# Load the CSV file
df = pd.read_csv("data/CC_2020-2025_New.csv")
holidays_df = pd.read_csv("data/Holidays.csv")

# Clean and process
df.columns = df.columns.str.strip()
holidays_df.columns = holidays_df.columns.str.strip()
df['QueueStartDate'] = pd.to_datetime(df['QueueStartDate'], dayfirst=True)
holidays_df['CalendarDate'] = pd.to_datetime(holidays_df['CalendarDate'], dayfirst=True,format='%Y-%m-%d')

# Feature engineering
df[['Hour', 'Minute']] = df['HalfHourInterval'].str.extract(r'(\d{2}):(\d{2})').astype(int)
df['Interval'] = df['Hour'] + (df['Minute'] >= 30) * 0.5
df['IsHoliday'] = df['QueueStartDate'].isin(holidays_df['CalendarDate'])
df['IsWeekend'] = df['QueueStartDate'].dt.dayofweek.isin([4, 5])  # 4 = Friday, 5 = Saturday
df['IsWorkDay'] = (~df['IsWeekend']) & (~df['IsHoliday'])

# Features and target
features = df[['QueueStartDateNumber', 'Interval', 'IsHoliday', 'IsWeekend', 'IsWorkDay']]
target = df['TotalAgents']

# Normalize
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split data
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2: {r2:.2f}")

# Save model and scaler
joblib.dump(model, "TotalAgentsModel.pkl")
joblib.dump(scaler, "Scaler.pkl")

# Prediction function
def predict_total_agents(QueueStartDateNumber: int, Interval: float, IsHoliday: bool, IsWeekend: bool, IsWorkDay: bool):
    model = joblib.load("TotalAgentsModel.pkl")
    scaler = joblib.load("Scaler.pkl")
    X_new = pd.DataFrame([[QueueStartDateNumber, Interval, IsHoliday, IsWeekend, IsWorkDay]],
                         columns=['QueueStartDateNumber', 'Interval', 'IsHoliday', 'IsWeekend', 'IsWorkDay'])
    X_new_scaled = scaler.transform(X_new)
    prediction = model.predict(X_new_scaled)[0]
    return round(prediction, 0)

def retrain_model(new_data_path: str, holidays_path: str = "data/Holidays.csv"):
    df = pd.read_csv(new_data_path)
    holidays_df = pd.read_csv(holidays_path)

    df.columns = df.columns.str.strip()
    holidays_df.columns = holidays_df.columns.str.strip()
    df['QueueStartDate'] = pd.to_datetime(df['QueueStartDate'], dayfirst=True)
    holidays_df['CalendarDate'] = pd.to_datetime(holidays_df['CalendarDate'], dayfirst=True,format='%Y-%m-%d')

    df[['Hour', 'Minute']] = df['HalfHourInterval'].str.extract(r'(\d{2}):(\d{2})').astype(int)
    df['Interval'] = df['Hour'] + (df['Minute'] >= 30) * 0.5
    df['IsHoliday'] = df['QueueStartDate'].isin(holidays_df['CalendarDate'])
    df['IsWeekend'] = df['QueueStartDate'].dt.dayofweek.isin([4, 5])
    df['IsWorkDay'] = (~df['IsWeekend']) & (~df['IsHoliday'])

    features = df[['QueueStartDateNumber', 'Interval', 'IsHoliday', 'IsWeekend', 'IsWorkDay']]
    target = df['TotalAgents']

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "TotalAgentsModel.pkl")
    joblib.dump(scaler, "Scaler.pkl")
    print("\n\nModel retrained and saved successfully.")

    # Evaluate
    y_pred = model.predict(features_scaled)
    mae = mean_absolute_error(target, y_pred)
    rmse = np.sqrt(mean_squared_error(target, y_pred))
    r2 = r2_score(target, y_pred)

    print("\nRetrained Model Evaluation:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.2f}")

def load_model(model_path: str = "TotalAgentsModel.pkl", scaler_path: str = "Scaler.pkl"):
    '''Load and return the trained model and scaler.'''
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Example usage
if __name__ == "__main__":
    prediction = predict_total_agents(QueueStartDateNumber=2, Interval=10.5, IsHoliday=False, IsWeekend=False, IsWorkDay=True)
    print(f"\nPredicted TotalAgents: {prediction}")

    retrain_model("data/CC_2020-2025_Retrain_New.csv")
    prediction_retrained = predict_total_agents(QueueStartDateNumber=2, Interval=10.5, IsHoliday=False, IsWeekend=False, IsWorkDay=True)
    print(f"\nPredicted TotalAgents (after retrain): {prediction_retrained}")

    model, scaler = load_model()
    print(f"\nModel and Scaler loaded successfully.")
