import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from random import gauss
from random import seed
from pandas import Series
from pandas.plotting import autocorrelation_plot
from matplotlib import pyplot
from plotly.subplots import make_subplots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet.plot import add_changepoints_to_plot
from prophet.diagnostics import cross_validation, performance_metrics
from prophet import Prophet
from statsmodels.tsa.stattools import acf
import plotly.graph_objects as go
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error , r2_score ,  make_scorer
import itertools
import warnings
warnings.filterwarnings("ignore")
import joblib

# Load the CSV file
df = pd.read_csv("data/CC_2020-2025_New.csv")
holidays_df = pd.read_csv("data/Holidays.csv")

# Basic exploration
# df_info = df.info()
# df_head = df.head()
# df_description = df.describe(include='all')
#
# # Return key parts of the analysis
# df_columns = df.columns.tolist()
# df_shape = df.shape
# df_columns, df_shape, df_head


# ✅ מאפיינים עיקריים:
# שורות: 33,601
# עמודות: 12

# | עמודה                    | תיאור                                                                      |
# | ------------------------ | -------------------------------------------------------------------------- |
# | **QueueStartDate**       | תאריך התחלה של השיחה בתור (בפורמט `yyyy-mm-dd`).                           |
# | **QueueStartDateNumber** | ייצוג מספרי של התאריך (לרוב בפורמט `yyyymmdd`).                            |
# | **QueueStartDateName**   | שם היום בשבוע שבו בוצעה השיחה (למשל: 'Sunday', 'Monday').                  |
# | **HourInterval**         | אינטרוול של שעה (למשל 08, 09... מציין את תחילת השעה).                      |
# | **HalfHourInterval**     | אינטרוול של חצי שעה (למשל 08.0, 08.5, 09.0 וכו'). מציין את תחילת חצי השעה. |
# | **TotalCallsOffered**    | מספר השיחות שהוצעו למוקד (נכנסות).                                         |
# | **TotalCallsAnswered**   | מספר השיחות שנענו בפועל.                                                   |
# | **TotalCallsAbandoned**  | מספר השיחות שננטשו (הלקוח ניתק לפני מענה).                                 |
# | **TotalCB**              | מספר ה־Callback שנרשמו (לקוחות שביקשו שיחזרו אליהם).                       |
# | **TotalTransfered**      | מספר השיחות שהועברו לנציג אחר/מוקד אחר.                                    |
# | **TotalWaitDuration**    | זמן ההמתנה הכולל (בשניות או ביחידת זמן אחרת – נבדוק אם תרצה).              |
# | **TotalAgents**          | מספר הנציגים הפעילים באותו אינטרוול זמן. *(זו עמודת היעד לחיזוי)*          |


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

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

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

# Cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_mae = cross_val_score(model, features, target, cv=cv, scoring=make_scorer(mean_absolute_error))
cv_rmse = cross_val_score(model, features, target, cv=cv, scoring=lambda m, X, y: np.sqrt(mean_squared_error(y, m.predict(X))))
cv_r2 = cross_val_score(model, features, target, cv=cv, scoring=make_scorer(r2_score))

print("\nCross-Validation:")
print(f"MAE (5-fold): {cv_mae.mean():.2f} ± {cv_mae.std():.2f}")
print(f"RMSE (5-fold): {cv_rmse.mean():.2f} ± {cv_rmse.std():.2f}")
print(f"R2 (5-fold): {cv_r2.mean():.2f} ± {cv_r2.std():.2f}")



# # Plot predictions vs actual
# plt.figure(figsize=(10, 5))
# plt.scatter(y_test, y_pred, alpha=0.3)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
# plt.xlabel('Actual TotalAgents')
# plt.ylabel('Predicted TotalAgents')
# plt.title('Prediction vs Actual')
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# Save model
joblib.dump(model, "TotalAgentsModel.pkl")

# Prediction function
def predict_total_agents(QueueStartDateNumber: int, Interval: float, IsHoliday: bool, IsWeekend: bool, IsWorkDay: bool):
    model = joblib.load("TotalAgentsModel.pkl")
    X_new = pd.DataFrame([[QueueStartDateNumber, Interval, IsHoliday, IsWeekend, IsWorkDay]],
                         columns=['QueueStartDateNumber', 'Interval', 'IsHoliday', 'IsWeekend', 'IsWorkDay'])
    prediction = model.predict(X_new)[0]
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
    df['IsWeekend'] = df['QueueStartDate'].dt.dayofweek.isin([4, 5])  # 4 = Friday, 5 = Saturday
    df['IsWorkDay'] = (~df['IsWeekend']) & (~df['IsHoliday'])

    features = df[['QueueStartDateNumber', 'Interval', 'IsHoliday', 'IsWeekend', 'IsWorkDay']]
    target = df['TotalAgents']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "TotalAgentsModel.pkl")
    print("\n\nModel retrained and saved successfully.")

    # Evaluate
    y_pred = model.predict(features)
    mae = mean_absolute_error(target, y_pred)
    rmse = np.sqrt(mean_squared_error(target, y_pred))
    r2 = r2_score(target, y_pred)

    print("\nRetrained Model Evaluation:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.2f}")



def load_model(model_path: str = "TotalAgentsModel.pkl"):
    """Load and return the trained model."""
    return joblib.load(model_path)
# Example usage
if __name__ == "__main__":
    prediction = predict_total_agents(QueueStartDateNumber=2, Interval=10.5, IsHoliday=False, IsWeekend=False, IsWorkDay=True)
    print(f"\nPredicted TotalAgents: {prediction}")

    # Example retraining
    retrain_model("data/CC_2020-2025_Retrain_New.csv")
    prediction_retrained = predict_total_agents(QueueStartDateNumber=2, Interval=10.5, IsHoliday=False, IsWeekend=False, IsWorkDay=True)
    print(f"\nPredicted TotalAgents: {prediction_retrained}")

    model = load_model()
    print(f"\nTotalAgents Model loaded successfuly")
