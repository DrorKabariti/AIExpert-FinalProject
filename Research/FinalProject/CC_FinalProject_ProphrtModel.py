# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from itertools import product
from IPython.display import clear_output
import joblib
from prophet import Prophet
from sklearn.model_selection import TimeSeriesSplit

def run_full_eda(cc_file: str, holidays_file: str,
                 weekday_start: int, weekday_end: int,
                 friday_start: int, friday_end: int):
    """
    מבצע EDA מלא ומחזיר DataFrames מוכנים לעבודה
    """
    # --- טעינת הקבצים ---
    cc_df = pd.read_csv(cc_file)
    holidays_df = pd.read_csv(holidays_file)

    # --- המרת תאריכים ---
    cc_df['QueueStartDate'] = pd.to_datetime(cc_df['QueueStartDate'], dayfirst=True, errors='coerce')
    holidays_df['Date'] = pd.to_datetime(holidays_df['Date'], dayfirst=True, errors='coerce')

    # --- שינוי שם עמודה לתאימות ---
    cc_df = cc_df.rename(columns={'QueueStartDate': 'Date'})

    # --- הוספת עמודות חג / ערב חג ---
    cc_df = cc_df.merge(
        holidays_df[['Date', 'IsHoliday', 'IsHolidayEve']],
        on='Date', how='left'
    )
    cc_df['IsHoliday'] = cc_df['IsHoliday'].fillna(0).astype(int)
    cc_df['IsHolidayEve'] = cc_df['IsHolidayEve'].fillna(0).astype(int)

    # --- הוספת יום בשבוע ---
    cc_df['Weekday'] = cc_df['Date'].dt.weekday
    cc_df['WeekdayName'] = cc_df['Date'].dt.day_name()

    # --- יצירת עמודת אינטרוול מתוך HourInterval ---
    if 'HourInterval' in cc_df.columns:
        cc_df['Interval'] = cc_df['HourInterval'].str.split('-').str[0].str.strip()
        cc_df['Interval'] = cc_df['Interval'].str.replace(':00', '').astype(int)

    # --- סינון לפי שעות פעילות ---
    cc_df = cc_df[(
        ((cc_df['Weekday'] < 4) & (cc_df['Interval'].between(weekday_start, weekday_end))) |
        ((cc_df['Weekday'] == 4) & (cc_df['Interval'].between(friday_start, friday_end))) |
        (cc_df['IsHoliday'] == 1)
    )]

    # # --- גרפים בסיסיים ---
    # px.histogram(cc_df, x='TotalAgents', title="התפלגות כמות נציגים").show()
    # px.box(cc_df, x='WeekdayName', y='TotalAgents', title="Boxplot כמות נציגים לפי יום בשבוע").show()
    # px.scatter(cc_df, x='Interval', y='TotalAgents', color='WeekdayName',
    #            title="כמות נציגים לפי אינטרוול ויום בשבוע").show()
    #
    # # --- מטריצת מתאם ---
    # corr_cols = ['TotalAgents', 'Interval', 'Weekday', 'IsHoliday', 'IsHolidayEve']
    # px.imshow(cc_df[corr_cols].corr(), text_auto=True, title="מטריצת מתאם").show()

    return cc_df, holidays_df

def retrain_best_model(cc_file: str, holidays_file: str,
                       weekday_start: int, weekday_end: int,
                       friday_start: int, friday_end: int):
    """
    מאמן מחדש את המודל הטוב ביותר לאחר ביצוע EDA מלא
    :param cc_file: קובץ ה-CSV של הנתונים
    :param holidays_file: קובץ ה-CSV של החגים
    :param weekday_start: שעת פתיחה בימי חול
    :param weekday_end: שעת סגירה בימי חול
    :param friday_start: שעת פתיחה ביום שישי
    :param friday_end: שעת סגירה ביום שישי
    :param model_save_path: נתיב לשמירת המודל המאומן
    :return: המודל המאומן וה-DataFrame המעובד
    """

    # --- שלב 1: EDA מלא ---
    cc_df, holidays_df = run_full_eda(
        cc_file, holidays_file,
        weekday_start, weekday_end,
        friday_start, friday_end
    )
    df = cc_df[['Date', 'Weekday', 'Interval', 'IsHoliday', 'IsHolidayEve', 'TotalAgents']]

    # --- שלב 2: הגדרת Features ו-Target ---
    features = ['Date','Weekday', 'Interval', 'IsHoliday', 'IsHolidayEve']
    target = 'TotalAgents'

    X = cc_df[features]
    y = cc_df[target]

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    holidays_df = holidays_df.rename(columns={'Date': 'ds', 'HolidayName': 'holiday'})
    holidays_df['ds'] = pd.to_datetime(holidays_df['ds'], errors='coerce')

    holidays_df = holidays_df.dropna(subset=['ds', 'holiday'])

    # ניתן להוסיף חלון השפעה (ימים לפני/אחרי החג)
    holidays_df['lower_window'] = 0
    holidays_df['upper_window'] = 0

    prophet_models = {}
    prophet_preds_all, prophet_true_all = [], []
    results = []

    # --- הגדרת שעות פעילות (ניתן לשנות לפי הצורך) ---
    weekday_start, weekday_end = 7, 19
    friday_start, friday_end = 7, 13

    for interval in X_test['Interval'].unique():
    # --- בדיקת עמידה בשעות הפעילות ---
    # שליפת כל התאריכים הרלוונטיים לאינטרוול זה
      dates_for_interval = X_test[X_test['Interval'] == interval]['Date'].unique()
      valid_dates = []
      for date in dates_for_interval:
          weekday = pd.to_datetime(date).weekday()
          if weekday == 5:  # שבת – לא מאמנים מודל
              continue
          if weekday == 4 and not (friday_start <= interval <= friday_end):
              continue
          if (weekday < 4 or weekday == 6) and not (weekday_start <= interval <= weekday_end):
              continue
          valid_dates.append(date)
      if not valid_dates:
          continue

      # --- הכנת הנתונים לאימון ---
      df_int = df[df['Interval'] == interval][['Date', 'TotalAgents']].copy()
      if df_int.empty or df_int['TotalAgents'].nunique() < 2:
          continue

      df_int = df_int.rename(columns={'Date': 'ds', 'TotalAgents': 'y'})
      train_df = df_int[df_int['ds'] < X_test['Date'].max()]
      future_df = df_int[df_int['ds'].isin(valid_dates)]
      if future_df.empty:
          continue

      # ✅ מודל Prophet עם חגים
      m = Prophet(
          daily_seasonality=True,
          weekly_seasonality=True,
          yearly_seasonality=True,
          seasonality_mode='multiplicative',
          changepoint_prior_scale=0.1,
          holidays_prior_scale=10,
          holidays=holidays_df
      )
      m.fit(train_df)
      forecast = m.predict(future_df[['ds']])

      # ✅ מניעת ערכים שליליים והמרה לשלמים
      forecast['yhat'] = forecast['yhat'].clip(lower=0).round().astype(int)


      prophet_preds_all.extend(forecast['yhat'].values)
      prophet_true_all.extend(future_df['y'].values)
      prophet_models[interval] = m

    return prophet_models

def save_model(prophet_models, model_save_path):

    joblib.dump(prophet_models, model_save_path)

    print(f"✅ המודל אומן מחדש ונשמר בנתיב: {model_save_path}")

def load_model(model_save_path):
    prophet_models = joblib.load(model_save_path)
    return prophet_models

def predict_future(model_path: str,
                   future_dates: list,
                   holidays_file: str,
                   weekday_start: int, weekday_end: int,
                   friday_start: int, friday_end: int):
    """
    תחזית עתידית עם מודלי Prophet פר אינטרוול בהתחשב בשעות פעילות, ימי שישי וחגים.
    מחזירה טבלה: Date | Interval (HH:00) | Predicted_TotalAgents (int, שבת=0).
    """
    # --- טעינת מודלים ---
    prophet_models = load_model(model_path)  # {interval: prophet_model}

    # --- טעינת חגים ---
    holidays_df = pd.read_csv(holidays_file)
    holidays_df['Date'] = pd.to_datetime(holidays_df['Date'])

    # --- יצירת טבלת תאריכים ---
    future_df = pd.DataFrame({'Date': pd.to_datetime(future_dates)})
    future_df['Weekday'] = future_df['Date'].dt.weekday
    future_df = future_df.merge(holidays_df[['Date', 'IsHoliday']],
                                on='Date', how='left')
    future_df['IsHoliday'] = future_df['IsHoliday'].fillna(0).astype(int)

    # --- הרחבת אינטרוולים ---
    all_rows = []
    for _, row in future_df.iterrows():
        weekday = row['Weekday']
        is_holiday = row['IsHoliday']

        if weekday == 5:  # שבת בלבד
            for hour in range(weekday_start, weekday_end + 1):
                all_rows.append({
                    'Date': row['Date'],
                    'Interval': hour,
                    'Predicted_TotalAgents': 0
                })
            continue

        if is_holiday == 1:
            hours_range = range(weekday_start, weekday_end + 1)
        elif weekday < 4 or weekday == 6:  # ימים ראשון-חמישי
            hours_range = range(weekday_start, weekday_end + 1)
        elif weekday == 4:  # שישי
            hours_range = range(friday_start, friday_end + 1)
        else:
            continue

        for hour in hours_range:
            all_rows.append({
                'Date': row['Date'],
                'Interval': hour
            })

    pred_df = pd.DataFrame(all_rows)

    # --- תחזית פר אינטרוול ---
    predictions = []
    for interval, group in pred_df.groupby("Interval"):
        if 'Predicted_TotalAgents' in group.columns:  # שבת שכבר מולאה
            predictions.append(group)
            continue

        if interval not in prophet_models:
            group['Predicted_TotalAgents'] = 0
            predictions.append(group)
            continue

        model = prophet_models[interval]
        future_dates_prophet = pd.DataFrame({"ds": group["Date"].unique()})
        forecast = model.predict(future_dates_prophet)

        forecast = forecast[["ds", "yhat"]].rename(columns={"ds": "Date", "yhat": "Predicted_TotalAgents"})
        forecast["Interval"] = interval

        merged = group.merge(forecast, on=["Date", "Interval"], how="left")
        merged["Predicted_TotalAgents"] = merged["Predicted_TotalAgents"].fillna(0).clip(lower=0).round().astype(int)

        predictions.append(merged)

    # --- המרת אינטרוול לפורמט HH:00 ---
    result = pd.concat(predictions, ignore_index=True)[['Date', 'Interval', 'Predicted_TotalAgents']]
    result['Interval'] = result['Interval'].apply(lambda x: f"{int(x):02d}:00")
    result = result.sort_values(by=['Date', 'Interval']).reset_index(drop=True)

    return result

prophet_models = retrain_best_model(
    cc_file="data/CC_2020-2025_New.csv",
    holidays_file="data/Holidays_New.csv",
    weekday_start=7, weekday_end=19,
    friday_start=7, friday_end=13,
)
model_save_path="data/prophet_model_retrained.pkl"
save_model(prophet_models, model_save_path)
future_predictions = predict_future(
    model_path=model_save_path,
    future_dates=['2025-07-01','2025-07-04'],
    holidays_file="data/Holidays_New.csv",
    weekday_start=7, weekday_end=19,
    friday_start=7, friday_end=13
)

print(future_predictions)