import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
import joblib
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st
import os

warnings.filterwarnings("ignore")
# Run Example: streamlit run FinalProjectApp_enforced.py

# -----------------------------
# Helpers
# -----------------------------
WEEKDAY_START_DEFAULT = 7
WEEKDAY_END_DEFAULT   = 19
FRIDAY_START_DEFAULT  = 7
FRIDAY_END_DEFAULT    = 13

def _parse_dates(df, date_col, dayfirst=True):
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=dayfirst, errors='coerce')
    return df

def _coalesce_flags(df):
    for c in ['IsHoliday', 'IsHolidayEve']:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int)
        else:
            df[c] = 0
    return df

# =============================================================================
# EDA / Training
# =============================================================================
def run_full_eda(cc_file_buffer, holidays_file_buffer,
                 weekday_start: int = WEEKDAY_START_DEFAULT, weekday_end: int = WEEKDAY_END_DEFAULT,
                 friday_start: int = FRIDAY_START_DEFAULT, friday_end: int = FRIDAY_END_DEFAULT):
    # Load
    cc_df = pd.read_csv(cc_file_buffer)
    holidays_df = pd.read_csv(holidays_file_buffer)

    # Parse dates
    cc_df = _parse_dates(cc_df, 'QueueStartDate', dayfirst=True).rename(columns={'QueueStartDate': 'Date'})
    holidays_df = _parse_dates(holidays_df, 'Date', dayfirst=True)

    # Merge holiday flags
    cc_df = cc_df.merge(holidays_df[['Date', 'IsHoliday', 'IsHolidayEve']], on='Date', how='left')
    cc_df = _coalesce_flags(cc_df)

    # Add weekday fields
    cc_df['Weekday'] = cc_df['Date'].dt.weekday  # Mon=0 ... Sun=6
    cc_df['WeekdayName'] = cc_df['Date'].dt.day_name()

    # Parse Interval from HourInterval "HH:00 - HH:00" -> HH (int)
    if 'HourInterval' in cc_df.columns:
        cc_df['Interval'] = cc_df['HourInterval'].str.split('-').str[0].str.strip()
        cc_df['Interval'] = cc_df['Interval'].str.replace(':00', '', regex=False).astype(int)

    # -------- Business rules for TRAINING data --------
    # 1) Exclude Holidays entirely from training (no work on holidays)
    # 2) Treat Holiday Eve like Friday (07-13)
    # 3) Weekdays Mon-Thu + Sun => 07-19
    # 4) Friday => 07-13
    # 5) Saturday excluded from training (no work)
    mask_weekdays = (
        (cc_df['IsHoliday'].eq(0)) &
        (cc_df['IsHolidayEve'].eq(0)) &
        ((cc_df['Weekday'] < 4) | (cc_df['Weekday'] == 6)) &  # Mon-Thu (0-3) and Sun (6)
        (cc_df['Interval'].between(weekday_start, weekday_end))
    )

    mask_friday = (
        (cc_df['IsHoliday'].eq(0)) &
        (cc_df['Weekday'] == 4) &  # Friday
        (cc_df['Interval'].between(friday_start, friday_end))
    )

    mask_eve = (
        (cc_df['IsHoliday'].eq(0)) &
        (cc_df['IsHolidayEve'].eq(1)) &
        (cc_df['Interval'].between(friday_start, friday_end))
    )

    # Combine masks; exclude Saturdays (5) and Holidays (handled above) from training
    cc_df = cc_df[mask_weekdays | mask_friday | mask_eve].copy()

    return cc_df, holidays_df

# === Tuned hyper-params per Interval (from your tuning summary) ===
TUNED_PARAMS = {
    7:  {'mode': 'multiplicative', 'cps': 0.3,  'sps': 10, 'hps': 5},
    8:  {'mode': 'additive',       'cps': 0.1,  'sps': 10, 'hps': 5},
    9:  {'mode': 'additive',       'cps': 0.1,  'sps':  5, 'hps': 1},
    10: {'mode': 'additive',       'cps': 0.05, 'sps':  1, 'hps': 5},
    11: {'mode': 'additive',       'cps': 0.1,  'sps':  1, 'hps': 5},
    12: {'mode': 'additive',       'cps': 0.1,  'sps':  5, 'hps': 1},
    13: {'mode': 'additive',       'cps': 0.1,  'sps':  1, 'hps': 1},
    14: {'mode': 'additive',       'cps': 0.1,  'sps': 10, 'hps': 1},
    15: {'mode': 'multiplicative', 'cps': 0.3,  'sps': 10, 'hps': 1},
    16: {'mode': 'additive',       'cps': 0.3,  'sps':  5, 'hps': 1},
    17: {'mode': 'additive',       'cps': 0.05, 'sps':  5, 'hps': 1},
    18: {'mode': 'additive',       'cps': 0.1,  'sps':  5, 'hps': 1},
    19: {'mode': 'multiplicative', 'cps': 0.1,  'sps':  5, 'hps': 5},  # UsedDefault=TRUE
}
# ברירת מחדל אם חסר אינטרוול:
TUNED_FALLBACK = {'mode': 'multiplicative', 'cps': 0.1, 'sps': 5.0, 'hps': 5.0}

def retrain_best_model(cc_file_buffer, holidays_file_buffer,
                       weekday_start: int = WEEKDAY_START_DEFAULT, weekday_end: int = WEEKDAY_END_DEFAULT,
                       friday_start: int = FRIDAY_START_DEFAULT, friday_end: int = FRIDAY_END_DEFAULT):
    # הכנת דאטה עם כללי פעילות
    cc_df, holidays_df_orig = run_full_eda(
        cc_file_buffer, holidays_file_buffer,
        weekday_start, weekday_end, friday_start, friday_end
    )

    df = cc_df[['Date', 'Interval', 'TotalAgents']].copy()

    # טבלת חגים ל-Prophet
    holidays_df = holidays_df_orig.rename(columns={'Date': 'ds', 'HolidayName': 'holiday'})
    holidays_df['ds'] = pd.to_datetime(holidays_df['ds'], errors='coerce')
    holidays_df = holidays_df.dropna(subset=['ds', 'holiday'])
    if not holidays_df.empty:
        holidays_df['lower_window'] = 0
        holidays_df['upper_window'] = 0

    prophet_models = {}
    unique_intervals = sorted(df['Interval'].unique())
    progress_bar = st.progress(0, text="מתחיל אימון (עם פרמטרים מכוונים)...")

    for i, interval in enumerate(unique_intervals):
        progress_bar.progress((i + 1) / max(1, len(unique_intervals)),
                              text=f"מאמן אינטרוול {int(interval):02d}:00")

        dfi = df[df['Interval'] == interval][['Date', 'TotalAgents']].copy()
        if dfi.empty or dfi['TotalAgents'].nunique() < 2:
            continue

        # Winsorize עדין להקטנת השפעת חריגים
        lo, hi = dfi['TotalAgents'].quantile([0.01, 0.99])
        dfi['TotalAgents'] = dfi['TotalAgents'].clip(lo, hi)

        dfi = dfi.rename(columns={'Date': 'ds', 'TotalAgents': 'y'}).sort_values('ds')

        # קביעת שימוש בעונתיות שנתית לפי אורך ההיסטוריה
        span_years = (dfi['ds'].max() - dfi['ds'].min()).days / 365.25
        use_yearly = span_years >= 2.0

        # פרמטרים מכוונים לפי הטבלה שסיפקת (עם fallback)
        cfg = TUNED_PARAMS.get(int(interval), TUNED_FALLBACK)
        mode, cps, sps, hps = cfg['mode'], cfg['cps'], cfg['sps'], cfg['hps']

        # מודל Prophet בהתאם להגדרות
        m = Prophet(
            growth='linear',
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=use_yearly,
            seasonality_mode=mode,
            changepoint_prior_scale=cps,
            seasonality_prior_scale=sps,
            holidays_prior_scale=hps,
            holidays=holidays_df if not holidays_df.empty else None,
            changepoint_range=0.9
        )
        # רכיב שבועי “חזק” ללכידת דפוסים של ימי השבוע
        m.add_seasonality(name='weekly_high', period=7, fourier_order=10, prior_scale=sps)

        # אימון
        m.fit(dfi)
        prophet_models[int(interval)] = m

    return prophet_models


def save_model(prophet_models, model_save_path):
    joblib.dump(prophet_models, model_save_path)


def load_model(model_file_buffer):
    return joblib.load(model_file_buffer)


# =============================================================================
# Prediction
# =============================================================================
def predict_future(prophet_models, future_dates: list, holidays_file_buffer,
                   weekday_start: int = WEEKDAY_START_DEFAULT, weekday_end: int = WEEKDAY_END_DEFAULT,
                   friday_start: int = FRIDAY_START_DEFAULT, friday_end: int = FRIDAY_END_DEFAULT):
    holidays_df = pd.read_csv(holidays_file_buffer)
    holidays_df['Date'] = pd.to_datetime(holidays_df['Date'])
    holidays_flags = holidays_df[['Date', 'IsHoliday', 'IsHolidayEve']].copy()
    holidays_flags = _coalesce_flags(holidays_flags)

    # Build the target date table
    future_df = pd.DataFrame({'Date': pd.to_datetime(future_dates)})
    future_df['Weekday'] = future_df['Date'].dt.weekday
    future_df = future_df.merge(holidays_flags, on='Date', how='left')
    future_df = _coalesce_flags(future_df)

    all_rows = []
    for _, row in future_df.iterrows():
        weekday = int(row['Weekday'])       # Mon=0 ... Sun=6
        is_holiday = int(row['IsHoliday'])
        is_eve = int(row['IsHolidayEve'])

        # Saturday or Holiday => explicit zeros
        if weekday == 5 or is_holiday == 1:
            for hour in range(weekday_start, weekday_end + 1):
                all_rows.append({'Date': row['Date'], 'Interval': hour, 'force_zero': 1})
            continue

        # Friday or Holiday Eve => 07-13
        if weekday == 4 or is_eve == 1:
            hours_range = range(friday_start, friday_end + 1)
        # Weekdays Mon-Thu or Sunday => 07-19
        elif (weekday < 4) or (weekday == 6):
            hours_range = range(weekday_start, weekday_end + 1)
        else:
            # Should not happen, but safe-guard
            continue

        for hour in hours_range:
            all_rows.append({'Date': row['Date'], 'Interval': hour, 'force_zero': 0})

    if not all_rows:
        return pd.DataFrame(columns=['Date', 'Interval', 'Predicted_TotalAgents'])

    pred_df = pd.DataFrame(all_rows)

    # Generate predictions per Interval
    pieces = []
    for interval, group in pred_df.groupby("Interval"):
        group = group.copy()

        # Rows that must be zero (Saturday/Holiday)
        zero_rows = group[group['force_zero'] == 1].copy()
        if not zero_rows.empty:
            zero_rows['Predicted_TotalAgents'] = 0
            pieces.append(zero_rows[['Date', 'Interval', 'Predicted_TotalAgents']])

        # Rows to predict
        pred_rows = group[group['force_zero'] == 0].copy()
        if not pred_rows.empty:
            if interval not in prophet_models:
                pred_rows['Predicted_TotalAgents'] = 0
                pieces.append(pred_rows[['Date', 'Interval', 'Predicted_TotalAgents']])
            else:
                model = prophet_models[interval]
                future_dates_prophet = pd.DataFrame({"ds": pred_rows["Date"].unique()})
                forecast = model.predict(future_dates_prophet)
                forecast['yhat'] = forecast['yhat'].clip(lower=0).round().astype(int)
                forecast = forecast[["ds", "yhat"]].rename(columns={"ds": "Date", "yhat": "Predicted_TotalAgents"})
                forecast["Interval"] = interval
                merged = pred_rows.merge(forecast, on=["Date", "Interval"], how="left")
                merged["Predicted_TotalAgents"] = merged["Predicted_TotalAgents"].fillna(0).astype(int)
                pieces.append(merged[['Date', 'Interval', 'Predicted_TotalAgents']])

    if not pieces:
        return pd.DataFrame(columns=['Date', 'Interval', 'Predicted_TotalAgents'])

    result = pd.concat(pieces, ignore_index=True)
    result['Interval'] = result['Interval'].apply(lambda x: f"{int(x):02d}:00")
    result['Date_str'] = result['Date'].dt.strftime('%d/%m/%Y')
    result = result.sort_values(by=['Date', 'Interval']).reset_index(drop=True)
    return result

# =============================================================================
# Evaluation (Metrics & Charts)
# =============================================================================
def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

def evaluate_models(prophet_models, cc_file_buffer, holidays_file_buffer,
                    val_days: int = 14,
                    weekday_start: int = WEEKDAY_START_DEFAULT, weekday_end: int = WEEKDAY_END_DEFAULT,
                    friday_start: int = FRIDAY_START_DEFAULT, friday_end: int = FRIDAY_END_DEFAULT):
    # Prepare data (already enforces business rules)
    cc_df, _ = run_full_eda(cc_file_buffer, holidays_file_buffer,
                            weekday_start, weekday_end, friday_start, friday_end)

    # Choose validation window: last val_days with data
    unique_dates = sorted(cc_df['Date'].dropna().unique())
    if len(unique_dates) == 0:
        return pd.DataFrame(), {}, pd.DataFrame()
    val_days = min(val_days, len(unique_dates))
    val_dates = unique_dates[-val_days:]
    df_val = cc_df[cc_df['Date'].isin(val_dates)].copy()

    # Predict per interval for those dates
    preds = []
    for interval, g in df_val.groupby('Interval'):
        model = prophet_models.get(int(interval))
        if model is None:
            # if model missing, fall back to zeros
            tmp = g[['Date', 'Interval', 'TotalAgents']].copy()
            tmp['y_pred'] = 0
            preds.append(tmp.rename(columns={'TotalAgents': 'y_true'}))
            continue
        future = pd.DataFrame({'ds': sorted(g['Date'].unique())})
        fc = model.predict(future)[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'y_pred'})
        fc['y_pred'] = fc['y_pred'].clip(lower=0).round().astype(int)
        merged = g[['Date', 'Interval', 'TotalAgents']].merge(fc, on='Date', how='left')
        merged['y_pred'] = merged['y_pred'].fillna(0).astype(int)
        preds.append(merged.rename(columns={'TotalAgents': 'y_true'}))

    if not preds:
        return pd.DataFrame(), {}, pd.DataFrame()

    eval_df = pd.concat(preds, ignore_index=True).sort_values(['Date', 'Interval'])
    # Overall metrics
    mae = float(mean_absolute_error(eval_df['y_true'], eval_df['y_pred']))
    rmse = float(np.sqrt(mean_squared_error(eval_df['y_true'], eval_df['y_pred'])))
    mape = _safe_mape(eval_df['y_true'].values, eval_df['y_pred'].values)
    r2 = float(r2_score(eval_df['y_true'], eval_df['y_pred']))

    overall = {'MAE': round(mae, 3), 'RMSE': round(rmse, 3), 'MAPE_%': None if pd.isna(mape) else round(mape, 2), 'R2': round(r2, 3)}

    # Per-interval metrics
    per_interval_rows = []
    for interval, g in eval_df.groupby('Interval'):
        i_mae = float(mean_absolute_error(g['y_true'], g['y_pred']))
        i_rmse = float(np.sqrt(mean_squared_error(g['y_true'], g['y_pred'])))
        i_mape = _safe_mape(g['y_true'].values, g['y_pred'].values)
        i_r2 = float(r2_score(g['y_true'], g['y_pred'])) if g['y_true'].nunique() > 1 else np.nan
        per_interval_rows.append({
            'Interval': interval,
            'MAE': round(i_mae, 3),
            'RMSE': round(i_rmse, 3),
            'MAPE_%': None if pd.isna(i_mape) else round(i_mape, 2),
            'R2': None if pd.isna(i_r2) else round(i_r2, 3)
        })
    per_interval_df = pd.DataFrame(per_interval_rows).sort_values('Interval')

    return eval_df, overall, per_interval_df
