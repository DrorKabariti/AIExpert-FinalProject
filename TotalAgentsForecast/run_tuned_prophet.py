#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Tuned Prophet Per-Interval for Contact Center "TotalAgents" Forecasting
- Business rules enforced:
  * Weekdays (Mon-Thu and Sun): 07:00–19:00
  * Friday and Holiday Eve:     07:00–13:00
  * Saturday and Holidays:      excluded from training; predictions should be 0 (handled in app layer)
- Tuning per-interval via a small grid with inner validation window
- Outputs:
  * model.pkl .................. joblib dict {interval -> Prophet model}
  * metrics_per_interval.csv ... per-interval MAE/RMSE/MAPE/R2 on the holdout
  * overall_metrics.json ....... summarized metrics
  * daily_actual_vs_pred.html .. Plotly line chart
  * scatter_actual_vs_pred.html  Plotly scatter chart
  * rmse_by_interval.html ...... Plotly bar chart
Usage:
  python run_tuned_prophet.py \
    --data_csv CC_2020-2025_New.csv \
    --holidays_csv Holidays_New.csv \
    --val_days 28 \
    --out_dir ./output
"""
import argparse
import json
import os
import warnings
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

try:
    from prophet import Prophet
except Exception as e:
    raise SystemExit(
        "Prophet is not installed. Please run:\n"
        "    pip install prophet\n"
        f"Underlying error: {e}"
    )

# ------------------------------
# Defaults
# ------------------------------
WEEKDAY_START_DEFAULT = 7
WEEKDAY_END_DEFAULT   = 19
FRIDAY_START_DEFAULT  = 7
FRIDAY_END_DEFAULT    = 13

# ------------------------------
# Utils
# ------------------------------
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

def _winsorize(series, lower=0.01, upper=0.99):
    lo, hi = series.quantile(lower), series.quantile(upper)
    return series.clip(lo, hi)

def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

# ------------------------------
# Data Prep with Business Rules
# ------------------------------
def load_and_prepare(data_csv: str, holidays_csv: str,
                     weekday_start: int = WEEKDAY_START_DEFAULT, weekday_end: int = WEEKDAY_END_DEFAULT,
                     friday_start: int = FRIDAY_START_DEFAULT, friday_end: int = FRIDAY_END_DEFAULT):
    cc_df = pd.read_csv(data_csv)
    holidays_df = pd.read_csv(holidays_csv)

    # Parse dates
    cc_df = _parse_dates(cc_df, 'QueueStartDate', dayfirst=True).rename(columns={'QueueStartDate': 'Date'})
    holidays_df = _parse_dates(holidays_df, 'Date', dayfirst=True)

    # Merge holiday flags
    cc_df = cc_df.merge(holidays_df[['Date', 'IsHoliday', 'IsHolidayEve']], on='Date', how='left')
    cc_df = _coalesce_flags(cc_df)

    # Weekday & Interval
    cc_df['Weekday'] = cc_df['Date'].dt.weekday  # Mon=0 ... Sun=6
    if 'HourInterval' in cc_df.columns:
        # Expect formats like "07:00 - 08:00" -> take the first HH:MM and cast to hour int
        start_part = cc_df['HourInterval'].astype(str).str.split('-').str[0].str.strip()
        cc_df['Interval'] = start_part.str.split(':').str[0].astype(int)
    elif 'Interval' in cc_df.columns:
        cc_df['Interval'] = cc_df['Interval'].astype(int)
    else:
        raise ValueError("Input CSV must contain either 'HourInterval' or 'Interval' column.")

    # -------- Business rules for TRAINING data --------
    # Holidays excluded completely; Saturday (5) excluded.
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
        (cc_df['IsHolidayEve'] == 1) &
        (cc_df['Interval'].between(friday_start, friday_end))
    )

    cc_df = cc_df[mask_weekdays | mask_friday | mask_eve].copy()

    return cc_df, holidays_df

def build_prophet_holidays(holidays_df: pd.DataFrame) -> pd.DataFrame:
    # Prefer HolidayName if available; otherwise create synthetic name for IsHoliday rows
    h = holidays_df.copy()
    if 'HolidayName' not in h.columns:
        h['HolidayName'] = np.where(h.get('IsHoliday', 0) == 1, 'Holiday', None)
    h = h.rename(columns={'Date': 'ds', 'HolidayName': 'holiday'})
    h['ds'] = pd.to_datetime(h['ds'], errors='coerce')
    h = h.dropna(subset=['ds', 'holiday'])
    if h.empty:
        return h
    h['lower_window'] = 0
    h['upper_window'] = 0
    return h

# ------------------------------
# Modeling
# ------------------------------
def _make_prophet(years_of_data: float,
                  seasonality_mode: str,
                  cps: float,
                  sps: float,
                  hps: float,
                  holidays_df: pd.DataFrame):
    use_yearly = years_of_data >= 2.0
    m = Prophet(
        growth='linear',
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=use_yearly,
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=cps,
        seasonality_prior_scale=sps,
        holidays_prior_scale=hps,
        holidays=holidays_df,
        changepoint_range=0.9
    )
    # Stronger weekly component
    m.add_seasonality(name='weekly_high', period=7, fourier_order=10, prior_scale=sps)
    return m

def train_tuned_models(cc_df: pd.DataFrame, holidays_df: pd.DataFrame, inner_val_days: int = 28):
    holidays_prophet = build_prophet_holidays(holidays_df)

    models = {}
    tuning_records = []

    seasonality_modes = ['multiplicative', 'additive']
    cps_grid = [0.05, 0.1, 0.3]
    sps_grid = [1.0, 5.0, 10.0]
    hps_grid = [1.0, 5.0]

    for interval in sorted(cc_df['Interval'].unique()):
        dfi = cc_df[cc_df['Interval'] == interval][['Date', 'TotalAgents']].copy()
        if dfi.empty or dfi['TotalAgents'].nunique() < 2:
            continue

        # Winsorize to reduce outlier impact
        dfi['TotalAgents'] = _winsorize(dfi['TotalAgents'], 0.01, 0.99)
        dfi = dfi.rename(columns={'Date': 'ds', 'TotalAgents': 'y'}).sort_values('ds')

        span_years = (dfi['ds'].max() - dfi['ds'].min()).days / 365.25
        cutoff = dfi['ds'].max() - pd.Timedelta(days=inner_val_days)
        traini = dfi[dfi['ds'] <= cutoff]
        vali  = dfi[dfi['ds'] >  cutoff]

        # If not enough for validation, train a reasonable default on full data
        if len(traini) < 30 or len(vali) < max(7, inner_val_days // 4):
            m = _make_prophet(span_years, 'multiplicative', 0.1, 5.0, 5.0, holidays_prophet)
            m.fit(dfi)
            models[interval] = m
            tuning_records.append({'Interval': interval, 'seasonality_mode': 'multiplicative',
                                   'cps': 0.1, 'sps': 5.0, 'hps': 5.0, 'RMSE_val': None, 'UsedDefault': True})
            continue

        best_rmse, best_cfg, best_model = float('inf'), None, None
        for mode in seasonality_modes:
            for cps in cps_grid:
                for sps in sps_grid:
                    for hps in hps_grid:
                        try:
                            m = _make_prophet(span_years, mode, cps, sps, hps, holidays_prophet)
                            m.fit(traini)
                            fc = m.predict(vali[['ds']])[['ds', 'yhat']]
                            y_true = vali.set_index('ds')['y'].reindex(fc['ds']).values
                            y_pred = np.clip(fc['yhat'].values, a_min=0, a_max=None)
                            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
                            if rmse < best_rmse:
                                best_rmse = rmse
                                best_cfg = (mode, cps, sps, hps)
                                best_model = m
                        except Exception:
                            continue

        # Refit on full data with best config (or fallback)
        if best_model is None:
            final_m = _make_prophet(span_years, 'multiplicative', 0.1, 5.0, 5.0, holidays_prophet)
        else:
            mode, cps, sps, hps = best_cfg
            final_m = _make_prophet(span_years, mode, cps, sps, hps, holidays_prophet)

        final_m.fit(dfi)
        models[interval] = final_m
        tuning_records.append({
            'Interval': interval,
            'seasonality_mode': best_cfg[0] if best_cfg else 'multiplicative',
            'cps': best_cfg[1] if best_cfg else 0.1,
            'sps': best_cfg[2] if best_cfg else 5.0,
            'hps': best_cfg[3] if best_cfg else 5.0,
            'RMSE_val': None if not np.isfinite(best_rmse) else round(best_rmse, 4),
            'UsedDefault': best_model is None
        })

    tuning_df = pd.DataFrame(tuning_records).sort_values('Interval')
    return models, tuning_df

# ------------------------------
# Evaluation
# ------------------------------
def evaluate_models(models: dict, cc_df: pd.DataFrame, val_days: int = 14, out_dir: str = "."):
    # Choose validation window: last val_days with data
    unique_dates = sorted(cc_df['Date'].dropna().unique())
    if len(unique_dates) == 0:
        raise SystemExit("No dates found for evaluation.")
    val_days = min(val_days, len(unique_dates))
    val_dates = unique_dates[-val_days:]
    df_val = cc_df[cc_df['Date'].isin(val_dates)].copy()

    preds = []
    for interval, g in df_val.groupby('Interval'):
        model = models.get(int(interval))
        if model is None:
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

    eval_df = pd.concat(preds, ignore_index=True).sort_values(['Date', 'Interval'])

    # Overall metrics
    mae = float(mean_absolute_error(eval_df['y_true'], eval_df['y_pred']))
    rmse = float(np.sqrt(mean_squared_error(eval_df['y_true'], eval_df['y_pred'])))
    mape = _safe_mape(eval_df['y_true'].values, eval_df['y_pred'].values)
    r2 = float(r2_score(eval_df['y_true'], eval_df['y_pred']))

    overall = {'MAE': round(mae, 3), 'RMSE': round(rmse, 3),
               'MAPE_%': None if pd.isna(mape) else round(mape, 2), 'R2': round(r2, 3)}

    # Per-interval metrics
    per_interval_rows = []
    for interval, g in eval_df.groupby('Interval'):
        i_mae = float(mean_absolute_error(g['y_true'], g['y_pred']))
        i_rmse = float(np.sqrt(mean_squared_error(g['y_true'], g['y_pred'])))
        i_mape = _safe_mape(g['y_true'].values, g['y_pred'].values)
        i_r2 = float(r2_score(g['y_true'], g['y_pred'])) if g['y_true'].nunique() > 1 else float("nan")
        per_interval_rows.append({
            'Interval': interval,
            'MAE': round(i_mae, 3),
            'RMSE': round(i_rmse, 3),
            'MAPE_%': None if pd.isna(i_mape) else round(i_mape, 2),
            'R2': None if pd.isna(i_r2) else round(i_r2, 3)
        })
    per_interval_df = pd.DataFrame(per_interval_rows).sort_values('Interval')

    # Save metrics
    os.makedirs(out_dir, exist_ok=True)
    per_interval_path = os.path.join(out_dir, "metrics_per_interval.csv")
    overall_path = os.path.join(out_dir, "overall_metrics.json")
    eval_path = os.path.join(out_dir, "eval_records.csv")

    per_interval_df.to_csv(per_interval_path, index=False, encoding='utf-8-sig')
    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)
    eval_df.to_csv(eval_path, index=False, encoding='utf-8-sig')

    # Plots
    daily = eval_df.groupby('Date', as_index=False).agg(y_true=('y_true', 'sum'),
                                                        y_pred=('y_pred', 'sum'))
    daily['Date_str'] = pd.to_datetime(daily['Date']).dt.strftime('%d/%m/%Y')
    fig1 = px.line(daily, x='Date_str', y=['y_true', 'y_pred'],
                   title='Actual vs Predicted – Daily Sum',
                   labels={'value': 'Agents', 'Date_str': 'Date', 'variable': 'Series'})
    fig1.write_html(os.path.join(out_dir, "daily_actual_vs_pred.html"), include_plotlyjs='cdn')

    fig2 = px.scatter(eval_df, x='y_true', y='y_pred', trendline='ols',
                      title='Scatter: Actual vs Predicted (All Intervals)',
                      labels={'y_true': 'Actual', 'y_pred': 'Predicted'})
    fig2.write_html(os.path.join(out_dir, "scatter_actual_vs_pred.html"), include_plotlyjs='cdn')

    fig3 = px.bar(per_interval_df, x='Interval', y='RMSE', title='RMSE by Interval',
                  labels={'Interval': 'Hour', 'RMSE': 'RMSE'})
    fig3.write_html(os.path.join(out_dir, "rmse_by_interval.html"), include_plotlyjs='cdn')

    return overall, per_interval_df, eval_df

# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", required=True, help="Path to CC_2020-2025_New.csv (or similar)")
    ap.add_argument("--holidays_csv", required=True, help="Path to Holidays_New.csv")
    ap.add_argument("--val_days", type=int, default=28, help="Validation window (days) for evaluation")
    ap.add_argument("--inner_val_days", type=int, default=28, help="Inner validation window for tuning per-interval")
    ap.add_argument("--out_dir", default="./output", help="Output directory")
    ap.add_argument("--model_out", default="model.pkl", help="Output path for joblib model")
    args = ap.parse_args()

    print("✅ Loading and preparing data...")
    cc_df, holidays_df = load_and_prepare(args.data_csv, args.holidays_csv)

    print("✅ Training tuned models per-interval...")
    models, tuning_df = train_tuned_models(cc_df, holidays_df, inner_val_days=args.inner_val_days)

    os.makedirs(args.out_dir, exist_ok=True)
    tuning_path = os.path.join(args.out_dir, "tuning_summary.csv")
    tuning_df.to_csv(tuning_path, index=False, encoding='utf-8-sig')
    print(f"💾 Saved tuning summary: {tuning_path}")

    model_path = os.path.join(args.out_dir, args.model_out)
    joblib.dump(models, model_path)
    print(f"💾 Saved model: {model_path}")

    print("✅ Evaluating models on recent holdout...")
    overall, per_interval_df, eval_df = evaluate_models(models, cc_df, val_days=args.val_days, out_dir=args.out_dir)

    print("\n=== Overall Metrics ===")
    print(json.dumps(overall, ensure_ascii=False, indent=2))

    print("\n=== Top 10 Worst Intervals by RMSE ===")
    print(per_interval_df.sort_values('RMSE', ascending=False).head(10).to_string(index=False))

    print("\nOutputs saved under:", os.path.abspath(args.out_dir))

if __name__ == "__main__":
    main()
