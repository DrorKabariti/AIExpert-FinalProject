import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import warnings
import joblib
from prophet import Prophet
import os

warnings.filterwarnings("ignore")

def run_full_eda(cc_file_buffer, holidays_file_buffer,
                 weekday_start: int, weekday_end: int,
                 friday_start: int, friday_end: int):
    cc_df = pd.read_csv(cc_file_buffer)
    holidays_df = pd.read_csv(holidays_file_buffer)
    cc_df['QueueStartDate'] = pd.to_datetime(cc_df['QueueStartDate'], dayfirst=True, errors='coerce')
    holidays_df['Date'] = pd.to_datetime(holidays_df['Date'], dayfirst=True, errors='coerce')
    cc_df = cc_df.rename(columns={'QueueStartDate': 'Date'})
    cc_df = cc_df.merge(holidays_df[['Date', 'IsHoliday', 'IsHolidayEve']], on='Date', how='left')
    cc_df['IsHoliday'] = cc_df['IsHoliday'].fillna(0).astype(int)
    cc_df['IsHolidayEve'] = cc_df['IsHolidayEve'].fillna(0).astype(int)
    cc_df['Weekday'] = cc_df['Date'].dt.weekday
    cc_df['WeekdayName'] = cc_df['Date'].dt.day_name()
    if 'HourInterval' in cc_df.columns:
        cc_df['Interval'] = cc_df['HourInterval'].str.split('-').str[0].str.strip()
        cc_df['Interval'] = cc_df['Interval'].str.replace(':00', '').astype(int)
    cc_df = cc_df[(
            ((cc_df['Weekday'] < 4) & (cc_df['Interval'].between(weekday_start, weekday_end))) |
            ((cc_df['Weekday'] == 4) & (cc_df['Interval'].between(friday_start, friday_end))) |
            (cc_df['IsHoliday'] == 1)
    )]
    return cc_df, holidays_df


def retrain_best_model(cc_file_buffer, holidays_file_buffer,
                       weekday_start: int, weekday_end: int,
                       friday_start: int, friday_end: int):
    cc_df, holidays_df_orig = run_full_eda(
        cc_file_buffer, holidays_file_buffer,
        weekday_start, weekday_end,
        friday_start, friday_end
    )
    df = cc_df[['Date', 'Weekday', 'Interval', 'IsHoliday', 'IsHolidayEve', 'TotalAgents']]
    holidays_df = holidays_df_orig.rename(columns={'Date': 'ds', 'HolidayName': 'holiday'})
    holidays_df['ds'] = pd.to_datetime(holidays_df['ds'], errors='coerce')
    holidays_df = holidays_df.dropna(subset=['ds', 'holiday'])
    holidays_df['lower_window'] = 0
    holidays_df['upper_window'] = 0
    prophet_models = {}
    unique_intervals = sorted(df['Interval'].unique())
    progress_bar = st.progress(0, text="מתחיל אימון...")
    for i, interval in enumerate(unique_intervals):
        progress_bar.progress((i + 1) / len(unique_intervals), text=f"מאמן מודל עבור שעה {interval}:00")
        df_int = df[df['Interval'] == interval][['Date', 'TotalAgents']].copy()
        if df_int.empty or df_int['TotalAgents'].nunique() < 2: continue
        df_int = df_int.rename(columns={'Date': 'ds', 'TotalAgents': 'y'})
        m = Prophet(
            daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True,
            seasonality_mode='multiplicative', changepoint_prior_scale=0.1,
            holidays_prior_scale=10, holidays=holidays_df
        )
        m.fit(df_int)
        prophet_models[interval] = m
    return prophet_models


def save_model(prophet_models, model_save_path):
    joblib.dump(prophet_models, model_save_path)


def load_model(model_file_buffer):
    return joblib.load(model_file_buffer)


def predict_future(prophet_models, future_dates: list, holidays_file_buffer,
                   weekday_start: int, weekday_end: int,
                   friday_start: int, friday_end: int):
    holidays_df = pd.read_csv(holidays_file_buffer)
    holidays_df['Date'] = pd.to_datetime(holidays_df['Date'])
    future_df = pd.DataFrame({'Date': pd.to_datetime(future_dates)})
    future_df['Weekday'] = future_df['Date'].dt.weekday
    future_df = future_df.merge(holidays_df[['Date', 'IsHoliday']], on='Date', how='left')
    future_df['IsHoliday'] = future_df['IsHoliday'].fillna(0).astype(int)
    all_rows = []
    for _, row in future_df.iterrows():
        weekday = row['Weekday']
        is_holiday = row['IsHoliday']
        if weekday == 5: continue
        if is_holiday == 1:
            hours_range = range(weekday_start, weekday_end + 1)
        elif weekday < 4 or weekday == 6:
            hours_range = range(weekday_start, weekday_end + 1)
        elif weekday == 4:
            hours_range = range(friday_start, friday_end + 1)
        else:
            continue
        for hour in hours_range:
            all_rows.append({'Date': row['Date'], 'Interval': hour})
    if not all_rows: return pd.DataFrame(columns=['Date', 'Interval', 'Predicted_TotalAgents'])
    pred_df = pd.DataFrame(all_rows)
    predictions = []
    for interval, group in pred_df.groupby("Interval"):
        if interval not in prophet_models:
            group['Predicted_TotalAgents'] = 0
            predictions.append(group)
            continue
        model = prophet_models[interval]
        future_dates_prophet = pd.DataFrame({"ds": group["Date"].unique()})
        forecast = model.predict(future_dates_prophet)
        forecast['yhat'] = forecast['yhat'].clip(lower=0).round().astype(int)
        forecast = forecast[["ds", "yhat"]].rename(columns={"ds": "Date", "yhat": "Predicted_TotalAgents"})
        forecast["Interval"] = interval
        merged = group.merge(forecast, on=["Date", "Interval"], how="left")
        merged["Predicted_TotalAgents"] = merged["Predicted_TotalAgents"].fillna(0)
        predictions.append(merged[['Date', 'Interval', 'Predicted_TotalAgents']])
    if not predictions: return pd.DataFrame(columns=['Date', 'Interval', 'Predicted_TotalAgents'])
    result = pd.concat(predictions, ignore_index=True)
    result['Interval'] = result['Interval'].apply(lambda x: f"{int(x):02d}:00")
    result['Date_str'] = result['Date'].dt.strftime('%d/%m/%Y')
    result = result.sort_values(by=['Date', 'Interval']).reset_index(drop=True)
    return result


# =============================================================================
# Streamlit App UI
# =============================================================================
st.set_page_config(layout="wide", page_title="מערכת חיזוי נציגים")

st.markdown("""
    <style>
        body, .main { direction: rtl !important; }
        [data-testid="stToolbar"], [data-testid="stFullScreenFrame"] { direction: ltr !important; }
        .st-emotion-cache-16txtl3 { text-align: right !important; }
        [data-testid="stSidebar"] button { left: 1rem !important; right: auto !important; }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("ניווט")
page = st.sidebar.radio("בחר דף:", ('ℹ️ אודות', '📊 חיזוי כמות נציגים', '⚙️ אימון מודל מחדש'))

if page == 'ℹ️ אודות':
    st.title("אודות מערכת החיזוי")
    st.markdown("...")  # תוכן דף אודות

elif page == '📊 חיזוי כמות נציגים':
    st.title('חיזוי כמות נציגים')

    col1, col2 = st.columns(2)
    with col1:
        holidays_file_pred = st.file_uploader("1. טען קובץ חגים (CSV)", type="csv", key="pred_holidays")
    with col2:
        model_file = st.file_uploader("2. טען קובץ מודל (PKL)", type="pkl", key="pred_model")

    st.markdown("---")

    col3, col4 = st.columns(2)
    with col3:
        start_date = st.date_input("3. בחר תאריך התחלה", format="DD/MM/YYYY")
    with col4:
        end_date = st.date_input("4. בחר תאריך סיום", format="DD/MM/YYYY")

    if st.button("🚀 בצע חיזוי"):
        if holidays_file_pred and model_file and start_date and end_date:
            if start_date > end_date:
                st.error("תאריך ההתחלה חייב להיות לפני תאריך הסיום.")
            else:
                with st.spinner("מבצע חיזוי, נא להמתין..."):
                    prophet_models = load_model(model_file)
                    date_list = pd.date_range(start_date, end_date).tolist()
                    holidays_file_pred.seek(0)
                    predictions = predict_future(
                        prophet_models=prophet_models,
                        future_dates=date_list,
                        holidays_file_buffer=holidays_file_pred,
                        weekday_start=7, weekday_end=19,
                        friday_start=7, friday_end=13
                    )

                if not predictions.empty:
                    st.success("החיזוי הושלם בהצלחה!")
                    st.subheader("טבלת חיזויים (מבט שעתי מול יומי)")
                    pivot_df = predictions.pivot_table(
                        index='Interval', columns='Date_str', values='Predicted_TotalAgents'
                    ).fillna(0).astype(int)
                    sorted_columns = sorted(pivot_df.columns, key=lambda d: datetime.strptime(d, '%d/%m/%Y'))
                    pivot_df = pivot_df[sorted_columns]
                    st.dataframe(pivot_df, use_container_width=True)

                    st.subheader("גרף חיזויים")
                    fig = px.bar(
                        predictions, x='Interval', y='Predicted_TotalAgents', color='Date_str',
                        barmode='group', title='כמות נציגים חזויה לפי שעה ותאריך',
                        labels={'Interval': 'שעה', 'Predicted_TotalAgents': 'כמות נציגים חזויה', 'Date_str': 'תאריך'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("לא נוצרו חיזויים עבור טווח התאריכים שנבחר (אולי מדובר בסוף שבוע בלבד?).")
        else:
            st.warning("יש לטעון את כל הקבצים ולבחור טווח תאריכים.")

elif page == '⚙️ אימון מודל מחדש':
    st.title('אימון מודל מחדש')
    col1, col2 = st.columns(2)
    with col1:
        holidays_file_train = st.file_uploader("1. טען קובץ חגים (CSV)", type="csv", key="train_holidays")
    with col2:
        data_file = st.file_uploader("2. טען קובץ נתונים (CSV)", type="csv", key="train_data")
    if st.button("💪 בצע אימון"):
        if holidays_file_train and data_file:
            holidays_file_train.seek(0)
            data_file.seek(0)
            trained_models = retrain_best_model(
                cc_file_buffer=data_file,
                holidays_file_buffer=holidays_file_train,
                weekday_start=7, weekday_end=19,
                friday_start=7, friday_end=13
            )
            model_save_path = "prophet_model_retrained.pkl"
            save_model(trained_models, model_save_path)
            st.success(f"המודל אומן בהצלחה!")
            st.info(f"המודל נשמר בנתיב: **{model_save_path}** בתיקייה שבה האפליקציה רצה.")
            with open(model_save_path, "rb") as file:
                st.download_button(
                    label="📥 הורד את קובץ המודל המאומן",
                    data=file, file_name=model_save_path, mime="application/octet-stream"
                )
        else:
            st.error("יש לטעון קובץ חגים וקובץ נתונים כדי להתחיל באימון.")