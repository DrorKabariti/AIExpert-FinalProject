
# app_streamlit.py
# -*- coding: utf-8 -*-
"""
ממשק Streamlit ל-TotalAgentForecast
מייבא פונקציות מהקובץ total_agent_core.py בלבד.
"""
import os
import json
import pandas as pd
import streamlit as st
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

from TotalAgentsForeCastBackend import (
    load_model,predict_future,retrain_best_model,save_model,evaluate_models,
    WEEKDAY_START_DEFAULT,WEEKDAY_END_DEFAULT,FRIDAY_START_DEFAULT,FRIDAY_END_DEFAULT
)

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
page = st.sidebar.radio("בחר דף:", (
    'ℹ️ אודות',
    '📊 חיזוי כמות נציגים',
    '⚙️ אימון מודל מחדש',
    '📘 פירוט פונקציות',
    '🧪 גרפים ומדדים'
))

if page == 'ℹ️ אודות':
    st.title("אודות מערכת חיזוי כמות נציגים")
    st.markdown("מערכת חיזוי כמות נציגים בחתך שעתי:")
    st.markdown("- **ימי חול (א׳-ה׳, א׳=Sunday)**: 07:00–19:00")
    st.markdown("- **שישי וערב חג**: 07:00–13:00")
    st.markdown("- **שבת חג**: תמיד 0")

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
                        weekday_start=WEEKDAY_START_DEFAULT, weekday_end=WEEKDAY_END_DEFAULT,
                        friday_start=FRIDAY_START_DEFAULT, friday_end=FRIDAY_END_DEFAULT
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
                    st.warning("לא נוצרו חיזויים עבור טווח התאריכים שנבחר.")
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
                weekday_start=WEEKDAY_START_DEFAULT, weekday_end=WEEKDAY_END_DEFAULT,
                friday_start=FRIDAY_START_DEFAULT, friday_end=FRIDAY_END_DEFAULT
            )
            model_save_path = "prophet_model_retrained.pkl"
            save_model(trained_models, model_save_path)
            st.success("המודל אומן בהצלחה!")
            st.info(f"המודל נשמר בנתיב: **{model_save_path}** בתיקייה שבה האפליקציה רצה.")
            with open(model_save_path, "rb") as file:
                st.download_button(
                    label="📥 הורד את קובץ המודל המאומן",
                    data=file, file_name=model_save_path, mime="application/octet-stream"
                )
        else:
            st.error("יש לטעון קובץ חגים וקובץ נתונים כדי להתחיל באימון.")

elif page == '📘 פירוט פונקציות':
    st.title('📘 פירוט פונקציות ומסמכי מערכת')
    # --- כללי פעילות ---
    st.header("כללי פעילות (Business Rules)")
    st.markdown("""
       - **ימי חול (א׳–ה׳)**: *07:00–19:00*  
       - **שישי וערב חג**: *07:00–13:00*  
       - **שבת וחג**: *תמיד 0 נציגים*
       """)
    st.divider()

    # --- מקרא קצר ---
    st.caption("טיפ: לחצו על כל כותרת כדי להרחיב/לצמצם את פרטי הפונקציה.")
    st.subheader("רשימת פונקציות")
    st.write("")

    # 1) _parse_dates
    with st.expander("🧭 _parse_dates(df, date_col, dayfirst=True)"):
        st.markdown("""
           **מטרה**: המרת עמודת תאריך ל־`datetime` (כולל תמיכה ב־`dd/mm/yyyy`) עם `errors='coerce'`.  
           **פרמטרים**:
           - `df: pd.DataFrame` — נתונים גולמיים  
           - `date_col: str` — שם עמודת התאריך  
           - `dayfirst: bool` — האם היום קודם לחודש (ברירת מחדל: True)  

           **מחזירה**: `pd.DataFrame` עם העמודה ממורכזת לפורמט תאריך.  
           """)
    st.markdown("---")

    # 2) _coalesce_flags
    with st.expander("🚩 _coalesce_flags(df)"):
        st.markdown("""
           **מטרה**: הבטחת קיום העמודות `IsHoliday` ו־`IsHolidayEve` כ־`int` (0/1), ומילוי `NaN` ל־0.  
           **פרמטרים**:
           - `df: pd.DataFrame`  

           **מחזירה**: `pd.DataFrame` מעודכן.  
           """)
    st.markdown("---")

    # 3) run_full_eda
    with st.expander(
            "🔎 run_full_eda(cc_file_buffer, holidays_file_buffer, weekday_start=7, weekday_end=19, friday_start=7, friday_end=13)"):
        st.markdown("""
           **מטרה**: טעינת CSV, מיזוג חגים, יצירת שדות `Weekday`/`Interval`, וסינון חלונות פעילות לפי כללים — כהכנה לאימון/הערכה.  
           **פרמטרים**:
           - `cc_file_buffer` — קובץ הנתונים (CSV)  
           - `holidays_file_buffer` — קובץ החגים (CSV)  
           - `weekday_start, weekday_end, friday_start, friday_end` — חלונות שעות עפ״י הכללים  

           **מחזירה**:  
           - `cc_df: DataFrame` — נתונים לאחר סינון וסטנדרטיזציה  
           - `holidays_df: DataFrame` — טבלת חגים המקורית (ל־Prophet)  

           **הערות**:
           - מוציאים שבת וחגים מאימון.  
           - ערב חג מטופל כמו יום שישי.  
           """)
    st.markdown("---")

    # 4) retrain_best_model
    with st.expander(
            "⚙️ retrain_best_model(cc_file_buffer, holidays_file_buffer, weekday_start=7, weekday_end=19, friday_start=7, friday_end=13)"):
        st.markdown("""
           **מטרה**: אימון מודל **Prophet פר־אינטרוול (שעה)** על בסיס הדאטה המסונן; החזרת מילון מודלים.  
           **פרמטרים**: זהים ל־`run_full_eda`.  

           **מחזירה**: `dict` במבנה `{interval (int) -> Prophet model}`.  

           **הערות**:
           - נעשה שימוש ב־holidays (אם קיימים שמות חגים), וקליפ לערכים שליליים בשלב החיזוי.  
           - מומלץ לשמור את המודל באמצעות `save_model`.  
           """)
    st.markdown("---")

    # 5) save_model
    with st.expander("💾 save_model(prophet_models, model_save_path)"):
        st.markdown("""
           **מטרה**: שמירת מילון המודלים לקובץ `pkl` באמצעות `joblib`.  
           **פרמטרים**:
           - `prophet_models: dict` — מילון `{interval -> model}`  
           - `model_save_path: str` — שם/נתיב קובץ לשמירה  

           **מחזירה**: `None`.  
           """)
    st.markdown("---")

    # 6) load_model
    with st.expander("📂 load_model(model_file_buffer)"):
        st.markdown("""
           **מטרה**: טעינת מילון המודלים מקובץ `pkl` באמצעות `joblib`.  
           **פרמטרים**:
           - `model_file_buffer` — buffer/נתיב לקובץ המודל  
           **מחזירה**: `dict` של מודלי Prophet לפי אינטרוול.  
           """)
    st.markdown("---")

    # 7) predict_future
    with st.expander(
            "🔮 predict_future(prophet_models, future_dates, holidays_file_buffer, weekday_start=7, weekday_end=19, friday_start=7, friday_end=13)"):
        st.markdown("""
           **מטרה**: הפקת תחזית לכל תאריך בטווח המבוקש תוך **אכיפה קשיחה של כללי הפעילות**:
           - א׳–ה׳: 07:00 - 19:00  
           - שישי/ערב חג: 07-00  - 13:00  
           - שבת/חג: תמיד 0  

           **פרמטרים**:
           - `prophet_models: dict` — מודלים לפי אינטרוול  
           - `future_dates: list[date|Timestamp|str]` — רשימת תאריכים לחיזוי  
           - `holidays_file_buffer: CSV` — טבלת חגים עם `IsHoliday`/`IsHolidayEve`  
           - חלונות שעות (`weekday_start/end`, `friday_start/end`)  

           **מחזירה**: `DataFrame` בעמודות:
           - `Date`, `Interval` (״HH:00״), `Predicted_TotalAgents`, `Date_str`  

           **הערות**:
           - עבור שבת/חג מחזירים שורות עם 0 לשמירת מבנה הטבלה.  
           """)
    st.markdown("---")

    # 8) _safe_mape
    with st.expander("🧮 _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float"):
        st.markdown("""
           **מטרה**: חישוב MAPE אמין (התעלמות מערכי אמת=0 כדי למנוע חלוקה באפס).  
           **פרמטרים**:
           - `y_true: ndarray`, `y_pred: ndarray`  

           **מחזירה**: `float` (או `NaN` אם אין ערכי אמת חיוביים).  
           """)
    st.markdown("---")

    # 9) evaluate_models
    with st.expander("📏 evaluate_models(prophet_models, cc_file_buffer, holidays_file_buffer, val_days=14, ...)"):
        st.markdown("""
           **מטרה**: הערכת ביצועים על **N הימים האחרונים** לאחר סינון לפי כללים; חישוב מדדים ויצירת טבלת אימות.  
           **פרמטרים**:
           - `prophet_models` — מילון מודלים  
           - `cc_file_buffer` / `holidays_file_buffer` — קובצי CSV  
           - `val_days` — כמות ימים להערכה (ברירת מחדל: 14)  
           - חלונות שעות (`weekday_start/end`, `friday_start/end`)  

           **מחזירה**:
           - `eval_df: DataFrame` — רשומות אמת/חזוי לכל אינטרוול  
           - `overall: dict` — MAE, RMSE, MAPE%, R²  
           - `per_interval_df: DataFrame` — מדדים לפי אינטרוול  

           **הערות**:
           - חיזוי מוקשח: חיתוך שלילי ל־0 ועיגול למספרים שלמים.  
           - אם אין מודל לאינטרוול מסוים — חוזרים 0 עבורו.  
           """)
    st.divider()

    # הערות כלליות
    st.subheader("הערות כלליות")
    st.markdown("""
       - **אכיפה קשיחה** של חלונות פעילות מיושמת גם באימון וגם בחיזוי.  
       - מומלץ לשמור מודל לאחר אימון ולגרסה אותו בין ריצות.  
       """)
    st.subheader("הערות יישומיות")
    st.markdown("""
    - **0 בשבת וחג**: שורות מוחזרות עם `Predicted_TotalAgents=0` כדי לשמר מבנה טבלאי מלא.
    - **ערב חג**: מטופל כמו יום שישי (07:00-13:00) הן באימון והן בחיזוי.
    - **מודל פר אינטרוול**: מודל נפרד לכל שעה כדי ללכוד דפוסי עומס שונים לאורך היום.
    """)


elif page == '🧪 גרפים ומדדים':
    st.title('🧪 גרפים ומדדים – הערכת ביצועי המודל')
    col1, col2, col3 = st.columns(3)
    with col1:
        holidays_file_eval = st.file_uploader("1. קובץ חגים (CSV)", type="csv", key="eval_holidays")
    with col2:
        data_file_eval = st.file_uploader("2. קובץ נתונים (CSV)", type="csv", key="eval_data")
    with col3:
        model_file_eval = st.file_uploader("3. קובץ מודל (PKL)", type="pkl", key="eval_model")

    val_days = int(st.number_input(
        "חלון אימות (ימים אחרונים)",
        min_value=7,
        max_value=365,
        value=14,
        step=1
    ))

    if st.button("📏 חשב מדדים והצג גרפים"):
        if holidays_file_eval and data_file_eval and model_file_eval:
            with st.spinner("מחשב תחזיות ומדדים..."):
                model_file_eval.seek(0)
                prophet_models_eval = load_model(model_file_eval)

                holidays_file_eval.seek(0)
                data_file_eval.seek(0)
                eval_df, overall, per_interval_df = evaluate_models(
                    prophet_models_eval, data_file_eval, holidays_file_eval,
                    val_days=val_days,
                    weekday_start=WEEKDAY_START_DEFAULT, weekday_end=WEEKDAY_END_DEFAULT,
                    friday_start=FRIDAY_START_DEFAULT, friday_end=FRIDAY_END_DEFAULT
                )

            if eval_df.empty:
                st.warning("לא נמצאו נתונים לחלון האימות שנבחר או שהמודל חסר לאינטרוולים הדרושים.")
            else:
                st.success("החישוב הושלם!")

                # --- KPIs ---
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("MAE", overall.get('MAE'))
                k2.metric("RMSE", overall.get('RMSE'))
                k3.metric("MAPE %", overall.get('MAPE_%') if overall.get('MAPE_%') is not None else "N/A")
                k4.metric("R²", overall.get('R2'))

                # --- Per-interval metrics table ---
                st.subheader("מדדים לפי אינטרוול")
                st.dataframe(per_interval_df, use_container_width=True)

                # --- Charts ---
                st.subheader("Actual vs. Predicted – סיכום יומי")
                daily = eval_df.groupby('Date', as_index=False).agg(y_true=('y_true', 'sum'),
                                                                    y_pred=('y_pred', 'sum'))
                daily['Date_str'] = daily['Date'].dt.strftime('%d/%m/%Y')
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=daily['Date_str'], y=daily['y_true'], mode='lines+markers', name='Actual'))
                fig1.add_trace(
                    go.Scatter(x=daily['Date_str'], y=daily['y_pred'], mode='lines+markers', name='Predicted'))
                fig1.update_layout(title="כמות נציגים – אמת מול חזוי (סה\"כ יומי)", xaxis_title="תאריך",
                                   yaxis_title="סה\"כ נציגים")
                st.plotly_chart(fig1, use_container_width=True)




        else:
            st.warning("יש לטעון קובץ חגים, נתונים, ומודל כדי לחשב מדדים.")