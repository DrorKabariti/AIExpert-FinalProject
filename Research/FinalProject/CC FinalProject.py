# 📊 EDA לתחזית נציגים + חיזוי TotalAgents

# טעינת ספריות בסיס
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import warnings
import joblib
warnings.filterwarnings("ignore")

# למידת מכונה ונוירונים
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from itertools import product
from IPython.display import clear_output
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.model_selection import TimeSeriesSplit

# קריאת קבצי CSV
cc_df = pd.read_csv("data/CC_2020-2025_New.csv")
holidays_df = pd.read_csv("data/Holidays.csv")

# עיבוד נתונים ראשוני
cc_df.columns = [col.strip().replace(" ", "_").replace("-", "_") for col in cc_df.columns]
holidays_df.columns = [col.strip().replace(" ", "_").replace("-", "_") for col in holidays_df.columns]
cc_df['QueueStartDate'] = pd.to_datetime(cc_df['QueueStartDate'], dayfirst=True, errors='coerce')
holidays_df['CalendarDate'] = pd.to_datetime(holidays_df['CalendarDate'], dayfirst=True, errors='coerce')
cc_df['Weekday'] = cc_df['QueueStartDate'].dt.day_name()
cc_df['IsWeekend'] = cc_df['Weekday'].isin(['Friday', 'Saturday'])
cc_df['AnsweredRatio'] = (cc_df['TotalCallsAnswered'] / cc_df['TotalAgents'])
cc_df['AnsweredRatio'] = cc_df['AnsweredRatio'].replace([np.inf, -np.inf], np.nan).fillna(0).round().astype(int)
cc_df = cc_df.merge(holidays_df[['CalendarDate', 'HolidayNameHebrew']], left_on='QueueStartDate', right_on='CalendarDate', how='left')
cc_df['IsHoliday'] = cc_df['HolidayNameHebrew'].notna()
cc_df.drop(columns=['CalendarDate', 'HolidayNameHebrew'], inplace=True)
cc_df.drop_duplicates(inplace=True)

# טיפול בערכים חסרים
print("\nMissing values per column:")
print(cc_df.isna().sum())

# הסרת כפילויות
original_len = len(cc_df)
cc_df.drop_duplicates(subset=['QueueStartDate', 'HalfHourInterval'], inplace=True)
print(f"\n🧹 נמחקו {original_len - len(cc_df)} כפילויות לפי QueueStartDate ו-HalfHourInterval")

# המרת אינטרוול לפורמט מספרי: '10:00 - 10:30' -> 10.5
def parse_interval(interval_str):
    start_time = interval_str.split(" - ")[0]
    hour, minute = map(int, start_time.split(":"))
    return hour + (0.5 if minute == 30 else 0.0)

cc_df['Interval'] = cc_df['HalfHourInterval'].apply(parse_interval)

# יצירת טבלת המרה ל AnseredRatio 

answer_ratio_lookup = cc_df.groupby(['Weekday', 'Interval', 'IsHoliday'])['AnsweredRatio'].mean().reset_index()
answer_ratio_lookup.to_csv("data/answer_ratio_lookup.csv", index=False)
# answer_ratio_lookup[
#         (answer_ratio_lookup['Weekday'] == 'Monday') &
#         (answer_ratio_lookup['Interval'] == 10.5) &
#         (answer_ratio_lookup['IsHoliday'] == False)
#     ]

# בחירת משתנים לחיזוי
features_full = ['Weekday', 'Interval', 'IsHoliday', 'AnsweredRatio']
target = 'TotalAgents'
X = cc_df[features_full]
y = cc_df[target]

# הגדרת Cross-Validation עם סדר כרונולוגי
tscv = TimeSeriesSplit(n_splits=5)

# קידוד וטרנספורמציה
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown='ignore'), ['Weekday']),
    ("num", 'passthrough', ['Interval', 'IsHoliday', 'AnsweredRatio'])
])

# חלוקה ל-Train/Validation/Test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2


# רשימת מודלים לבדיקה

models = {
    'Linear Regression': Pipeline([
        ("prep", preprocessor),
        ("model", LinearRegression())
    ]),
    'Random Forest': Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestRegressor(random_state=42))
    ]),
    'XGBoost': Pipeline([
        ("prep", preprocessor),
        ("model", XGBRegressor(random_state=42))
    ]),
    'Gradient Boosting': Pipeline([
        ("prep", preprocessor),
        ("model", GradientBoostingRegressor(random_state=42))
    ])
}


# אימון וביצועים
results = []
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    results.append({
        'Model': name,
        'MAE': mean_absolute_error(y_test, preds),
        'RMSE': np.sqrt(mean_squared_error(y_test, preds)),
        'R2': r2_score(y_test, preds)
    })

results_df = pd.DataFrame(results).sort_values(by='RMSE')
print("🔍 תוצאות המודלים הקלאסיים:\n")
print(results_df.to_string(index=False))

# מודל Neural Network (Keras)
X_transformed = preprocessor.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_transformed)
X_train_nn, X_temp_nn, y_train_nn, y_temp_nn = train_test_split(X_scaled, y, test_size=0.4, random_state=42)
X_val_nn, X_test_nn, y_val_nn, y_test_nn = train_test_split(X_temp_nn, y_temp_nn, test_size=0.5, random_state=42)

model_nn = Sequential([
    Dense(256, activation='relu', input_shape=(X_train_nn.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)
])
model_nn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_nn.fit(X_train_nn, y_train_nn, validation_data=(X_val_nn, y_val_nn), epochs=100, batch_size=64, verbose=0, callbacks=[early_stop])

loss, mae = model_nn.evaluate(X_test_nn, y_test_nn, verbose=0)
preds_nn = model_nn.predict(X_test_nn).flatten()
print("\nNeural Network")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_nn, preds_nn)):.2f}")
print(f"R2: {r2_score(y_test_nn, preds_nn):.2f}")

results_df = pd.concat([results_df, pd.DataFrame([{
    'Model': 'Neural Network',
    'MAE': mae,
    'RMSE': np.sqrt(mean_squared_error(y_test_nn, preds_nn)),
    'R2': r2_score(y_test_nn, preds_nn)
}])], ignore_index=True)

# מודל SARIMA
def evaluate_sarima(df, order=(1,1,1), seasonal_order=(1,1,1,7)):
    tscv = TimeSeriesSplit(n_splits=5)
    errors = []

    # איחוד לפי תאריך כדי להסיר כפילויות
    series = df.groupby('QueueStartDate')['TotalAgents'].sum().sort_index()
    series = series.asfreq('D').fillna(method='ffill')

    for train_index, test_index in tscv.split(series):
        train, test = series.iloc[train_index], series.iloc[test_index]
        try:
            model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False)
            results = model.fit(disp=False)
            forecast = results.forecast(steps=len(test))
            rmse = np.sqrt(mean_squared_error(test, forecast))
            errors.append(rmse)
        except Exception as e:
            print(f"SARIMA Fold Error: {e}")
            continue
    return np.mean(errors)

sarima_rmse = evaluate_sarima(cc_df[['QueueStartDate', 'TotalAgents']])
print("\nSARIMA")
print(f"MAE: {np.nan}")
print(f"RMSE: {sarima_rmse:.2f}")
print(f"R2: {np.nan}")

results_df = pd.concat([results_df, pd.DataFrame([{
    'Model': 'SARIMA',
    'MAE': np.nan,
    'RMSE': sarima_rmse,
    'R2': np.nan
}])], ignore_index=True)

# מודל Prophet
def evaluate_prophet(df):
    df_prophet = df[['QueueStartDate', 'TotalAgents']].rename(columns={'QueueStartDate': 'ds', 'TotalAgents': 'y'})
    df_prophet = df_prophet.sort_values('ds')
    tscv = TimeSeriesSplit(n_splits=5)
    errors = []
    for train_index, test_index in tscv.split(df_prophet):
        train, test = df_prophet.iloc[train_index], df_prophet.iloc[test_index]
        model = Prophet()
        model.fit(train)
        future = test[['ds']]
        forecast = model.predict(future)
        rmse = np.sqrt(mean_squared_error(test['y'], forecast['yhat']))
        errors.append(rmse)
    return np.mean(errors)

prophet_rmse = evaluate_prophet(cc_df[['QueueStartDate', 'TotalAgents']])

print("\nProphet")
print(f"MAE: {np.nan}")
print(f"RMSE: {prophet_rmse:.2f}")
print(f"R2: {np.nan}")

results_df = pd.concat([results_df, pd.DataFrame([{
    'Model': 'Prophet',
    'MAE': np.nan,
    'RMSE': prophet_rmse,
    'R2': np.nan
}])], ignore_index=True)

# יצירת טבלת תוצאות מעודכנת

results_df = pd.concat([
    results_df
], ignore_index=True).sort_values(by='RMSE')

print(f"\n🔍 טבלת ביצועים מעודכנת:")
print(results_df.to_string(index=False))

# מודל אופטימלי לפי RMSE
best_model_row = results_df.sort_values(by='RMSE').iloc[0]
print(f"\n✅ המודל האופטימלי לפי RMSE הוא: {best_model_row['Model']}")

# גרף השוואה של ביצועי המודלים
fig_perf = px.bar(results_df, x='Model', y='RMSE', text='RMSE', title='<b>השוואת RMSE בין המודלים</b>')
fig_perf.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig_perf.update_layout(title={'x': 0.5}, yaxis_title='RMSE', xaxis_title='Model')
fig_perf.show()


# מחיקת מודלים ישנים
import os
import glob
model_files = glob.glob('data/*.pkl') + glob.glob('data/*.keras') + glob.glob('data/*.h5')
for f in model_files:
    try:
        os.remove(f)
        print(f"🗑️ נמחק: {f}")
    except Exception as e:
        print(f"⚠️ שגיאה במחיקת {f}: {e}")

# שמירת המודל האופטימלי

best_model_name = best_model_row['Model']

best_model_row = results_df.sort_values(by='RMSE').iloc[0]
best_model_name = best_model_row['Model']
print(f"\n✅ המודל האופטימלי לפי RMSE הוא: {best_model_name}")

if best_model_name == 'Neural Network':
    model_filename = "data/neural_network.keras"
    model_nn.save(model_filename)
    joblib.dump(preprocessor, "data/preprocessor.pkl")
    joblib.dump(scaler, "data/scaler.pkl")

elif best_model_name in models:
    model_filename = f"data/{best_model_name.replace(' ', '_').lower()}.pkl"
    model_with_preprocessing = Pipeline([("prep", preprocessor), ("model", models[best_model_name].named_steps['model'])])
    joblib.dump(model_with_preprocessing, model_filename)

elif best_model_name == 'Prophet':
    prophet_df = cc_df[['QueueStartDate', 'TotalAgents']].rename(columns={
        'QueueStartDate': 'ds',
        'TotalAgents': 'y'
    })
    model = Prophet()
    model.fit(prophet_df)
    model_filename = "data/prophet_model.pkl"
    joblib.dump(model, model_filename)

elif best_model_name == 'SARIMA':
    series = cc_df.set_index('QueueStartDate')['TotalAgents'].asfreq('D').fillna(method='ffill')
    model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,7)).fit(disp=False)
    model_filename = "data/sarima_model.pkl"
    joblib.dump(model, model_filename)

print(f"\n📦 המודל הטוב ביותר נשמר בשם: {model_filename}")

# טעינת המודל מהקובץ
import pathlib
Path = pathlib.Path

def load_model(path):
    if Path(path).exists():
        if path.endswith('.keras'):
            return tf.keras.models.load_model(path)
        elif path.endswith('.pkl'):
            return joblib.load(path)
        else:
            print("⚠️ סיומת לא נתמכת לקובץ המודל.")
            return None
    else:
        print("⚠️ קובץ המודל לא נמצא.")
        return None


# פונקציה לאימון מחדש על דאטה חדש
def retrain_model(model_name, X_new, y_new):
    if model_name == 'Neural Network':
        pre = joblib.load("data/preprocessor.pkl")
        scaler_loaded = joblib.load("data/scaler.pkl")
        X_transformed = pre.transform(X_new)
        X_scaled = scaler_loaded.transform(X_transformed)
        model = tf.keras.models.load_model("data/neural_network.keras")
        model.fit(X_scaled, y_new, epochs=30, batch_size=32, verbose=0)
        model.save("data/neural_network_retrained.keras")
        return model
    elif model_name in models:
        pipe = models.get(model_name)
        pipe.fit(X_new, y_new)
        joblib.dump(pipe, f"data/{model_name.replace(' ', '_').lower()}_retrained.pkl")
        return pipe
    elif model_name == 'Prophet':
        model = Prophet()
        df = pd.concat([X_new, y_new], axis=1).rename(columns={'QueueStartDate': 'ds', 'TotalAgents': 'y'})
        model.fit(df)
        joblib.dump(model, "data/prophet_model_retrained.pkl")
        return model
    elif model_name == 'SARIMA':
        series = pd.concat([X_new, y_new], axis=1).set_index('QueueStartDate')['TotalAgents'].asfreq('D').fillna(method='ffill')
        model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,7)).fit(disp=False)
        joblib.dump(model, "data/sarima_model_retrained.pkl")
        return model
    else:
        print(f"\n⚠️ מודל {model_name} לא נתמך.")
        return None

# פונקציית חיזוי לפי קלט חדש
def predict_total_agents(model_path, weekday: str, interval: float, is_holiday: bool):
    model = load_model(model_path)
    if model is None:
        return None

    ratio_row = answer_ratio_lookup[
        (answer_ratio_lookup['Weekday'] == weekday) &
        (answer_ratio_lookup['Interval'] == interval) &
        (answer_ratio_lookup['IsHoliday'] == is_holiday)
    ]

    if ratio_row.empty:
        print("⚠️ לא נמצא AnswerRatio מתאים, משתמש ב-0.8")
        answer_ratio = 0.8
    else:
        answer_ratio = ratio_row['AnsweredRatio'].values[0]

    input_df = pd.DataFrame([{
        'Weekday': weekday,
        'Interval': interval,
        'IsHoliday': is_holiday,
        'AnsweredRatio': answer_ratio
    }])

    if model_path.endswith('.keras'):
        pre = joblib.load("data/preprocessor.pkl")
        scaler_loaded = joblib.load("data/scaler.pkl")
        X_transformed = pre.transform(input_df)
        X_scaled = scaler_loaded.transform(X_transformed)
        prediction = model.predict(X_scaled)
    else:
        prediction = model.predict(input_df)

    predict = max(0, float(prediction[0]))
    print(f"🔮 תחזית נציגים: {int(round(float(predict)))}")
    return int(round(float(predict)))

# ממשק אינטראקטיבי לחיזוי
import ipywidgets as widgets
from IPython.display import display, clear_output

weekday_widget = widgets.Dropdown(
    options=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'],
    description='Weekday:',
    value='Monday'
)
interval_widget = widgets.FloatSlider(
    value=10.5, min=0, max=23.5, step=0.5,
    description='Interval:'
)
holiday_widget = widgets.Checkbox(
    value=False,
    description='Holiday?'
)
run_button = widgets.Button(description="🔮 חזה")
output_box = widgets.Output()

model_paths = {
    'Neural Network': 'data/neural_network.keras',
    'Random Forest': 'data/random_forest.pkl',
    'Gradient Boosting': 'data/gradient_boosting.pkl',
    'XGBoost': 'data/xgboost.pkl',
    'Linear Regression': 'data/linear_regression.pkl',
    'Prophet': 'data/prophet_model.pkl',
    'SARIMA': 'data/sarima_model.pkl',
}

selected_model_path = model_paths.get(best_model_name)

@output_box.capture()
def on_run_clicked(b):
    clear_output()
    predict_total_agents(
        model_path=selected_model_path,
        weekday=weekday_widget.value,
        interval=interval_widget.value,
        is_holiday=holiday_widget.value
    )

run_button.on_click(on_run_clicked)

print("\n📋 ממשק חיזוי אינטראקטיבי:")
# display(weekday_widget, interval_widget, holiday_widget, run_button, output_box)

# ממשק Gradio לחיזוי
import gradio as gr
from gradio.themes.base import Base

class CustomTheme(Base):
    def __init__(self):
        super().__init__()
        self.primary_hue = "green"
        self.font = "Alef"
        self.button_primary_background = "#00796B"
        self.button_primary_text = "white"

def gradio_predict(weekday, interval, is_holiday):
    return predict_total_agents(model_filename, weekday, interval, is_holiday)

gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Dropdown(['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'], label="יום בשבוע"),
        gr.Dropdown(
            choices=[(label, parse_interval(label)) for label in sorted(cc_df['HalfHourInterval'].unique().tolist())],
            label="אינטרוול (חצי שעה)"
        ),
        gr.Checkbox(label="האם זה חג?")
    ],
    outputs=gr.Number(label="תחזית נציגים", precision=0),
    title="🔮 חיזוי מספר נציגים",
    description="בחר יום בשבוע, אינטרוול והאם זה חג – לקבלת תחזית לפי המודל הטוב ביותר",
    theme=CustomTheme(),
    live=True
).launch(share=True)

