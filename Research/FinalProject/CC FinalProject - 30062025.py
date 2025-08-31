# 📊 EDA לתחזית נציגים + חיזוי TotalAgents

# 1. טעינת ספריות בסיס
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
# 3. קריאת קבצי CSV
cc_df = pd.read_csv("data/CC_2020-2025_New.csv")
holidays_df = pd.read_csv("data/Holidays.csv")

# 4. עיבוד נתונים ראשוני
cc_df.columns = [col.strip().replace(" ", "_").replace("-", "_") for col in cc_df.columns]
holidays_df.columns = [col.strip().replace(" ", "_").replace("-", "_") for col in holidays_df.columns]
cc_df['QueueStartDate'] = pd.to_datetime(cc_df['QueueStartDate'], dayfirst=True, errors='coerce')
holidays_df['CalendarDate'] = pd.to_datetime(holidays_df['CalendarDate'], dayfirst=True, errors='coerce')
cc_df['Weekday'] = cc_df['QueueStartDate'].dt.day_name()
cc_df['IsWeekend'] = cc_df['Weekday'].isin(['Friday', 'Saturday'])
cc_df = cc_df.merge(holidays_df[['CalendarDate', 'HolidayNameHebrew']], left_on='QueueStartDate', right_on='CalendarDate', how='left')
cc_df['IsHoliday'] = cc_df['HolidayNameHebrew'].notna()
cc_df.drop(columns=['CalendarDate', 'HolidayNameHebrew'], inplace=True)
cc_df.drop_duplicates(inplace=True)

# המרת אינטרוול לפורמט מספרי: '10:00 - 10:30' -> 10.5
cc_df['Interval'] = cc_df['HalfHourInterval'].str.extract(r'(\d{1,2}):(\d{2})')[0].astype(float)
cc_df.loc[cc_df['HalfHourInterval'].str.contains('30'), 'Interval'] += 0.5

# 5. בחירת משתנים לחיזוי
features = ['Weekday', 'Interval', 'IsHoliday']
target = 'TotalAgents'
X = cc_df[features]
y = cc_df[target]

# 6. קידוד וטרנספורמציה
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown='ignore'), ['Weekday']),
    ("num", 'passthrough', ['Interval', 'IsHoliday'])
])

# 7. חלוקה ל-Train/Validation/Test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2


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

# 9. אימון וביצועים
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
print("\n🔍 תוצאות המודלים הקלאסיים:")
print(results_df.to_string(index=False))

# 10. מודל Neural Network (Keras)
X_transformed = preprocessor.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_transformed)
X_train_nn, X_temp_nn, y_train_nn, y_temp_nn = train_test_split(X_scaled, y, test_size=0.4, random_state=42)
X_val_nn, X_test_nn, y_val_nn, y_test_nn = train_test_split(X_temp_nn, y_temp_nn, test_size=0.5, random_state=42)  # 0.2 each

model_nn = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_nn.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])
model_nn.compile(optimizer='adam', loss='mse', metrics=['mae'])
model_nn.fit(X_train_nn, y_train_nn, validation_data=(X_val_nn, y_val_nn), epochs=30, batch_size=32, verbose=0)

loss, mae = model_nn.evaluate(X_test_nn, y_test_nn, verbose=0)
preds_nn = model_nn.predict(X_test_nn).flatten()
print("\nNeural Network")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_nn, preds_nn)):.2f}")
print(f"R2: {r2_score(y_test_nn, preds_nn):.2f}")

# הוספת התוצאה לטבלת הסיכום
results_df = pd.concat([
    results_df,
    pd.DataFrame([{
        'Model': 'Neural Network',
        'MAE': mae,
        'RMSE': np.sqrt(mean_squared_error(y_test_nn, preds_nn)),
        'R2': r2_score(y_test_nn, preds_nn)
    }])
], ignore_index=True).sort_values(by='RMSE')

print("\n🔍 טבלת ביצועים מעודכנת:")
print(results_df.to_string(index=False))

# מודל אופטימלי לפי RMSE
best_model_row = results_df.sort_values(by='RMSE').iloc[0]
print(f"\n✅ המודל האופטימלי לפי RMSE הוא: {best_model_row['Model']}")

# גרף השוואה של ביצועי המודלים
fig_perf = px.bar(results_df, x='Model', y='RMSE', text='RMSE', title='<b>השוואת RMSE בין המודלים</b>')
fig_perf.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig_perf.update_layout(title={'x': 0.5}, yaxis_title='RMSE', xaxis_title='Model')
fig_perf.show()



best_model_name = best_model_row['Model']

if best_model_name == 'Neural Network':
    model_filename = f"data/neural_network.keras"
    model_nn.save(model_filename)
    joblib.dump(preprocessor, "data/preprocessor.pkl")
    joblib.dump(scaler, "data/scaler.pkl")
    print(f"\n📦 המודל נשמר בשם: {model_filename}")
else:
    best_model_pipeline = models.get(best_model_name)
    if best_model_pipeline:
        model_filename = f"data/{best_model_name.replace(' ', '_').lower()}.pkl"
        joblib.dump(best_model_pipeline, model_filename)
        print(f"\n📦 המודל נשמר בשם: {model_filename}")

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

        model_path = "data/neural_network_retrained.keras"
        model.save(model_path)
        print(f"✅ המודל Neural Network אומן מחדש ונשמר כ: {model_path}")
        return model
    else:
        pipe = models.get(model_name)
        if not pipe:
            print(f"⚠️ מודל {model_name} לא נמצא.")
            return None
        pipe.fit(X_new, y_new)
        model_path = f"data/{model_name.replace(' ', '_').lower()}_retrained.pkl"
        joblib.dump(pipe, model_path)
        print(f"✅ המודל {model_name} אומן מחדש ונשמר כ: {model_path}")
        return pipe

# פונקציית חיזוי לפי קלט חדש

# 🧪 דוגמה לשימוש בפונקציית חיזוי
# model_path = 'data/random_forest.pkl'  # או בהתאם לשם המודל השמור
# predict_total_agents(model_path, weekday='Monday', interval=10.5, is_holiday=False)


def predict_total_agents(model_path, weekday: str, interval: float, is_holiday: bool):
    model = load_model(model_path)
    if model is None:
        return None
    input_df = pd.DataFrame([{
        'Weekday': weekday,
        'Interval': interval,
        'IsHoliday': is_holiday
    }])
    if model_path.endswith('.keras'):
        pre = joblib.load("data/preprocessor.pkl")
        scaler_loaded = joblib.load("data/scaler.pkl")
        X_transformed = pre.transform(input_df)
        X_scaled = scaler_loaded.transform(X_transformed)
        prediction = model.predict(X_scaled)
    else:
        prediction = model.predict(input_df)
    print(f"🔮 תחזית למספר נציגים: {int(round(float(prediction[0])))}")
    return int(round(float(prediction[0])))

# ממשק אינטראקטיבי לחיזוי
import ipywidgets as widgets
from IPython.display import display

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
model_path_widget = widgets.Text(
    value=model_filename,
    description='Model Path:'
)

run_button = widgets.Button(description="🔮 חזה")
output_box = widgets.Output()

def on_run_clicked(b):
    with output_box:
        output_box.clear_output()
        predict_total_agents(
            model_path=model_path_widget.value,
            weekday=weekday_widget.value,
            interval=interval_widget.value,
            is_holiday=holiday_widget.value
        )

run_button.on_click(on_run_clicked)

print("\n📋 ממשק חיזוי אינטראקטיבי:")
# display(weekday_widget, interval_widget, holiday_widget, model_path_widget, run_button, output_box)

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

gr_interface = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Dropdown(['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'], label="יום בשבוע"),
        gr.Slider(0, 23.5, step=0.5, label="אינטרוול (שעות)"),
        gr.Checkbox(label="האם זה חג?")
    ],
    outputs=gr.Number(label="תחזית נציגים", precision=0),
    title="🔮 חיזוי מספר נציגים",
    description="בחר יום בשבוע, אינטרוול והאם זה חג – לקבלת תחזית",
    theme=CustomTheme(),
    live=True
)

gr_interface.launch(share=True)

