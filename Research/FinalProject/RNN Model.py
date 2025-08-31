import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Input
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

# שלב 1: טעינת הנתונים
cc_df = pd.read_csv("data/CC_2020-2025_New.csv")
holidays_df = pd.read_csv("data/Holidays.csv")

# שלב 2: ניקוי ועיבוד עמודות
cc_df.columns = cc_df.columns.str.strip().str.replace(" ", "_").str.replace("-", "_")
holidays_df.columns = holidays_df.columns.str.strip().str.replace(" ", "_").str.replace("-", "_")

cc_df['QueueStartDate'] = pd.to_datetime(cc_df['QueueStartDate'], dayfirst=True, errors='coerce')
holidays_df['CalendarDate'] = pd.to_datetime(holidays_df['CalendarDate'], dayfirst=True, errors='coerce')

# יום בשבוע וחג
cc_df['Weekday'] = cc_df['QueueStartDate'].dt.day_name()
cc_df = cc_df.merge(holidays_df[['CalendarDate']], left_on='QueueStartDate', right_on='CalendarDate', how='left')
cc_df['IsHoliday'] = cc_df['CalendarDate'].notna()
cc_df.drop(columns=['CalendarDate'], inplace=True)

# המרת אינטרוול לפורמט מספרי
cc_df['Interval'] = cc_df['HalfHourInterval'].str.extract(r'(\d{1,2}):(\d{2})')[0].astype(float)
cc_df.loc[cc_df['HalfHourInterval'].str.contains('30'), 'Interval'] += 0.5

# רק עמודות רלוונטיות
features = ['Weekday', 'Interval', 'IsHoliday']
target = 'TotalAgents'
X = cc_df[features]
y = cc_df[target]

# שלב 3: Preprocessing – קידוד וסקלינג
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown='ignore'), ['Weekday']),
    ("num", StandardScaler(), ['Interval', 'IsHoliday'])
])

X_processed = preprocessor.fit_transform(X)
X_dense = X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed
X_rnn = X_dense.reshape((X_dense.shape[0], X_dense.shape[1], 1))  # צור צורת קלט ל־RNN

# פיצול ל־Train/Test
X_train, X_test, y_train, y_test = train_test_split(X_rnn, y, test_size=0.2, random_state=42)

# שלב 4: בניית המודל RNN
model_rnn = Sequential([
    Input(shape=(X_rnn.shape[1], 1)),
    SimpleRNN(64, activation='relu', return_sequences=False),
    Dense(32, activation='relu'),
    Dense(1)
])
model_rnn.compile(optimizer='adam', loss='mse', metrics=['mae'])

# שלב 5: אימון
model_rnn.fit(X_train, y_train, validation_split=0.2, epochs=30, batch_size=32, verbose=1)

# שלב 6: שמירה
model_rnn.save("data/rnn_model.keras")
joblib.dump(preprocessor, "data/rnn_preprocessor.pkl")

# שלב 7: פונקציית חיזוי
def predict_total_agents_rnn(weekday: str, interval: float, is_holiday: bool):
    if is_holiday:
        print("🔕 יום חג – אין נציגים")
        return 0

    model = tf.keras.models.load_model("data/rnn_model.keras")
    pre = joblib.load("data/rnn_preprocessor.pkl")

    input_df = pd.DataFrame([{
        'Weekday': weekday,
        'Interval': interval,
        'IsHoliday': is_holiday
    }])
    input_transformed = pre.transform(input_df)
    input_dense = input_transformed.toarray() if hasattr(input_transformed, "toarray") else input_transformed
    input_rnn = input_dense.reshape((input_dense.shape[0], input_dense.shape[1], 1))

    prediction = model.predict(input_rnn)[0][0]
    print(f"🔮 תחזית למספר נציגים: {int(round(prediction))}")
    return int(round(prediction))

# הערכה רק לימים שאינם חגים
mask = ~cc_df['IsHoliday']
X_eval = X_rnn[mask]
y_eval = y[mask]

# חיזוי וחישוב ביצועים
y_pred = model_rnn.predict(X_eval).flatten()
mae = mean_absolute_error(y_eval, y_pred)
rmse = np.sqrt(mean_squared_error(y_eval, y_pred))
r2 = r2_score(y_eval, y_pred)

print("\n📊 דוח ביצועים - RNN (ללא חגים):")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# דוגמה:
predict_total_agents_rnn('Monday', 10.5, False)
