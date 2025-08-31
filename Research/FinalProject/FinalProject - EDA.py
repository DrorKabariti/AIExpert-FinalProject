# -*- coding: utf-8 -*-
"""EDA Script for Call Center Analysis - PyCharm Version"""

import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
from zipfile import ZipFile

# קריאת קבצי CSV
cc_df = pd.read_csv("data/CC_2020-2025_New.csv")
holidays_df = pd.read_csv("data/Holidays.csv")

# שינוי שמות עמודות
cc_df.columns = [col.strip().replace(" ", "_").replace("-", "_") for col in cc_df.columns]
holidays_df.columns = [col.strip().replace(" ", "_").replace("-", "_") for col in holidays_df.columns]

# המרת תאריכים
cc_df['QueueStartDate'] = pd.to_datetime(cc_df['QueueStartDate'], dayfirst=True, errors='coerce')
holidays_df['CalendarDate'] = pd.to_datetime(holidays_df['CalendarDate'], dayfirst=True, errors='coerce')

# עמודות עזר
cc_df['Weekday'] = cc_df['QueueStartDate'].dt.day_name()
cc_df['IsWeekend'] = cc_df['Weekday'].isin(['Friday', 'Saturday'])
cc_df["AnsweredPerAgent"] = cc_df["TotalCallsAnswered"] / cc_df["TotalAgents"]
cc_df["AnsweredPerAgent"] = cc_df["AnsweredPerAgent"].replace([float("inf"), -float("inf")], pd.NA)


# מיזוג עם טבלת חגים
cc_df = cc_df.merge(holidays_df[['CalendarDate', 'HolidayNameHebrew']], left_on='QueueStartDate', right_on='CalendarDate', how='left')
cc_df['IsHoliday'] = cc_df['HolidayNameHebrew'].notna()
cc_df.drop(columns=['CalendarDate', 'HolidayNameHebrew'], inplace=True)

# טיפול בערכים חסרים
print("\nMissing values per column:")
print(cc_df.isna().sum())

# הסרת כפילויות
before = len(cc_df)
cc_df.drop_duplicates(inplace=True)
print(f"\nDuplicates removed: {before - len(cc_df)}")

# תיאור סטטיסטי
print("\nDescriptive stats:")
print(cc_df.describe())

# גרפים
fig1 = px.histogram(cc_df, x='Weekday',
    category_orders={'Weekday': ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']},
    title='<b>מספר רשומות לפי יום בשבוע</b>')
fig1.update_layout(title={'x':0.5})
fig1.show()

fig2 = px.box(cc_df, x='HourInterval', y='TotalAgents', title='<b>התפלגות מספר נציגים לפי אינטרוול שעתי</b>')
fig2.update_layout(title={'x':0.5})
fig2.show()

fig3 = px.box(cc_df, x='HalfHourInterval', y='TotalAgents', title='<b>התפלגות מספר נציגים לפי אינטרוול חצאי שעות</b>')
fig3.update_layout(title={'x':0.5})
fig3.show()

print(cc_df.head())
