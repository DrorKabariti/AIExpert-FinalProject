from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import datetime
import logging
import os
from fastapi.openapi.utils import get_openapi

app = FastAPI()

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logging.basicConfig(level=logging.INFO, filename="prediction_logs.log", filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Load model
MODEL_PATH = "TotalAgentsModel.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")
model = joblib.load(MODEL_PATH)
logging.info("Model loaded successfully from %s", MODEL_PATH)

# Request schema
class PredictRequest(BaseModel):
    QueueStartDayNumber: int
    Interval: float
    IsHoliday: bool
    IsWeekend: bool
    IsWorkDay: bool

@app.post("/api/predict", tags=["Prediction"])
async def predict(request: PredictRequest):
    input_df = pd.DataFrame([[
        request.QueueStartDayNumber,
        request.Interval,
        request.IsHoliday,
        request.IsWeekend,
        request.IsWorkDay
    ]], columns=[
        'QueueStartDateNumber', 'Interval', 'IsHoliday', 'IsWeekend', 'IsWorkDay'
    ])

    prediction = model.predict(input_df)[0]
    result = round(prediction, 0)

    # Log the prediction
    logging.info("Prediction input: %s | Result: %s", input_df.to_dict(orient='records')[0], result)

    return {"prediction": result}

# Mount static files from React build folder
if os.path.exists("frontend/build"):
    app.mount("/", StaticFiles(directory="frontend/build", html=True), name="static")
else:
    @app.get("/", response_class=HTMLResponse, tags=["UI"])
    async def root():
        return """
        <html>
            <head><title>Total Agents Predictor</title></head>
            <body>
                <h2>Total Agents Predictor</h2>
                <p>React frontend not found. Build the React app and place it under /frontend/build.</p>
                <p>Or use <a href='/docs'>Swagger UI</a> to test the API.</p>
            </body>
        </html>
        """

# Custom OpenAPI schema (optional for UI branding or clarity)
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Total Agents Predictor API",
        version="1.0.0",
        description="API for predicting number of total agents based on date and interval features",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
