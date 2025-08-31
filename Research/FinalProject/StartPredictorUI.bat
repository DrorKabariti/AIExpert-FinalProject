@echo off
cd /d "%~dp0"
echo Starting FastAPI server...
python -m uvicorn PredictTotalAgentsUI:app --reload
pause
