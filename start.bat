@echo off
echo 🚀 Starting ZenStock - Smart Inventory Management Tool
echo.
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Starting Streamlit application...
streamlit run app.py
pause
