@echo off
echo 🌐 Starting SolTrader Web Dashboard
echo ==================================

REM Activate virtual environment
echo 📦 Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if Flask is installed
echo 🔍 Checking Flask installation...
python -c "import flask; print('✅ Flask available')" 2>nul || (
    echo ❌ Flask not found. Installing...
    pip install flask>=2.0.0
)

echo.
echo 🚀 Starting web dashboard...
echo 📊 Dashboard will be available at: http://localhost:5000
echo 🔄 Auto-refreshes every 5 seconds
echo 🛑 Press Ctrl+C to stop
echo.

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

python create_monitoring_dashboard.py

echo.
echo Dashboard stopped. Press any key to exit...
pause >nul