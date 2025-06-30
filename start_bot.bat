@echo off
echo 🦍 SolTrader APE Bot Launcher
echo ================================

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ❌ Virtual environment not found!
    echo Please run setup first:
    echo.
    echo python -m venv venv
    echo venv\Scripts\activate
    echo pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo 📦 Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if .env exists
if not exist ".env" (
    echo ❌ .env file not found!
    echo Please create .env file following SETUP_GUIDE.md
    echo.
    pause
    exit /b 1
)

REM Run verification first
echo 🔍 Running setup verification...
python verify_setup.py

if errorlevel 1 (
    echo.
    echo ❌ Setup verification failed!
    echo Please fix the issues above before running the bot.
    echo.
    pause
    exit /b 1
)

echo.
echo ✅ Setup verification passed!
echo.

REM Ask user what to run
echo What would you like to do?
echo.
echo 1) Run Paper Trading (Recommended for first time)
echo 2) Run Live Trading (Only after successful paper trading)
echo 3) Run Tests Only
echo 4) Exit
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo 🎮 Starting Paper Trading...
    echo Press Ctrl+C to stop the bot
    echo.
    python main.py
) else if "%choice%"=="2" (
    echo.
    echo ⚠️  WARNING: LIVE TRADING MODE
    echo This will use real money!
    echo.
    set /p confirm="Type 'CONFIRM' to proceed: "
    if "%confirm%"=="CONFIRM" (
        echo.
        echo 🚨 Starting Live Trading...
        echo Press Ctrl+C to stop the bot
        echo.
        python main.py
    ) else (
        echo Cancelled.
    )
) else if "%choice%"=="3" (
    echo.
    echo 🧪 Running Tests...
    python test_ape_strategy.py
) else if "%choice%"=="4" (
    echo Goodbye!
) else (
    echo Invalid choice.
)

echo.
echo Press any key to exit...
pause >nul