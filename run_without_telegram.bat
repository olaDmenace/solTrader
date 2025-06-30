@echo off
echo 🚀 Starting SolTrader APE Bot (No Telegram)
echo ==========================================

REM Activate virtual environment
call venv\Scripts\activate.bat

echo 📝 Temporarily disabling Telegram integration...

REM Create temporary .env without telegram settings
copy .env .env.backup 2>nul

REM Remove telegram lines from .env temporarily
findstr /v "TELEGRAM_BOT_TOKEN TELEGRAM_CHAT_ID" .env > .env.temp
copy .env.temp .env
del .env.temp

echo 🎮 Starting bot without Telegram notifications...
echo Press Ctrl+C to stop the bot
echo.

python main.py

echo.
echo 🔄 Restoring original .env file...
copy .env.backup .env 2>nul
del .env.backup 2>nul

echo.
echo Bot stopped. Press any key to exit...
pause >nul