@echo off
echo 🔧 Installing Missing Packages for SolTrader
echo =============================================

REM Activate virtual environment
echo 📦 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo 🔄 Upgrading pip...
python -m pip install --upgrade pip

echo.
echo 📥 Installing missing packages...
echo.

REM Install python-dotenv (for .env file support)
echo Installing python-dotenv...
pip install python-dotenv>=1.0.0
if errorlevel 1 (
    echo ❌ Failed to install python-dotenv
) else (
    echo ✅ python-dotenv installed successfully
)

REM Install scipy
echo Installing scipy...
pip install scipy>=1.11.3
if errorlevel 1 (
    echo ❌ Failed to install scipy
) else (
    echo ✅ scipy installed successfully
)

REM Install python-telegram-bot
echo Installing python-telegram-bot...
pip install python-telegram-bot>=20.0
if errorlevel 1 (
    echo ⚠️ Trying older version of python-telegram-bot...
    pip install python-telegram-bot>=13.0
    if errorlevel 1 (
        echo ❌ Failed to install python-telegram-bot
    ) else (
        echo ✅ python-telegram-bot (older version) installed successfully
    )
) else (
    echo ✅ python-telegram-bot installed successfully
)

REM Install solana
echo Installing solana...
pip install solana>=0.32.0
if errorlevel 1 (
    echo ⚠️ Trying older version of solana...
    pip install solana>=0.30.0
    if errorlevel 1 (
        echo ❌ Failed to install solana
    ) else (
        echo ✅ solana (older version) installed successfully
    )
) else (
    echo ✅ solana installed successfully
)

REM Install solders
echo Installing solders...
pip install "solders>=0.20.0,<0.22.0"
if errorlevel 1 (
    echo ⚠️ Trying compatible version of solders...
    pip install "solders>=0.18.0,<0.19.0"
    if errorlevel 1 (
        echo ❌ Failed to install solders
    ) else (
        echo ✅ solders (compatible version) installed successfully
    )
) else (
    echo ✅ solders installed successfully
)

REM Install anchorpy
echo Installing anchorpy...
pip install anchorpy>=0.19.0
if errorlevel 1 (
    echo ⚠️ Trying older version of anchorpy...
    pip install anchorpy>=0.18.0
    if errorlevel 1 (
        echo ❌ Failed to install anchorpy
    ) else (
        echo ✅ anchorpy (older version) installed successfully
    )
) else (
    echo ✅ anchorpy installed successfully
)

echo.
echo 🔍 Verifying installations...
echo.

python -c "import dotenv; print('✅ python-dotenv: OK')" 2>nul || echo "❌ python-dotenv: FAILED"
python -c "import scipy; print('✅ scipy: OK')" 2>nul || echo "❌ scipy: FAILED"
python -c "import telegram; print('✅ python-telegram-bot: OK')" 2>nul || echo "❌ telegram: FAILED"
python -c "import solana; print('✅ solana: OK')" 2>nul || echo "❌ solana: FAILED"
python -c "import solders; print('✅ solders: OK')" 2>nul || echo "❌ solders: FAILED"
python -c "import anchorpy; print('✅ anchorpy: OK')" 2>nul || echo "❌ anchorpy: FAILED"

echo.
echo 🧪 Running full setup verification...
python verify_setup.py

echo.
echo 📋 Installation Summary:
echo If all packages show "OK" above, you're ready to run the bot!
echo If any packages failed, check the error messages above.
echo.
echo Next steps:
echo 1. If verification passed: python main.py
echo 2. If verification failed: Check DEPENDENCY_FIX.md
echo.
pause