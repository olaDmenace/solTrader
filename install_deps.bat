@echo off
echo 🔧 SolTrader Dependency Installer
echo ================================

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo 📦 Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        echo Make sure Python is installed and in PATH
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo 📦 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip first
echo 🔄 Upgrading pip...
python -m pip install --upgrade pip

REM Clear any cached packages
echo 🧹 Clearing pip cache...
pip cache purge

REM Try to install using updated requirements
echo 📥 Installing packages (attempt 1 - updated versions)...
pip install -r requirements_updated.txt
if not errorlevel 1 (
    echo ✅ Installation successful with updated requirements!
    goto :verify
)

echo ⚠️ Updated requirements failed, trying individual packages...

REM Install core packages individually
echo 📥 Installing core packages individually...
pip install "aiohttp>=3.9.0" --no-cache-dir
pip install "base58>=2.1.1" --no-cache-dir
pip install "python-dotenv>=1.0.0" --no-cache-dir
pip install "numpy>=1.24.3" --no-cache-dir
pip install "pandas>=2.0.3" --no-cache-dir
pip install "pytest>=7.4.3" --no-cache-dir
pip install "async-timeout>=4.0.0" --no-cache-dir
pip install "backoff>=2.2.1" --no-cache-dir

REM Try Solana packages
echo 📥 Installing Solana packages...
pip install "solana>=0.32.0" --no-cache-dir
if errorlevel 1 (
    echo ⚠️ Solana package failed, trying older version...
    pip install "solana>=0.30.0" --no-cache-dir
)

pip install "solders>=0.20.0" --no-cache-dir
if errorlevel 1 (
    echo ⚠️ Solders package failed, trying older version...
    pip install "solders>=0.18.0" --no-cache-dir
)

REM Try telegram bot
echo 📥 Installing Telegram bot...
pip install "python-telegram-bot>=20.0" --no-cache-dir
if errorlevel 1 (
    echo ⚠️ Telegram bot failed, trying older version...
    pip install "python-telegram-bot>=13.0" --no-cache-dir
)

REM Try anchorpy
echo 📥 Installing AnchorPy...
pip install "anchorpy>=0.19.0" --no-cache-dir
if errorlevel 1 (
    echo ⚠️ AnchorPy failed, trying older version...
    pip install "anchorpy>=0.18.0" --no-cache-dir
)

:verify
echo.
echo 🔍 Verifying installation...
python -c "import aiohttp; print('✅ aiohttp:', aiohttp.__version__)" 2>nul || echo "❌ aiohttp failed"
python -c "import base58; print('✅ base58: OK')" 2>nul || echo "❌ base58 failed"
python -c "import dotenv; print('✅ python-dotenv: OK')" 2>nul || echo "❌ python-dotenv failed"
python -c "import numpy; print('✅ numpy: OK')" 2>nul || echo "❌ numpy failed"
python -c "import pandas; print('✅ pandas: OK')" 2>nul || echo "❌ pandas failed"

echo.
echo 🧪 Running setup verification...
python verify_setup.py

if errorlevel 1 (
    echo.
    echo ❌ Some packages failed to install properly.
    echo 📖 Please check DEPENDENCY_FIX.md for alternative solutions.
    echo.
    echo Common fixes:
    echo 1. Install Visual Studio Build Tools
    echo 2. Use Python 3.11 instead of 3.12
    echo 3. Install packages from conda instead of pip
) else (
    echo.
    echo 🎉 Installation completed successfully!
    echo You can now run: python main.py
)

echo.
echo Press any key to exit...
pause >nul