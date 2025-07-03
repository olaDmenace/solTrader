@echo off
echo 🔧 Installing ALL Dependencies for SolTrader
echo ============================================

REM Activate virtual environment
echo 📦 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo 🔄 Upgrading pip...
python -m pip install --upgrade pip

echo.
echo 📥 Installing all required packages...
echo.

REM Core packages
echo Installing core packages...
pip install aiohttp>=3.9.0
pip install base58>=2.1.1
pip install python-dotenv>=1.0.0
pip install numpy>=1.24.3
pip install pandas>=2.0.3
pip install scipy>=1.11.3
pip install pytest>=7.4.3
pip install pytest-asyncio>=0.21.1
pip install async-timeout>=4.0.0
pip install backoff>=2.2.1

REM File I/O packages
echo Installing file I/O packages...
pip install aiofiles>=23.0.0

REM Solana packages (compatible versions)
echo Installing Solana packages...
pip install solana==0.32.0
pip install "solders>=0.18.0,<0.19.0"
pip install anchorpy==0.18.0

REM Telegram bot
echo Installing Telegram bot...
pip install python-telegram-bot>=20.0

REM Additional common packages that might be needed
echo Installing additional packages...
pip install requests>=2.31.0
pip install websockets>=11.0
pip install typing-extensions>=4.5.0
pip install python-dateutil>=2.8.0
pip install pytz>=2023.3
pip install colorama>=0.4.6
pip install rich>=13.0.0

REM Web dashboard
echo Installing web dashboard packages...
pip install flask>=2.0.0
pip install flask-cors>=4.0.0

REM Machine learning packages (if needed)
echo Installing ML packages...
pip install scikit-learn>=1.3.0
pip install matplotlib>=3.7.0

REM Data processing
echo Installing data processing packages...
pip install xmltodict
pip install lxml

echo.
echo 🔍 Verifying all installations...
echo.

REM Test each package
python -c "import aiohttp; print('✅ aiohttp: OK')" 2>nul || echo "❌ aiohttp: FAILED"
python -c "import aiofiles; print('✅ aiofiles: OK')" 2>nul || echo "❌ aiofiles: FAILED"
python -c "import base58; print('✅ base58: OK')" 2>nul || echo "❌ base58: FAILED"
python -c "from dotenv import load_dotenv; print('✅ python-dotenv: OK')" 2>nul || echo "❌ dotenv: FAILED"
python -c "import numpy; print('✅ numpy: OK')" 2>nul || echo "❌ numpy: FAILED"
python -c "import pandas; print('✅ pandas: OK')" 2>nul || echo "❌ pandas: FAILED"
python -c "import scipy; print('✅ scipy: OK')" 2>nul || echo "❌ scipy: FAILED"
python -c "import solana; print('✅ solana: OK')" 2>nul || echo "❌ solana: FAILED"
python -c "import solders; print('✅ solders: OK')" 2>nul || echo "❌ solders: FAILED"
python -c "import anchorpy; print('✅ anchorpy: OK')" 2>nul || echo "❌ anchorpy: FAILED"
python -c "import telegram; print('✅ telegram: OK')" 2>nul || echo "❌ telegram: FAILED"

echo.
echo 🧪 Running comprehensive verification...
python verify_setup.py

echo.
echo 🎮 Testing bot startup...
echo This will test if the bot can start without errors...
timeout /t 5 >nul
python -c "
try:
    from src.config.settings import load_settings
    from src.trading.strategy import TradingStrategy, TradingMode
    from src.api.alchemy import AlchemyClient
    from src.api.jupiter import JupiterClient
    from src.token_scanner import TokenScanner
    from src.phantom_wallet import PhantomWallet
    from src.telegram_bot import TradingBot as TgramBot
    print('✅ All imports successful!')
    print('🎉 Bot is ready to run!')
except Exception as e:
    print(f'❌ Import error: {e}')
    print('Some dependencies may still be missing.')
"

echo.
echo 📋 Installation Complete!
echo.
echo Next steps:
echo 1. If all checks passed: python main.py
echo 2. If imports failed: Check error messages above
echo 3. For help: Check DEPENDENCY_FIX.md
echo.
pause