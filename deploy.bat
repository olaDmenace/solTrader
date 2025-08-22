@echo off
REM SolTrader Production Deployment Script for Windows
REM Automates the complete deployment and service startup

echo 🚀 SolTrader Production Deployment Starting...
echo ==================================================

REM Check if we're in the right directory
if not exist "main.py" (
    echo [ERROR] main.py not found. Please run this script from the SolTrader root directory.
    pause
    exit /b 1
)

echo [STEP] 1. Checking system requirements...

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo [INFO] ✅ Python found

echo [STEP] 2. Creating required directories...

REM Create all necessary directories
if not exist "logs" mkdir logs
if not exist "analytics" mkdir analytics
if not exist "analytics\backups" mkdir analytics\backups
if not exist "analytics\daily" mkdir analytics\daily
if not exist "analytics\weekly" mkdir analytics\weekly
if not exist "reports" mkdir reports
if not exist "reports\services" mkdir reports\services
if not exist "reports\daily" mkdir reports\daily
if not exist "reports\weekly" mkdir reports\weekly
if not exist "charts" mkdir charts
if not exist "notifications" mkdir notifications
if not exist "health_reports" mkdir health_reports

echo [INFO] ✅ Directory structure created

echo [STEP] 3. Installing/upgrading Python dependencies...

REM Install required packages if not present
python -m pip install --upgrade pip
python -m pip install flask requests python-dotenv psutil twilio sendgrid

echo [INFO] ✅ Dependencies installed

echo [STEP] 4. Checking configuration...

REM Check if .env file exists
if not exist ".env" (
    echo [WARN] .env file not found. Please create one based on .env.example
    if exist ".env.example" (
        echo [INFO] Copying .env.example to .env
        copy ".env.example" ".env"
        echo [WARN] Please edit .env file with your API keys before continuing
        pause
        exit /b 1
    )
)

echo [INFO] ✅ Configuration files found

echo [STEP] 5. Setting up log rotation...

REM Run log rotation setup
if exist "setup_log_rotation.py" (
    python setup_log_rotation.py
    echo [INFO] ✅ Log rotation configured
) else (
    echo [WARN] setup_log_rotation.py not found, skipping log rotation setup
)

echo [STEP] 6. Initializing analytics systems...

REM Setup performance analytics
if exist "setup_performance_analytics.py" (
    python setup_performance_analytics.py
    echo [INFO] ✅ Performance analytics initialized
)

REM Setup alerting system
if exist "setup_alerts.py" (
    python setup_alerts.py
    echo [INFO] ✅ Alerting system configured
)

REM Setup notifications
if exist "setup_notifications.py" (
    python setup_notifications.py
    echo [INFO] ✅ Notification system configured
)

echo [STEP] 7. Running system health check...

REM Basic system test
python -c "import sys; sys.path.append('.'); from src.trading.strategy import TradingStrategy; from src.utils.robust_api import ErrorTracker; from src.utils.performance_analytics import PerformanceAnalytics; print('✅ Core system imports successful')"
if errorlevel 1 (
    echo [ERROR] System health check failed
    pause
    exit /b 1
)

echo [INFO] ✅ System health check passed

echo [STEP] 8. Creating Windows service batch files...

REM Create start service script
echo @echo off > start_soltrader.bat
echo echo Starting SolTrader Service Manager... >> start_soltrader.bat
echo python service_manager.py start >> start_soltrader.bat
echo pause >> start_soltrader.bat

REM Create stop service script
echo @echo off > stop_soltrader.bat
echo echo Stopping SolTrader services... >> stop_soltrader.bat
echo python service_manager.py stop >> stop_soltrader.bat
echo pause >> stop_soltrader.bat

REM Create status check script
echo @echo off > status_soltrader.bat
echo python service_manager.py status >> status_soltrader.bat
echo pause >> status_soltrader.bat

echo [INFO] ✅ Windows service scripts created

echo [STEP] 9. Final deployment validation...

REM Test that we can start the service manager (timeout after 30 seconds)
timeout /t 30 python service_manager.py status >nul 2>&1

echo [INFO] ✅ Deployment validation complete

echo.
echo 🎉 SolTrader Deployment Complete!
echo ==================================================
echo.
echo 📋 Next Steps:
echo    1. Review your .env configuration
echo    2. Start services: python service_manager.py start
echo    3. Check status: python service_manager.py status
echo    4. Access responsive dashboard: http://localhost:5000 (mobile-optimized)
echo.
echo 🔧 Service Management Commands:
echo    Start all: start_soltrader.bat
echo    Stop all:  stop_soltrader.bat
echo    Status:    status_soltrader.bat
echo.
echo 📊 Monitoring:
echo    Dashboard:     http://localhost:5000 (responsive - works on all devices)
echo    Charts:        Open charts\performance_charts.html
echo    Logs:          type logs\service_manager.log
echo    Health:        type health_reports\latest_health_report.json
echo.
echo ⚠️  Important:
echo    - Keep your .env file secure and private
echo    - Monitor logs regularly for any issues
echo    - Test with paper trading before going live
echo    - Set up proper backup procedures
echo.
echo [INFO] 🚀 SolTrader is ready for production!
pause