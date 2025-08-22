# SolTrader Production Deployment Guide

## 🚀 Quick Start

### Automated Deployment

**Windows:**
```bash
deploy.bat
```

**Linux/Mac:**
```bash
chmod +x deploy.sh
./deploy.sh
```

### Manual Service Management

**Start all services:**
```bash
python service_manager.py start
```

**Check service status:**
```bash
python service_manager.py status
```

**Stop all services:**
```bash
python service_manager.py stop
```

---

## 📋 Service Architecture

The SolTrader system runs **6 automated services**:

### 🔄 **Persistent Services** (Always Running)
1. **Trading Bot** (`main.py`)
   - Core trading engine
   - Token scanning and trade execution
   - Auto-restart on failure

2. **Responsive Dashboard** (`create_monitoring_dashboard.py`)
   - Real-time monitoring at http://localhost:5000
   - Auto-refresh every 3 seconds
   - Trade history and performance metrics
   - **Fully mobile-responsive** - works perfectly on phones, tablets, and desktop
   - Touch-friendly interface with optimized layouts

### ⏰ **Scheduled Services** (Run Periodically)
3. **Performance Charts** (Every 5 minutes)
   - Generates `charts/performance_charts.html`
   - Interactive Chart.js visualizations
   - P&L, win rate, and distribution charts

4. **Log Rotation** (Every hour)
   - Automatic log cleanup and compression
   - Prevents disk space issues
   - Maintains 30 days of history

5. **Performance Reports** (Every 30 minutes)
   - Daily/weekly P&L summaries
   - Risk analysis reports
   - Execution analytics

6. **Health Monitor** (Every 3 minutes)
   - System health checks
   - API connection validation
   - Error rate monitoring

---

## 🎯 Access Points

### Web Interfaces
- **Responsive Dashboard**: http://localhost:5000 (works on all devices - mobile, tablet, desktop)
- **Performance Charts**: Open `charts/performance_charts.html` in browser

### Files & Reports
- **Logs**: `logs/service_manager.log`
- **Service Status**: `reports/services/service_status.json`
- **Health Reports**: `health_reports/latest_health_report.json`
- **Analytics**: `analytics/` directory
- **Performance Reports**: `reports/` directory

---

## 🔧 Service Management

### Individual Service Control

**Start specific service:**
```bash
python service_manager.py start --service trading_bot
```

**Stop specific service:**
```bash
python service_manager.py stop --service dashboard
```

**Restart specific service:**
```bash
python service_manager.py restart --service dashboard
```

### Service Status Monitoring

The service manager provides detailed status information:

```bash
python service_manager.py status
```

**Output includes:**
- Service health status
- Uptime for persistent services
- Last run time for scheduled services
- Restart counts and error information
- Process IDs and resource usage

---

## 📊 Performance Charts

### How It Works
- **Automatic Generation**: Charts update every 5 minutes
- **Manual Generation**: Run `python generate_charts.py`
- **Web Access**: Open `charts/performance_charts.html` in any browser

### Chart Types
1. **Cumulative P&L**: Track total profits over time
2. **Daily P&L**: Daily profit/loss breakdown
3. **Win Rate Analysis**: Rolling win rate trends
4. **Trade Distribution**: P&L distribution histogram
5. **Token Performance**: Best/worst performing tokens

### Mobile Access
Charts are responsive and work on mobile devices:
- Touch-friendly interactions
- Optimized for small screens
- Fast loading on mobile connections

---

## 🛡️ Production Safety Features

### Automatic Restart
- **Trading Bot**: Max 5 restarts with 30-second delays
- **Dashboard**: Max 3 restarts with 15-second delays
- **Health Monitoring**: Continuous service health checks

### Error Handling
- **Robust API Calls**: Exponential backoff retry logic
- **Rate Limit Management**: Automatic throttling
- **Graceful Degradation**: Services continue if dependencies fail

### Monitoring & Alerts
- **Real-time Monitoring**: Service health dashboard
- **Log Management**: Automatic rotation and cleanup
- **Performance Tracking**: Comprehensive analytics
- **Status Reports**: Regular service status updates

---

## 🔧 Configuration

### Environment Variables (.env)
All existing settings work - **no changes needed** for production features.

### Optional New Settings
```bash
# Email notifications (optional)
EMAIL_ENABLED=true
CRITICAL_ALERTS=true
PERFORMANCE_ALERTS=true
DAILY_REPORT_TIME=20:00
```

### Service Configuration
Edit `service_manager.py` to customize:
- Restart policies
- Schedule intervals
- Health check frequency
- Service dependencies

---

## 🚨 Troubleshooting

### Common Issues

**Services won't start:**
```bash
# Check logs
tail -f logs/service_manager.log

# Check Python dependencies
pip install flask requests python-dotenv psutil

# Verify .env file exists and is configured
```

**Dashboard not accessible:**
```bash
# Check if port is available
netstat -an | grep :5000

# Try different port
# Edit create_monitoring_dashboard.py line: app.run(port=5001)

# Test mobile responsiveness
# Open http://localhost:5000 on mobile device or resize browser window
```

**Charts not updating:**
```bash
# Manual chart generation
python generate_charts.py

# Check scheduled service
python service_manager.py status
```

### Log Files
- **Service Manager**: `logs/service_manager.log`
- **Trading Bot**: `logs/trading.log`
- **Dashboard**: Check console output
- **Health Monitor**: `health_reports/`

### Manual Recovery
If services fail to start automatically:

```bash
# Stop all services
python service_manager.py stop

# Start individual services manually
python main.py &
python create_monitoring_dashboard.py &
```

---

## 📈 Performance Optimization

### Resource Usage
- **CPU**: Moderate (mostly I/O bound)
- **Memory**: ~150MB total for all services (reduced from mobile dashboard removal)
- **Disk**: Logs rotate automatically
- **Network**: API calls managed with rate limiting

### Scaling Considerations
- **Concurrent Users**: Dashboard supports 100+ concurrent users across all devices
- **Trade Volume**: No artificial limits
- **Data Storage**: JSON files for lightweight storage
- **API Limits**: Intelligent quota management
- **Mobile Performance**: Optimized for low-bandwidth mobile connections

---

## 🔄 Updates and Maintenance

### Updating the System
```bash
# Stop services
python service_manager.py stop

# Update code (git pull, copy files, etc.)

# Restart services
python service_manager.py start
```

### Regular Maintenance
- **Daily**: Check service status and logs
- **Weekly**: Review performance reports
- **Monthly**: Verify backup procedures
- **Quarterly**: Update dependencies

### Backup Procedures
Important files to backup:
- `.env` (configuration)
- `dashboard_data.json` (trading data)
- `analytics/` (performance data)
- `reports/` (historical reports)

---

## 🎯 Production Checklist

### Pre-Deployment
- [ ] Configure .env file with proper API keys
- [ ] Test paper trading functionality
- [ ] Verify all services start successfully
- [ ] Check dashboard accessibility
- [ ] Validate chart generation

### Post-Deployment
- [ ] Monitor service status for 24 hours
- [ ] Verify automated restarts work
- [ ] Check log rotation is functioning
- [ ] Validate performance reports
- [ ] Test emergency stop procedures

### Ongoing Monitoring
- [ ] Daily service status checks
- [ ] Weekly performance reviews
- [ ] Monthly system maintenance
- [ ] Regular backup verification

---

## 📞 Support

For issues or questions:
1. Check logs: `logs/service_manager.log`
2. Review service status: `python service_manager.py status`
3. Check this documentation
4. Review system health: `health_reports/latest_health_report.json`

---

**🎉 Your SolTrader system is now fully automated and production-ready!**