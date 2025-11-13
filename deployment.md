# SolTrader Production Deployment Guide

## Overview
This guide provides comprehensive instructions for deploying SolTrader to production environments with enterprise-grade security, monitoring, and reliability.

## Prerequisites

### System Requirements
- **OS**: Ubuntu 20.04+ or Windows Server 2019+
- **RAM**: 8GB minimum, 16GB recommended
- **CPU**: 4 cores minimum, 8 cores recommended
- **Storage**: 100GB SSD minimum
- **Network**: Reliable internet with low latency to Solana mainnet

### Software Dependencies
```bash
# Python 3.11+
sudo apt update
sudo apt install python3.11 python3.11-pip python3.11-venv

# Redis (for caching)
sudo apt install redis-server

# PostgreSQL (for persistent data)
sudo apt install postgresql postgresql-contrib

# Monitoring tools
sudo apt install prometheus grafana
```

## Production Environment Setup

### 1. Environment Configuration

Create production environment file:
```bash
cp production.env.template .env
```

Configure critical production variables:
```bash
# Security
MASTER_ENCRYPTION_KEY=your-256-bit-encryption-key
API_RATE_LIMIT_PER_MINUTE=1000

# Blockchain Configuration
ALCHEMY_RPC_URL=https://solana-mainnet.g.alchemy.com/v2/your-key
HELIUS_RPC_URL=https://mainnet.helius-rpc.com/?api-key=your-key
QUICKNODE_RPC_URL=https://your-endpoint.solana-mainnet.quiknode.pro/your-key/

# Trading Configuration
PAPER_TRADING=false  # CRITICAL: Set to true for testing
INITIAL_CAPITAL=1000.00
MAX_POSITION_SIZE=0.05  # 5% maximum position size
MAX_SLIPPAGE=0.005      # 0.5% maximum slippage

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost/soltrader
REDIS_URL=redis://localhost:6379/0

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
SENTRY_DSN=your-sentry-dsn-for-error-tracking
```

### 2. Security Hardening

#### API Key Management
```bash
# Generate secure API keys
python secrets_manager.py --generate-keys

# Rotate keys automatically
python secrets_manager.py --enable-rotation
```

#### Network Security
```bash
# Configure firewall
sudo ufw allow 22    # SSH
sudo ufw allow 5000  # Dashboard
sudo ufw allow 9090  # Prometheus
sudo ufw allow 3000  # Grafana
sudo ufw enable

# SSL/TLS Configuration
sudo apt install certbot nginx
sudo certbot --nginx -d your-domain.com
```

### 3. Database Setup

#### PostgreSQL Configuration
```sql
-- Create database and user
CREATE DATABASE soltrader;
CREATE USER soltrader_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE soltrader TO soltrader_user;

-- Initialize schema
python -c "from management.data_manager import UnifiedDataManager; UnifiedDataManager().initialize_database()"
```

#### Redis Configuration
```bash
# Edit Redis configuration
sudo nano /etc/redis/redis.conf

# Key configurations:
# maxmemory 2gb
# maxmemory-policy allkeys-lru
# requirepass your_redis_password

sudo systemctl restart redis
```

### 4. Application Deployment

#### Using Docker (Recommended)
```yaml
# docker-compose.yml
version: '3.8'
services:
  soltrader:
    build: .
    restart: unless-stopped
    environment:
      - ENV=production
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    ports:
      - "5000:5000"
    depends_on:
      - redis
      - postgres
  
  redis:
    image: redis:7-alpine
    restart: unless-stopped
    volumes:
      - redis_data:/data
  
  postgres:
    image: postgres:15-alpine
    restart: unless-stopped
    environment:
      POSTGRES_DB: soltrader
      POSTGRES_USER: soltrader_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  redis_data:
  postgres_data:
```

Deploy with Docker:
```bash
docker-compose up -d
```

#### Manual Deployment
```bash
# Create production user
sudo useradd -m -s /bin/bash soltrader

# Setup application
sudo -u soltrader bash
cd /home/soltrader
git clone https://github.com/your-repo/soltrader.git
cd soltrader

# Virtual environment
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements_updated.txt

# Configure environment
cp production.env.template .env
# Edit .env with production values

# Create systemd service
sudo cp deployment/soltrader.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable soltrader
sudo systemctl start soltrader
```

### 5. Monitoring Setup

#### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'soltrader'
    static_configs:
      - targets: ['localhost:8000']  # Metrics endpoint
    scrape_interval: 10s
```

#### Grafana Dashboard
1. Install Grafana dashboard: `monitoring/grafana_dashboard.json`
2. Configure data sources:
   - Prometheus: `http://localhost:9090`
3. Import SolTrader monitoring dashboard
4. Set up alerting rules for critical metrics

### 6. Security Audit

Run comprehensive security audit:
```bash
python security_audit.py --full-scan --export-report
```

Address all CRITICAL and HIGH severity findings before deployment.

## Deployment Checklist

### Pre-Deployment
- [ ] Security audit passed (no critical vulnerabilities)
- [ ] All environment variables configured
- [ ] Database schema initialized
- [ ] SSL certificates installed
- [ ] Firewall rules configured
- [ ] Monitoring dashboards operational

### Deployment
- [ ] Application deployed successfully
- [ ] Health checks passing
- [ ] All services starting correctly
- [ ] Logs being written properly
- [ ] Metrics being collected

### Post-Deployment
- [ ] Trading functionality tested (paper mode first)
- [ ] API endpoints responding
- [ ] Database connections stable
- [ ] Monitoring alerts configured
- [ ] Backup procedures tested

## Monitoring & Maintenance

### Key Metrics to Monitor
- **Trading Performance**: Win rate, PnL, position counts
- **System Health**: CPU, memory, disk usage
- **API Performance**: Response times, error rates, quota usage
- **Database Performance**: Query times, connection counts
- **Network**: Latency, packet loss to Solana RPC

### Log Management
```bash
# Log rotation configuration
sudo nano /etc/logrotate.d/soltrader

# Content:
/home/soltrader/soltrader/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    sharedscripts
}
```

### Backup Strategy
```bash
# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)

# Database backup
pg_dump soltrader > /backups/db_backup_$DATE.sql

# Configuration backup
tar -czf /backups/config_backup_$DATE.tar.gz .env logs/ data/

# Upload to cloud storage (configure as needed)
aws s3 cp /backups/ s3://your-backup-bucket/ --recursive
```

## Troubleshooting

### Common Issues

#### High Memory Usage
```bash
# Monitor memory usage
python performance_profiler.py --profile-memory

# Optimize if needed
python performance_optimizer.py --optimize-memory
```

#### API Rate Limits
```bash
# Check quota usage
python -c "from api.adaptive_quota_manager import AdaptiveQuotaManager; print(AdaptiveQuotaManager().get_quota_status())"

# Implement rate limiting
python -c "from performance_optimizer import ProductionOptimizer; ProductionOptimizer().configure_rate_limiting()"
```

#### Database Performance
```sql
-- Check slow queries
SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;

-- Optimize indexes
ANALYZE;
REINDEX DATABASE soltrader;
```

### Emergency Procedures

#### System Overload
1. Enable emergency controls: `python emergency_controls.py --activate`
2. Reduce position sizes: `python -c "from config.settings import Settings; settings = Settings(); settings.MAX_POSITION_SIZE = 0.01"`
3. Increase monitoring frequency

#### Critical Error Recovery
1. Check error logs: `tail -f logs/trading.log`
2. Run health check: `python health_check.py --comprehensive`
3. Restart services if needed: `sudo systemctl restart soltrader`

## Performance Optimization

### Production Tuning
- **Connection Pooling**: 20-50 connections per service
- **Cache TTL**: 300-600 seconds for market data
- **Rate Limiting**: 100 requests/minute per API
- **Memory Limits**: 4-8GB per process
- **CPU Affinity**: Dedicate cores to critical processes

### Scaling Considerations
- **Horizontal Scaling**: Multiple instances behind load balancer
- **Database Sharding**: Separate read/write operations
- **Cache Clustering**: Redis cluster for high availability
- **CDN**: Static assets and API responses

## Security Best Practices

### Access Control
- Use dedicated service accounts
- Implement least-privilege principles
- Regular key rotation (monthly)
- Multi-factor authentication for admin access

### Network Security
- VPN or bastion host for remote access
- Network segmentation
- DDoS protection
- Regular security updates

### Data Protection
- Encryption at rest and in transit
- Regular security audits
- Compliance with data protection regulations
- Secure backup storage

---

**Support**: For deployment assistance, contact: support@soltrader.com
**Documentation**: https://docs.soltrader.com
**Status Page**: https://status.soltrader.com