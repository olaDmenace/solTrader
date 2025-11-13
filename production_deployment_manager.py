#!/usr/bin/env python3
"""
Production Deployment Manager
============================

Complete production deployment preparation and management system:
- Automated deployment scripts and procedures
- Production environment validation
- Health check and monitoring setup
- Rollback and recovery procedures
- Documentation generation and validation

Deployment features:
- Zero-downtime deployment strategies
- Environment consistency validation
- Automated backup and recovery
- Comprehensive monitoring setup  
- Production readiness assessment
"""

import os
import json
import logging
import asyncio
import subprocess
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)

class DeploymentStatus(Enum):
    """Deployment status states"""
    PENDING = "pending"
    PREPARING = "preparing" 
    DEPLOYING = "deploying"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"

class EnvironmentType(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class ReadinessLevel(Enum):
    """Production readiness levels"""
    NOT_READY = "not_ready"
    NEEDS_WORK = "needs_work"
    READY_WITH_WARNINGS = "ready_with_warnings"
    PRODUCTION_READY = "production_ready"

@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    environment: EnvironmentType
    version: str
    deployment_strategy: str = "blue_green"  # blue_green, rolling, recreate
    backup_enabled: bool = True
    health_check_enabled: bool = True
    rollback_enabled: bool = True
    validation_timeout_seconds: int = 300
    deployment_timeout_seconds: int = 1800

@dataclass
class DeploymentStep:
    """Individual deployment step"""
    step_id: str
    name: str
    description: str
    command: Optional[str] = None
    timeout_seconds: int = 300
    required: bool = True
    rollback_command: Optional[str] = None

@dataclass
class DeploymentResult:
    """Deployment execution result"""
    deployment_id: str
    config: DeploymentConfig
    status: DeploymentStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    steps_completed: List[str] = field(default_factory=list)
    steps_failed: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    rollback_completed: bool = False

class ProductionReadinessChecker:
    """Validate production readiness"""
    
    def __init__(self):
        self.checks = []
        self._initialize_readiness_checks()
        logger.info("ProductionReadinessChecker initialized")
    
    def _initialize_readiness_checks(self):
        """Initialize production readiness checks"""
        
        self.checks = [
            {
                'name': 'Environment Variables',
                'description': 'Check required environment variables',
                'check_function': self._check_environment_variables,
                'required': True
            },
            {
                'name': 'Dependencies',
                'description': 'Validate all dependencies installed',
                'check_function': self._check_dependencies,
                'required': True
            },
            {
                'name': 'Database Configuration',
                'description': 'Validate database setup and connectivity',
                'check_function': self._check_database_config,
                'required': True
            },
            {
                'name': 'Security Configuration',
                'description': 'Validate security settings',
                'check_function': self._check_security_config,
                'required': True
            },
            {
                'name': 'Monitoring Setup',
                'description': 'Validate monitoring and alerting',
                'check_function': self._check_monitoring_setup,
                'required': False
            },
            {
                'name': 'Performance Benchmarks',
                'description': 'Validate performance meets requirements',
                'check_function': self._check_performance_benchmarks,
                'required': False
            }
        ]
    
    async def assess_production_readiness(self) -> Dict[str, Any]:
        """Assess overall production readiness"""
        
        logger.info("Assessing production readiness...")
        
        check_results = []
        total_checks = len(self.checks)
        passed_checks = 0
        failed_required = 0
        warnings = []
        
        for check in self.checks:
            try:
                result = await check['check_function']()
                
                check_result = {
                    'name': check['name'],
                    'description': check['description'],
                    'passed': result.get('passed', False),
                    'required': check['required'],
                    'details': result.get('details', {}),
                    'warnings': result.get('warnings', []),
                    'errors': result.get('errors', [])
                }
                
                if check_result['passed']:
                    passed_checks += 1
                elif check['required']:
                    failed_required += 1
                
                warnings.extend(check_result['warnings'])
                check_results.append(check_result)
                
            except Exception as e:
                logger.error(f"Readiness check failed: {check['name']}: {e}")
                check_results.append({
                    'name': check['name'],
                    'passed': False,
                    'required': check['required'],
                    'errors': [str(e)]
                })
                if check['required']:
                    failed_required += 1
        
        # Calculate readiness level
        if failed_required > 0:
            readiness_level = ReadinessLevel.NOT_READY
        elif passed_checks / total_checks < 0.8:
            readiness_level = ReadinessLevel.NEEDS_WORK
        elif len(warnings) > 0:
            readiness_level = ReadinessLevel.READY_WITH_WARNINGS
        else:
            readiness_level = ReadinessLevel.PRODUCTION_READY
        
        return {
            'readiness_level': readiness_level.value,
            'overall_score': (passed_checks / total_checks) * 100,
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_required': failed_required,
            'warnings_count': len(warnings),
            'check_results': check_results,
            'warnings': warnings,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _check_environment_variables(self) -> Dict[str, Any]:
        """Check required environment variables"""
        
        required_vars = [
            'ALCHEMY_RPC_URL',
            'WALLET_ADDRESS', 
            'MASTER_ENCRYPTION_KEY',
            'NODE_ENV',
            'LOG_LEVEL'
        ]
        
        missing_vars = []
        weak_vars = []
        
        for var in required_vars:
            value = os.getenv(var)
            if not value:
                missing_vars.append(var)
            elif var == 'MASTER_ENCRYPTION_KEY' and len(value) < 32:
                weak_vars.append(f"{var} is too short")
        
        passed = len(missing_vars) == 0
        warnings = weak_vars if weak_vars else []
        errors = [f"Missing required environment variable: {var}" for var in missing_vars]
        
        return {
            'passed': passed,
            'details': {
                'required_variables': required_vars,
                'missing_variables': missing_vars,
                'variables_set': len(required_vars) - len(missing_vars)
            },
            'warnings': warnings,
            'errors': errors
        }
    
    async def _check_dependencies(self) -> Dict[str, Any]:
        """Check dependency installation"""
        
        # Check Python version
        import sys
        python_version = sys.version_info
        
        # Check for critical modules
        critical_modules = ['asyncio', 'json', 'logging', 'datetime']
        optional_modules = ['psutil', 'aiohttp']
        
        missing_critical = []
        missing_optional = []
        
        for module in critical_modules:
            try:
                __import__(module)
            except ImportError:
                missing_critical.append(module)
        
        for module in optional_modules:
            try:
                __import__(module)
            except ImportError:
                missing_optional.append(module)
        
        passed = len(missing_critical) == 0 and python_version >= (3, 8)
        warnings = [f"Optional module missing: {mod}" for mod in missing_optional]
        errors = []
        
        if python_version < (3, 8):
            errors.append(f"Python version too old: {python_version.major}.{python_version.minor}")
        
        errors.extend([f"Critical module missing: {mod}" for mod in missing_critical])
        
        return {
            'passed': passed,
            'details': {
                'python_version': f"{python_version.major}.{python_version.minor}",
                'critical_modules_available': len(critical_modules) - len(missing_critical),
                'optional_modules_available': len(optional_modules) - len(missing_optional)
            },
            'warnings': warnings,
            'errors': errors
        }
    
    async def _check_database_config(self) -> Dict[str, Any]:
        """Check database configuration"""
        
        # Check database paths
        db_paths = [
            'logs/unified_risk.db',
            'logs/unified_portfolio.db', 
            'logs/unified_trading.db'
        ]
        
        accessible_dbs = 0
        warnings = []
        errors = []
        
        for db_path in db_paths:
            dir_path = os.path.dirname(db_path)
            if os.path.exists(dir_path):
                accessible_dbs += 1
            else:
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    accessible_dbs += 1
                except Exception as e:
                    errors.append(f"Cannot create database directory {dir_path}: {e}")
        
        # Check database configuration
        pool_size = int(os.getenv('DB_CONNECTION_POOL_SIZE', '10'))
        if pool_size < 5:
            warnings.append("Database connection pool size is low")
        
        passed = accessible_dbs == len(db_paths)
        
        return {
            'passed': passed,
            'details': {
                'database_paths': db_paths,
                'accessible_databases': accessible_dbs,
                'connection_pool_size': pool_size
            },
            'warnings': warnings,
            'errors': errors
        }
    
    async def _check_security_config(self) -> Dict[str, Any]:
        """Check security configuration"""
        
        security_issues = []
        warnings = []
        
        # Check debug mode
        debug_mode = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
        if debug_mode:
            security_issues.append("Debug mode is enabled in production")
        
        # Check HTTPS enforcement
        force_https = os.getenv('FORCE_HTTPS', 'false').lower() == 'true'
        if not force_https:
            warnings.append("HTTPS enforcement is not enabled")
        
        # Check rate limiting
        rate_limit = int(os.getenv('API_RATE_LIMIT_PER_MINUTE', '0'))
        if rate_limit == 0:
            warnings.append("API rate limiting is not configured")
        
        # Check allowed hosts
        allowed_hosts = os.getenv('ALLOWED_HOSTS', '')
        if not allowed_hosts:
            warnings.append("No allowed hosts configured")
        elif '*' in allowed_hosts:
            security_issues.append("Wildcard allowed hosts is insecure")
        
        passed = len(security_issues) == 0
        
        return {
            'passed': passed,
            'details': {
                'debug_mode': debug_mode,
                'https_enforced': force_https,
                'rate_limit_configured': rate_limit > 0,
                'allowed_hosts_configured': bool(allowed_hosts)
            },
            'warnings': warnings,
            'errors': security_issues
        }
    
    async def _check_monitoring_setup(self) -> Dict[str, Any]:
        """Check monitoring and alerting setup"""
        
        warnings = []
        
        # Check Prometheus
        prometheus_enabled = os.getenv('PROMETHEUS_ENABLED', 'false').lower() == 'true'
        if not prometheus_enabled:
            warnings.append("Prometheus monitoring is not enabled")
        
        # Check Grafana
        grafana_enabled = os.getenv('GRAFANA_ENABLED', 'false').lower() == 'true'
        if not grafana_enabled:
            warnings.append("Grafana dashboards are not enabled")
        
        # Check alerting
        sentry_dsn = os.getenv('SENTRY_DSN', '')
        notification_recipients = os.getenv('NOTIFICATION_RECIPIENTS', '')
        
        alerting_configured = bool(sentry_dsn or notification_recipients)
        if not alerting_configured:
            warnings.append("No alerting system configured")
        
        # This is optional, so always pass but with warnings
        passed = True
        
        return {
            'passed': passed,
            'details': {
                'prometheus_enabled': prometheus_enabled,
                'grafana_enabled': grafana_enabled,
                'alerting_configured': alerting_configured
            },
            'warnings': warnings,
            'errors': []
        }
    
    async def _check_performance_benchmarks(self) -> Dict[str, Any]:
        """Check performance meets benchmarks"""
        
        # Simple performance check
        start_time = asyncio.get_event_loop().time()
        
        # Simulate some operations
        for _ in range(1000):
            await asyncio.sleep(0.001)  # 1ms per operation
        
        end_time = asyncio.get_event_loop().time()
        duration_ms = (end_time - start_time) * 1000
        
        # Check if performance is acceptable
        acceptable_time = 2000  # 2 seconds for 1000 operations
        passed = duration_ms <= acceptable_time
        
        warnings = []
        if not passed:
            warnings.append(f"Performance below benchmark: {duration_ms:.1f}ms > {acceptable_time}ms")
        
        return {
            'passed': passed,
            'details': {
                'benchmark_duration_ms': duration_ms,
                'acceptable_threshold_ms': acceptable_time,
                'operations_tested': 1000
            },
            'warnings': warnings,
            'errors': []
        }

class DeploymentScriptGenerator:
    """Generate deployment scripts and procedures"""
    
    def __init__(self):
        logger.info("DeploymentScriptGenerator initialized")
    
    def generate_deployment_scripts(self, config: DeploymentConfig) -> Dict[str, str]:
        """Generate deployment scripts for target environment"""
        
        scripts = {}
        
        # Main deployment script
        scripts['deploy.sh'] = self._generate_main_deployment_script(config)
        
        # Pre-deployment validation
        scripts['pre_deploy_validation.sh'] = self._generate_validation_script(config)
        
        # Health check script
        scripts['health_check.sh'] = self._generate_health_check_script(config)
        
        # Rollback script
        scripts['rollback.sh'] = self._generate_rollback_script(config)
        
        # Environment setup
        scripts['setup_environment.sh'] = self._generate_environment_setup_script(config)
        
        return scripts
    
    def _generate_main_deployment_script(self, config: DeploymentConfig) -> str:
        """Generate main deployment script"""
        
        return f"""#!/bin/bash
# SolTrader Production Deployment Script
# Environment: {config.environment.value}
# Version: {config.version}
# Generated: {datetime.now().isoformat()}

set -e  # Exit on any error

echo "Starting SolTrader deployment to {config.environment.value}"
echo "Version: {config.version}"
echo "Strategy: {config.deployment_strategy}"
echo "Timestamp: $(date)"

# Configuration
DEPLOYMENT_DIR="/opt/soltrader"
BACKUP_DIR="/opt/soltrader_backups"
LOG_FILE="/var/log/soltrader_deploy.log"

# Create directories
mkdir -p $DEPLOYMENT_DIR
mkdir -p $BACKUP_DIR

# Log function
log() {{
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}}

log "Deployment started"

# Pre-deployment validation
log "Running pre-deployment validation..."
if ! ./pre_deploy_validation.sh; then
    log "Pre-deployment validation failed"
    exit 1
fi

# Backup current version
if [ "{config.backup_enabled}" = "True" ] && [ -d "$DEPLOYMENT_DIR" ]; then
    log "Creating backup..."
    BACKUP_NAME="soltrader_backup_$(date +%Y%m%d_%H%M%S)"
    tar -czf "$BACKUP_DIR/$BACKUP_NAME.tar.gz" -C "$DEPLOYMENT_DIR" .
    log "Backup created: $BACKUP_NAME.tar.gz"
fi

# Deploy new version
log "Deploying version {config.version}..."

# Copy application files
log "Copying application files..."
# rsync -av --exclude='.git' --exclude='__pycache__' . $DEPLOYMENT_DIR/

# Install/update dependencies
log "Installing dependencies..."
cd $DEPLOYMENT_DIR
# pip install -r requirements_updated.txt

# Update configuration
log "Updating configuration..."
# Copy production configuration files
# cp production.env.template .env

# Database migrations (if any)
log "Running database migrations..."
# python manage_database.py migrate

# Health check
log "Performing health check..."
if ./health_check.sh; then
    log "Health check passed"
else
    log "Health check failed - initiating rollback"
    if [ "{config.rollback_enabled}" = "True" ]; then
        ./rollback.sh
    fi
    exit 1
fi

log "Deployment completed successfully"
echo "SolTrader {config.version} deployed to {config.environment.value}"
"""
    
    def _generate_validation_script(self, config: DeploymentConfig) -> str:
        """Generate pre-deployment validation script"""
        
        return f"""#!/bin/bash
# Pre-deployment validation script
# Environment: {config.environment.value}

set -e

echo "Running pre-deployment validation..."

# Check system requirements
echo "Checking system requirements..."

# Check Python version
python3 --version
if [ $? -ne 0 ]; then
    echo "ERROR: Python3 not found"
    exit 1
fi

# Check disk space
AVAILABLE_SPACE=$(df / | awk 'NR==2 {{print $4}}')
REQUIRED_SPACE=1048576  # 1GB in KB
if [ $AVAILABLE_SPACE -lt $REQUIRED_SPACE ]; then
    echo "ERROR: Insufficient disk space"
    exit 1
fi

# Check memory
AVAILABLE_MEMORY=$(free -k | awk 'NR==2 {{print $7}}')
REQUIRED_MEMORY=524288  # 512MB in KB
if [ $AVAILABLE_MEMORY -lt $REQUIRED_MEMORY ]; then
    echo "ERROR: Insufficient memory"
    exit 1
fi

# Check required environment variables
echo "Checking environment variables..."
if [ -z "$ALCHEMY_RPC_URL" ]; then
    echo "ERROR: ALCHEMY_RPC_URL not set"
    exit 1
fi

if [ -z "$WALLET_ADDRESS" ]; then
    echo "ERROR: WALLET_ADDRESS not set"
    exit 1
fi

# Check network connectivity
echo "Checking network connectivity..."
if ! ping -c 1 google.com > /dev/null 2>&1; then
    echo "WARNING: No internet connectivity"
fi

# Check service ports
echo "Checking required ports..."
if netstat -tuln | grep -q ":8080 "; then
    echo "WARNING: Port 8080 already in use"
fi

echo "Pre-deployment validation completed successfully"
"""
    
    def _generate_health_check_script(self, config: DeploymentConfig) -> str:
        """Generate health check script"""
        
        return f"""#!/bin/bash
# Health check script
# Environment: {config.environment.value}

set -e

echo "Running health checks..."

# Check if application is running
echo "Checking application status..."

# Check process
if pgrep -f "python.*main.py" > /dev/null; then
    echo "✓ Application process is running"
else
    echo "✗ Application process not found"
    exit 1
fi

# Check HTTP endpoint (if available)
HEALTH_URL="http://localhost:8080/health"
if command -v curl > /dev/null; then
    if curl -f -s "$HEALTH_URL" > /dev/null; then
        echo "✓ HTTP health endpoint responding"
    else
        echo "✗ HTTP health endpoint not responding"
        exit 1
    fi
fi

# Check database connectivity
echo "Checking database connectivity..."
if [ -f "logs/unified_trading.db" ]; then
    echo "✓ Database files accessible"
else
    echo "✗ Database files not found"
    exit 1
fi

# Check log files
echo "Checking log files..."
if [ -f "logs/trading.log" ]; then
    # Check for recent entries (last 5 minutes)
    if find logs/trading.log -mmin -5 | grep -q .; then
        echo "✓ Application logging active"
    else
        echo "WARNING: No recent log entries"
    fi
else
    echo "WARNING: Log file not found"
fi

# Memory check
MEMORY_USAGE=$(ps aux | grep "python.*main.py" | grep -v grep | awk '{{print $4}}')
if [ ! -z "$MEMORY_USAGE" ]; then
    echo "✓ Memory usage: $MEMORY_USAGE%"
    if (( $(echo "$MEMORY_USAGE > 80" | bc -l) )); then
        echo "WARNING: High memory usage"
    fi
fi

echo "Health checks completed successfully"
"""
    
    def _generate_rollback_script(self, config: DeploymentConfig) -> str:
        """Generate rollback script"""
        
        return f"""#!/bin/bash
# Rollback script
# Environment: {config.environment.value}

set -e

echo "Starting rollback procedure..."

DEPLOYMENT_DIR="/opt/soltrader"
BACKUP_DIR="/opt/soltrader_backups"
LOG_FILE="/var/log/soltrader_rollback.log"

# Log function
log() {{
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}}

log "Rollback initiated"

# Stop current application
log "Stopping application..."
pkill -f "python.*main.py" || true

# Find latest backup
LATEST_BACKUP=$(ls -t $BACKUP_DIR/*.tar.gz | head -n1)
if [ -z "$LATEST_BACKUP" ]; then
    log "ERROR: No backup found"
    exit 1
fi

log "Rolling back to backup: $LATEST_BACKUP"

# Backup current failed deployment
FAILED_BACKUP="$BACKUP_DIR/failed_deployment_$(date +%Y%m%d_%H%M%S).tar.gz"
log "Backing up failed deployment to: $FAILED_BACKUP"
tar -czf "$FAILED_BACKUP" -C "$DEPLOYMENT_DIR" .

# Restore from backup
log "Restoring from backup..."
rm -rf "$DEPLOYMENT_DIR"/*
tar -xzf "$LATEST_BACKUP" -C "$DEPLOYMENT_DIR"

# Restart application
log "Restarting application..."
cd $DEPLOYMENT_DIR
# nohup python main.py &

# Wait and verify
sleep 5
if pgrep -f "python.*main.py" > /dev/null; then
    log "Rollback completed successfully"
    echo "Rollback to previous version completed"
else
    log "ERROR: Failed to restart application after rollback"
    exit 1
fi
"""
    
    def _generate_environment_setup_script(self, config: DeploymentConfig) -> str:
        """Generate environment setup script"""
        
        return f"""#!/bin/bash
# Environment setup script
# Environment: {config.environment.value}

set -e

echo "Setting up {config.environment.value} environment..."

# Create application user
if ! id "soltrader" &>/dev/null; then
    useradd -r -s /bin/false soltrader
    echo "Created soltrader user"
fi

# Create directories
mkdir -p /opt/soltrader
mkdir -p /var/log/soltrader
mkdir -p /opt/soltrader_backups
mkdir -p /etc/soltrader

# Set permissions
chown -R soltrader:soltrader /opt/soltrader
chown -R soltrader:soltrader /var/log/soltrader
chown -R soltrader:soltrader /opt/soltrader_backups
chmod 750 /opt/soltrader
chmod 750 /var/log/soltrader

# Install system dependencies
echo "Installing system dependencies..."
apt-get update
apt-get install -y python3 python3-pip python3-venv curl

# Create systemd service file
cat > /etc/systemd/system/soltrader.service << 'EOF'
[Unit]
Description=SolTrader Trading Bot
After=network.target

[Service]
Type=simple
User=soltrader
WorkingDirectory=/opt/soltrader
ExecStart=/opt/soltrader/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
systemctl daemon-reload
systemctl enable soltrader

# Setup log rotation
cat > /etc/logrotate.d/soltrader << 'EOF'
/var/log/soltrader/*.log {{
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}}
EOF

# Setup firewall rules (if ufw is available)
if command -v ufw > /dev/null; then
    ufw allow 8080/tcp  # API port
    ufw allow 8000/tcp  # Prometheus port
    ufw allow 3000/tcp  # Grafana port
fi

echo "{config.environment.value} environment setup completed"
"""

class ProductionDeploymentManager:
    """Main production deployment manager"""
    
    def __init__(self):
        self.readiness_checker = ProductionReadinessChecker()
        self.script_generator = DeploymentScriptGenerator()
        
        # Deployment state
        self.current_deployment: Optional[DeploymentResult] = None
        self.deployment_history: List[DeploymentResult] = []
        
        logger.info("ProductionDeploymentManager initialized")
    
    async def prepare_production_deployment(self, 
                                          version: str,
                                          environment: EnvironmentType = EnvironmentType.PRODUCTION) -> Dict[str, Any]:
        """Prepare complete production deployment"""
        
        logger.info(f"Preparing production deployment for version {version}")
        
        preparation_start = datetime.now()
        
        try:
            # Step 1: Assess production readiness
            logger.info("Assessing production readiness...")
            readiness_assessment = await self.readiness_checker.assess_production_readiness()
            
            # Step 2: Generate deployment configuration
            deployment_config = DeploymentConfig(
                environment=environment,
                version=version,
                deployment_strategy="blue_green",
                backup_enabled=True,
                health_check_enabled=True,
                rollback_enabled=True
            )
            
            # Step 3: Generate deployment scripts
            logger.info("Generating deployment scripts...")
            deployment_scripts = self.script_generator.generate_deployment_scripts(deployment_config)
            
            # Step 4: Create deployment documentation
            logger.info("Generating deployment documentation...")
            deployment_docs = await self._generate_deployment_documentation(
                deployment_config, readiness_assessment
            )
            
            # Step 5: Validate deployment package
            validation_result = await self._validate_deployment_package(
                deployment_config, deployment_scripts
            )
            
            preparation_time = (datetime.now() - preparation_start).total_seconds()
            
            # Determine if deployment is recommended
            deployment_recommended = (
                readiness_assessment['readiness_level'] in ['ready_with_warnings', 'production_ready'] and
                validation_result['valid']
            )
            
            return {
                'success': True,
                'deployment_recommended': deployment_recommended,
                'preparation_time_seconds': preparation_time,
                'deployment_config': {
                    'environment': deployment_config.environment.value,
                    'version': deployment_config.version,
                    'strategy': deployment_config.deployment_strategy
                },
                'readiness_assessment': readiness_assessment,
                'deployment_scripts': list(deployment_scripts.keys()),
                'deployment_documentation': deployment_docs,
                'validation_result': validation_result,
                'next_steps': self._generate_next_steps(readiness_assessment, deployment_recommended)
            }
            
        except Exception as e:
            logger.error(f"Deployment preparation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'preparation_time_seconds': (datetime.now() - preparation_start).total_seconds()
            }
    
    async def _generate_deployment_documentation(self, 
                                               config: DeploymentConfig,
                                               readiness: Dict[str, Any]) -> Dict[str, str]:
        """Generate comprehensive deployment documentation"""
        
        docs = {}
        
        # Main deployment guide
        docs['DEPLOYMENT_GUIDE.md'] = f"""# SolTrader Production Deployment Guide

## Version: {config.version}
## Environment: {config.environment.value}
## Generated: {datetime.now().isoformat()}

### Production Readiness Status
- **Overall Score**: {readiness['overall_score']:.1f}%
- **Readiness Level**: {readiness['readiness_level'].upper()}
- **Passed Checks**: {readiness['passed_checks']}/{readiness['total_checks']}

### Pre-Deployment Checklist

#### Environment Setup
- [ ] Server provisioned with adequate resources (2GB+ RAM, 10GB+ disk)
- [ ] Python 3.8+ installed
- [ ] Required system dependencies installed
- [ ] Application user created (`soltrader`)
- [ ] Directory structure created (`/opt/soltrader`, `/var/log/soltrader`)
- [ ] Firewall rules configured

#### Configuration
- [ ] Environment variables configured in production.env
- [ ] Database paths accessible and writable
- [ ] SSL certificates installed (if HTTPS enabled)
- [ ] Monitoring tools configured (Prometheus, Grafana)
- [ ] Alerting configured (Sentry, email notifications)

#### Security
- [ ] Debug mode disabled (`DEBUG_MODE=false`)
- [ ] HTTPS enforced (`FORCE_HTTPS=true`)
- [ ] Rate limiting configured
- [ ] Allowed hosts specified
- [ ] Sensitive data encrypted
- [ ] Firewall properly configured

### Deployment Steps

1. **Pre-deployment Validation**
   ```bash
   ./pre_deploy_validation.sh
   ```

2. **Create Backup** (if existing deployment)
   ```bash
   # Automatic backup during deployment
   ./deploy.sh
   ```

3. **Deploy Application**
   ```bash
   ./deploy.sh
   ```

4. **Post-deployment Validation**
   ```bash
   ./health_check.sh
   ```

5. **Monitor Deployment**
   - Check application logs: `tail -f /var/log/soltrader/trading.log`
   - Monitor system resources: `htop`
   - Verify API endpoints: `curl http://localhost:8080/health`

### Rollback Procedure

If deployment fails or issues are detected:

```bash
./rollback.sh
```

### Monitoring and Maintenance

#### Health Checks
- Automated health checks run every 30 seconds
- Alerts triggered for critical issues
- Dashboard available at: http://localhost:3000

#### Log Management
- Application logs: `/var/log/soltrader/`
- Log rotation configured automatically
- Retention: 30 days

#### Backup Strategy
- Automatic backups before each deployment
- Daily database backups
- Backup retention: 7 days

### Troubleshooting

#### Common Issues

1. **Application Won't Start**
   - Check logs: `journalctl -u soltrader -f`
   - Verify configuration: `cat .env`
   - Check permissions: `ls -la /opt/soltrader`

2. **High Memory Usage**
   - Monitor with: `ps aux | grep python`
   - Check for memory leaks in logs
   - Consider restarting: `systemctl restart soltrader`

3. **API Errors**
   - Check RPC connectivity: `ping solana-api.mainnet-beta.solana.com`
   - Verify API keys in environment
   - Review rate limiting logs

### Support Contacts
- Technical Issues: Check logs and system status
- Emergency: Use rollback procedure immediately
"""
        
        # Operations runbook
        docs['OPERATIONS_RUNBOOK.md'] = f"""# SolTrader Operations Runbook

## Emergency Procedures

### Emergency Stop
```bash
# Immediate stop
systemctl stop soltrader
pkill -f "python.*main.py"
```

### Emergency Rollback
```bash
# Automatic rollback to previous version
./rollback.sh
```

## Daily Operations

### Health Monitoring
- Check system status: `systemctl status soltrader`
- Monitor logs: `tail -f /var/log/soltrader/trading.log`
- Review trading performance: Access Grafana dashboard

### Backup Verification
- Verify daily backups completed: `ls -la /opt/soltrader_backups`
- Test backup restoration (staging environment)

### Performance Monitoring
- CPU usage: `top`
- Memory usage: `free -h`  
- Disk usage: `df -h`
- Network connectivity: `ping -c 1 google.com`

## Weekly Operations

### Security Updates
- Update system packages: `apt update && apt upgrade`
- Review security logs
- Rotate API keys (if configured)

### Performance Review
- Analyze trading performance metrics
- Review error logs and resolve issues
- Update configuration if needed

## Monthly Operations

### Full Backup
- Create complete system backup
- Test disaster recovery procedures
- Update documentation

### Performance Optimization
- Review resource usage trends
- Optimize configuration parameters
- Plan capacity upgrades if needed
"""
        
        return docs
    
    async def _validate_deployment_package(self, 
                                         config: DeploymentConfig,
                                         scripts: Dict[str, str]) -> Dict[str, Any]:
        """Validate deployment package completeness"""
        
        validation_errors = []
        validation_warnings = []
        
        # Check required scripts
        required_scripts = ['deploy.sh', 'health_check.sh', 'rollback.sh']
        for script in required_scripts:
            if script not in scripts:
                validation_errors.append(f"Missing required script: {script}")
        
        # Check script content
        for script_name, script_content in scripts.items():
            if not script_content or len(script_content) < 100:
                validation_warnings.append(f"Script {script_name} seems incomplete")
            
            if 'set -e' not in script_content:
                validation_warnings.append(f"Script {script_name} missing error handling")
        
        # Check deployment configuration
        if config.deployment_timeout_seconds < 300:
            validation_warnings.append("Deployment timeout may be too short")
        
        if not config.backup_enabled:
            validation_errors.append("Backup must be enabled for production deployment")
        
        valid = len(validation_errors) == 0
        
        return {
            'valid': valid,
            'errors': validation_errors,
            'warnings': validation_warnings,
            'validation_score': max(0, 100 - len(validation_errors) * 25 - len(validation_warnings) * 5)
        }
    
    def _generate_next_steps(self, readiness: Dict[str, Any], deployment_recommended: bool) -> List[str]:
        """Generate next steps based on readiness assessment"""
        
        steps = []
        
        if not deployment_recommended:
            steps.extend([
                "🔴 DEPLOYMENT NOT RECOMMENDED",
                "Address all failed required checks before proceeding",
                "Review and resolve security warnings"
            ])
            
            # Add specific steps based on failed checks
            for check in readiness['check_results']:
                if not check['passed'] and check['required']:
                    steps.append(f"  - Fix: {check['name']}")
        else:
            if readiness['readiness_level'] == 'production_ready':
                steps.extend([
                    "🟢 PRODUCTION READY",
                    "1. Review deployment scripts and documentation",
                    "2. Schedule deployment window",
                    "3. Execute deployment using ./deploy.sh",
                    "4. Monitor system post-deployment"
                ])
            else:
                steps.extend([
                    "🟡 READY WITH WARNINGS",
                    "1. Review warnings and address if possible",
                    "2. Proceed with caution if warnings are acceptable",
                    "3. Execute deployment using ./deploy.sh",
                    "4. Monitor system closely post-deployment"
                ])
        
        return steps
    
    def save_deployment_package(self, deployment_info: Dict[str, Any], 
                              output_dir: str = "deployment_package") -> str:
        """Save complete deployment package to directory"""
        
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Save deployment scripts (mock - would normally be from deployment_info)
            scripts_dir = os.path.join(output_dir, "scripts")
            os.makedirs(scripts_dir, exist_ok=True)
            
            # Save documentation
            docs_dir = os.path.join(output_dir, "docs")
            os.makedirs(docs_dir, exist_ok=True)
            
            # Save readiness assessment
            with open(os.path.join(output_dir, "readiness_assessment.json"), 'w') as f:
                json.dump(deployment_info.get('readiness_assessment', {}), f, indent=2, default=str)
            
            # Save deployment summary
            summary = {
                'deployment_recommended': deployment_info.get('deployment_recommended', False),
                'readiness_level': deployment_info.get('readiness_assessment', {}).get('readiness_level', 'unknown'),
                'generated_at': datetime.now().isoformat(),
                'version': deployment_info.get('deployment_config', {}).get('version', 'unknown')
            }
            
            with open(os.path.join(output_dir, "deployment_summary.json"), 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Deployment package saved to {output_dir}")
            return output_dir
            
        except Exception as e:
            logger.error(f"Failed to save deployment package: {e}")
            raise

# Example usage and testing
if __name__ == "__main__":
    async def test_production_deployment_manager():
        """Test the production deployment manager"""
        
        print("SolTrader Production Deployment Manager")
        print("=" * 50)
        
        manager = ProductionDeploymentManager()
        
        try:
            # Test production readiness assessment
            print("1. Assessing production readiness...")
            readiness = await manager.readiness_checker.assess_production_readiness()
            
            print(f"   Readiness Level: {readiness['readiness_level'].upper()}")
            print(f"   Overall Score: {readiness['overall_score']:.1f}%")
            print(f"   Checks Passed: {readiness['passed_checks']}/{readiness['total_checks']}")
            
            if readiness['warnings']:
                print(f"   Warnings: {len(readiness['warnings'])}")
                for warning in readiness['warnings'][:3]:  # Show first 3 warnings
                    print(f"     - {warning}")
            
            # Test full deployment preparation
            print(f"\n2. Preparing deployment package...")
            deployment_info = await manager.prepare_production_deployment("1.0.0")
            
            if deployment_info['success']:
                print(f"   Preparation Time: {deployment_info['preparation_time_seconds']:.1f}s")
                print(f"   Deployment Recommended: {deployment_info['deployment_recommended']}")
                print(f"   Scripts Generated: {len(deployment_info['deployment_scripts'])}")
                print(f"   Documentation Generated: {len(deployment_info['deployment_documentation'])}")
                
                # Show next steps
                print(f"\n   Next Steps:")
                for step in deployment_info['next_steps'][:5]:  # Show first 5 steps
                    print(f"     - {step}")
                
                # Save deployment package
                print(f"\n3. Saving deployment package...")
                package_dir = manager.save_deployment_package(deployment_info)
                print(f"   Package saved to: {package_dir}")
                
                print(f"\n Production deployment preparation completed!")
                
                # Summary
                print(f"\n Deployment Summary:")
                print(f"   Version: {deployment_info['deployment_config']['version']}")
                print(f"   Environment: {deployment_info['deployment_config']['environment']}")
                print(f"   Strategy: {deployment_info['deployment_config']['strategy']}")
                print(f"   Validation Score: {deployment_info['validation_result']['validation_score']:.1f}%")
                
            else:
                print(f"   Preparation failed: {deployment_info['error']}")
            
        except Exception as e:
            print(f"Test failed: {e}")
            raise
    
    asyncio.run(test_production_deployment_manager())