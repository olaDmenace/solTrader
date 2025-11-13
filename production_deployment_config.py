#!/usr/bin/env python3
"""
Production Deployment Configuration
===================================

Production-ready configuration management with:
- Environment variable validation and loading
- Configuration encryption and security
- Hot-reloading configuration updates
- Production deployment scripts
- Resource monitoring and alerting

Integration with production.env.template for complete deployment setup.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import hashlib
import hmac
import base64
from cryptography.fernet import Fernet
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration"""
    connection_pool_size: int = 10
    connection_timeout: int = 30
    query_timeout: int = 10
    batch_size: int = 100
    checkpoint_interval: int = 1000
    cache_size_pages: int = 2000
    journal_mode: str = "WAL"
    synchronous: str = "NORMAL"

@dataclass
class CachingConfig:
    """Caching configuration"""
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_max_connections: int = 20
    default_cache_ttl: int = 300
    token_data_cache_ttl: int = 60
    position_cache_ttl: int = 30
    price_cache_ttl: int = 15
    enable_memory_cache: bool = True
    memory_cache_size_mb: int = 100

@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    prometheus_enabled: bool = True
    prometheus_port: int = 8000
    prometheus_metrics_path: str = "/metrics"
    grafana_enabled: bool = True
    grafana_port: int = 3000
    health_check_interval_seconds: int = 30
    health_check_timeout_seconds: int = 5
    enable_performance_monitoring: bool = True
    performance_sample_interval_seconds: int = 60
    memory_usage_alert_threshold_mb: int = 1500
    cpu_usage_alert_threshold_percent: int = 80

@dataclass 
class AlertingConfig:
    """Alerting configuration"""
    sentry_dsn: Optional[str] = None
    sentry_environment: str = "production"
    sentry_sample_rate: float = 1.0
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    notification_recipients: List[str] = field(default_factory=list)
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    emergency_alert_loss_threshold: float = 100.0
    high_priority_alert_loss_threshold: float = 50.0
    position_size_alert_threshold: float = 0.20

@dataclass
class SecurityConfig:
    """Security configuration"""
    master_encryption_key: str = ""
    api_rate_limit_per_minute: int = 1000
    api_timeout_seconds: int = 30
    enable_api_key_rotation: bool = True
    api_key_rotation_hours: int = 24
    allowed_hosts: List[str] = field(default_factory=lambda: ["localhost", "127.0.0.1"])
    enable_cors: bool = False
    cors_allowed_origins: List[str] = field(default_factory=list)

@dataclass
class TradingConfig:
    """Trading configuration"""
    paper_trading: bool = False
    live_trading: bool = True
    enable_emergency_stop: bool = True
    initial_capital: float = 1000.0
    max_position_size: float = 0.25
    max_daily_loss: float = 50.0
    max_portfolio_risk: float = 0.15
    max_drawdown: float = 0.20
    emergency_liquidation_threshold: float = 0.15
    max_trades_per_day: int = 100
    max_trades_per_strategy: int = 25
    min_trade_interval_seconds: int = 30
    max_concurrent_trades: int = 10

@dataclass
class ProductionConfig:
    """Complete production configuration"""
    environment: str = "production"
    debug_mode: bool = False
    log_level: str = "INFO"
    service_name: str = "soltrader"
    service_port: int = 8080
    service_host: str = "0.0.0.0"
    deployment_version: str = "1.0.0"
    
    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    caching: CachingConfig = field(default_factory=CachingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    alerting: AlertingConfig = field(default_factory=AlertingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)

class ConfigurationManager:
    """Production configuration management system"""
    
    def __init__(self, config_file: str = "production.env"):
        self.config_file = config_file
        self.config: Optional[ProductionConfig] = None
        self.encryption_key: Optional[bytes] = None
        self.config_hash: Optional[str] = None
        
        # Configuration validation rules
        self.required_env_vars = [
            'ALCHEMY_RPC_URL', 'WALLET_ADDRESS', 'MASTER_ENCRYPTION_KEY'
        ]
        
        logger.info(f"ConfigurationManager initialized for {config_file}")
    
    def load_configuration(self) -> ProductionConfig:
        """Load and validate production configuration"""
        
        try:
            # Load environment variables
            self._load_env_file()
            
            # Validate required variables
            self._validate_required_vars()
            
            # Create configuration object
            config = self._build_configuration()
            
            # Validate configuration
            self._validate_configuration(config)
            
            # Setup encryption
            self._setup_encryption(config.security.master_encryption_key)
            
            # Calculate configuration hash for change detection
            self.config_hash = self._calculate_config_hash(config)
            
            self.config = config
            logger.info("Production configuration loaded successfully")
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _load_env_file(self):
        """Load environment variables from file"""
        
        if not os.path.exists(self.config_file):
            logger.warning(f"Configuration file {self.config_file} not found, using environment variables only")
            return
        
        try:
            with open(self.config_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        # Only set if not already in environment
                        if key.strip() not in os.environ:
                            os.environ[key.strip()] = value.strip()
            
            logger.info(f"Loaded configuration from {self.config_file}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration file {self.config_file}: {e}")
            raise
    
    def _validate_required_vars(self):
        """Validate required environment variables"""
        
        missing_vars = []
        
        for var in self.required_env_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Required environment variables missing: {', '.join(missing_vars)}")
        
        logger.info("All required environment variables present")
    
    def _build_configuration(self) -> ProductionConfig:
        """Build configuration object from environment variables"""
        
        # Database configuration
        database = DatabaseConfig(
            connection_pool_size=int(os.getenv('DB_CONNECTION_POOL_SIZE', 10)),
            connection_timeout=int(os.getenv('DB_CONNECTION_TIMEOUT', 30)),
            query_timeout=int(os.getenv('DB_QUERY_TIMEOUT', 10)),
            batch_size=int(os.getenv('DB_BATCH_SIZE', 100)),
            checkpoint_interval=int(os.getenv('DB_CHECKPOINT_INTERVAL', 1000)),
            cache_size_pages=int(os.getenv('DB_CACHE_SIZE_PAGES', 2000)),
            journal_mode=os.getenv('DB_JOURNAL_MODE', 'WAL'),
            synchronous=os.getenv('DB_SYNCHRONOUS', 'NORMAL')
        )
        
        # Caching configuration
        caching = CachingConfig(
            redis_host=os.getenv('REDIS_HOST', 'localhost'),
            redis_port=int(os.getenv('REDIS_PORT', 6379)),
            redis_db=int(os.getenv('REDIS_DB', 0)),
            redis_max_connections=int(os.getenv('REDIS_MAX_CONNECTIONS', 20)),
            default_cache_ttl=int(os.getenv('DEFAULT_CACHE_TTL', 300)),
            token_data_cache_ttl=int(os.getenv('TOKEN_DATA_CACHE_TTL', 60)),
            position_cache_ttl=int(os.getenv('POSITION_CACHE_TTL', 30)),
            price_cache_ttl=int(os.getenv('PRICE_CACHE_TTL', 15)),
            enable_memory_cache=os.getenv('ENABLE_MEMORY_CACHE', 'true').lower() == 'true',
            memory_cache_size_mb=int(os.getenv('MEMORY_CACHE_SIZE_MB', 100))
        )
        
        # Monitoring configuration
        monitoring = MonitoringConfig(
            prometheus_enabled=os.getenv('PROMETHEUS_ENABLED', 'true').lower() == 'true',
            prometheus_port=int(os.getenv('PROMETHEUS_PORT', 8000)),
            prometheus_metrics_path=os.getenv('PROMETHEUS_METRICS_PATH', '/metrics'),
            grafana_enabled=os.getenv('GRAFANA_ENABLED', 'true').lower() == 'true',
            grafana_port=int(os.getenv('GRAFANA_PORT', 3000)),
            health_check_interval_seconds=int(os.getenv('HEALTH_CHECK_INTERVAL_SECONDS', 30)),
            health_check_timeout_seconds=int(os.getenv('HEALTH_CHECK_TIMEOUT_SECONDS', 5)),
            enable_performance_monitoring=os.getenv('ENABLE_PERFORMANCE_MONITORING', 'true').lower() == 'true',
            performance_sample_interval_seconds=int(os.getenv('PERFORMANCE_SAMPLE_INTERVAL_SECONDS', 60)),
            memory_usage_alert_threshold_mb=int(os.getenv('MEMORY_USAGE_ALERT_THRESHOLD_MB', 1500)),
            cpu_usage_alert_threshold_percent=int(os.getenv('CPU_USAGE_ALERT_THRESHOLD_PERCENT', 80))
        )
        
        # Alerting configuration
        alerting = AlertingConfig(
            sentry_dsn=os.getenv('SENTRY_DSN'),
            sentry_environment=os.getenv('SENTRY_ENVIRONMENT', 'production'),
            sentry_sample_rate=float(os.getenv('SENTRY_SAMPLE_RATE', 1.0)),
            smtp_host=os.getenv('SMTP_HOST', 'smtp.gmail.com'),
            smtp_port=int(os.getenv('SMTP_PORT', 587)),
            notification_recipients=os.getenv('NOTIFICATION_RECIPIENTS', '').split(',') if os.getenv('NOTIFICATION_RECIPIENTS') else [],
            telegram_bot_token=os.getenv('TELEGRAM_BOT_TOKEN'),
            telegram_chat_id=os.getenv('TELEGRAM_CHAT_ID'),
            emergency_alert_loss_threshold=float(os.getenv('EMERGENCY_ALERT_LOSS_THRESHOLD', 100.0)),
            high_priority_alert_loss_threshold=float(os.getenv('HIGH_PRIORITY_ALERT_LOSS_THRESHOLD', 50.0)),
            position_size_alert_threshold=float(os.getenv('POSITION_SIZE_ALERT_THRESHOLD', 0.20))
        )
        
        # Security configuration
        security = SecurityConfig(
            master_encryption_key=os.getenv('MASTER_ENCRYPTION_KEY', ''),
            api_rate_limit_per_minute=int(os.getenv('API_RATE_LIMIT_PER_MINUTE', 1000)),
            api_timeout_seconds=int(os.getenv('API_TIMEOUT_SECONDS', 30)),
            enable_api_key_rotation=os.getenv('ENABLE_API_KEY_ROTATION', 'true').lower() == 'true',
            api_key_rotation_hours=int(os.getenv('API_KEY_ROTATION_HOURS', 24)),
            allowed_hosts=os.getenv('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(','),
            enable_cors=os.getenv('ENABLE_CORS', 'false').lower() == 'true',
            cors_allowed_origins=os.getenv('CORS_ALLOWED_ORIGINS', '').split(',') if os.getenv('CORS_ALLOWED_ORIGINS') else []
        )
        
        # Trading configuration
        trading = TradingConfig(
            paper_trading=os.getenv('PAPER_TRADING', 'false').lower() == 'true',
            live_trading=os.getenv('LIVE_TRADING', 'true').lower() == 'true',
            enable_emergency_stop=os.getenv('ENABLE_EMERGENCY_STOP', 'true').lower() == 'true',
            initial_capital=float(os.getenv('INITIAL_CAPITAL', 1000.0)),
            max_position_size=float(os.getenv('MAX_POSITION_SIZE', 0.25)),
            max_daily_loss=float(os.getenv('MAX_DAILY_LOSS', 50.0)),
            max_portfolio_risk=float(os.getenv('MAX_PORTFOLIO_RISK', 0.15)),
            max_drawdown=float(os.getenv('MAX_DRAWDOWN', 0.20)),
            emergency_liquidation_threshold=float(os.getenv('EMERGENCY_LIQUIDATION_THRESHOLD', 0.15)),
            max_trades_per_day=int(os.getenv('MAX_TRADES_PER_DAY', 100)),
            max_trades_per_strategy=int(os.getenv('MAX_TRADES_PER_STRATEGY', 25)),
            min_trade_interval_seconds=int(os.getenv('MIN_TRADE_INTERVAL_SECONDS', 30)),
            max_concurrent_trades=int(os.getenv('MAX_CONCURRENT_TRADES', 10))
        )
        
        # Main configuration
        return ProductionConfig(
            environment=os.getenv('NODE_ENV', 'production'),
            debug_mode=os.getenv('DEBUG_MODE', 'false').lower() == 'true',
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            service_name=os.getenv('SERVICE_NAME', 'soltrader'),
            service_port=int(os.getenv('SERVICE_PORT', 8080)),
            service_host=os.getenv('SERVICE_HOST', '0.0.0.0'),
            deployment_version=os.getenv('DEPLOYMENT_VERSION', '1.0.0'),
            database=database,
            caching=caching,
            monitoring=monitoring,
            alerting=alerting,
            security=security,
            trading=trading
        )
    
    def _validate_configuration(self, config: ProductionConfig):
        """Validate configuration values"""
        
        validation_errors = []
        
        # Validate trading parameters
        if config.trading.max_position_size > 0.5:
            validation_errors.append("Max position size too high (maximum 50%)")
        
        if config.trading.max_daily_loss <= 0:
            validation_errors.append("Max daily loss must be positive")
        
        if config.trading.initial_capital <= 0:
            validation_errors.append("Initial capital must be positive")
        
        # Validate security parameters
        if len(config.security.master_encryption_key) < 32:
            validation_errors.append("Master encryption key too short (minimum 32 characters)")
        
        if config.security.api_rate_limit_per_minute > 10000:
            validation_errors.append("API rate limit too high (maximum 10000/minute)")
        
        # Validate monitoring parameters
        if config.monitoring.health_check_interval_seconds < 10:
            validation_errors.append("Health check interval too short (minimum 10 seconds)")
        
        if config.monitoring.memory_usage_alert_threshold_mb < 512:
            validation_errors.append("Memory alert threshold too low (minimum 512MB)")
        
        # Validate database parameters
        if config.database.connection_pool_size > 100:
            validation_errors.append("Database connection pool too large (maximum 100)")
        
        if validation_errors:
            raise ValueError(f"Configuration validation failed:\n{chr(10).join(validation_errors)}")
        
        logger.info("Configuration validation passed")
    
    def _setup_encryption(self, master_key: str):
        """Setup configuration encryption"""
        
        try:
            # Generate or derive encryption key from master key
            key_material = hashlib.sha256(master_key.encode()).digest()
            self.encryption_key = base64.urlsafe_b64encode(key_material)
            
            logger.info("Configuration encryption initialized")
            
        except Exception as e:
            logger.error(f"Failed to setup encryption: {e}")
            raise
    
    def _calculate_config_hash(self, config: ProductionConfig) -> str:
        """Calculate configuration hash for change detection"""
        
        try:
            config_json = json.dumps(asdict(config), sort_keys=True)
            return hashlib.sha256(config_json.encode()).hexdigest()
            
        except Exception as e:
            logger.warning(f"Failed to calculate config hash: {e}")
            return ""
    
    def encrypt_sensitive_value(self, value: str) -> str:
        """Encrypt sensitive configuration value"""
        
        if not self.encryption_key:
            raise ValueError("Encryption not initialized")
        
        try:
            fernet = Fernet(self.encryption_key)
            encrypted = fernet.encrypt(value.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
            
        except Exception as e:
            logger.error(f"Failed to encrypt value: {e}")
            raise
    
    def decrypt_sensitive_value(self, encrypted_value: str) -> str:
        """Decrypt sensitive configuration value"""
        
        if not self.encryption_key:
            raise ValueError("Encryption not initialized")
        
        try:
            fernet = Fernet(self.encryption_key)
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode())
            decrypted = fernet.decrypt(encrypted_bytes)
            return decrypted.decode()
            
        except Exception as e:
            logger.error(f"Failed to decrypt value: {e}")
            raise
    
    def reload_configuration(self) -> bool:
        """Reload configuration if changes detected"""
        
        try:
            # Reload environment file
            self._load_env_file()
            
            # Build new configuration
            new_config = self._build_configuration()
            
            # Check if configuration changed
            new_hash = self._calculate_config_hash(new_config)
            
            if new_hash != self.config_hash:
                # Validate new configuration
                self._validate_configuration(new_config)
                
                # Update configuration
                old_config = self.config
                self.config = new_config
                self.config_hash = new_hash
                
                logger.info("Configuration reloaded successfully")
                return True
            else:
                logger.debug("Configuration unchanged")
                return False
                
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            raise
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get configuration summary for monitoring"""
        
        if not self.config:
            return {'error': 'Configuration not loaded'}
        
        return {
            'timestamp': datetime.now().isoformat(),
            'environment': self.config.environment,
            'service_name': self.config.service_name,
            'deployment_version': self.config.deployment_version,
            'config_hash': self.config_hash,
            'debug_mode': self.config.debug_mode,
            'trading_mode': 'paper' if self.config.trading.paper_trading else 'live',
            'monitoring_enabled': self.config.monitoring.prometheus_enabled,
            'alerting_configured': bool(self.config.alerting.sentry_dsn or self.config.alerting.notification_recipients),
            'encryption_enabled': self.encryption_key is not None,
            'component_status': {
                'database': 'configured',
                'caching': 'redis' if self.config.caching.redis_host != 'localhost' else 'memory',
                'monitoring': 'enabled' if self.config.monitoring.prometheus_enabled else 'disabled',
                'security': 'enabled'
            }
        }
    
    def export_configuration(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Export configuration for backup or analysis"""
        
        if not self.config:
            raise ValueError("Configuration not loaded")
        
        config_dict = asdict(self.config)
        
        if not include_sensitive:
            # Remove sensitive fields
            sensitive_fields = [
                'master_encryption_key', 'sentry_dsn', 'smtp_password',
                'telegram_bot_token', 'redis_password'
            ]
            
            def remove_sensitive(obj, parent_key=''):
                if isinstance(obj, dict):
                    return {
                        k: remove_sensitive(v, f"{parent_key}.{k}" if parent_key else k)
                        for k, v in obj.items()
                        if not any(sensitive in k.lower() for sensitive in ['password', 'key', 'token', 'secret'])
                    }
                else:
                    return obj
            
            config_dict = remove_sensitive(config_dict)
        
        return {
            'export_timestamp': datetime.now().isoformat(),
            'config_hash': self.config_hash,
            'include_sensitive': include_sensitive,
            'configuration': config_dict
        }

# Factory functions
def create_production_config_manager(config_file: str = "production.env") -> ConfigurationManager:
    """Create production configuration manager"""
    
    manager = ConfigurationManager(config_file)
    config = manager.load_configuration()
    
    return manager

def validate_production_environment() -> Dict[str, Any]:
    """Validate production environment readiness"""
    
    validation_results = {
        'timestamp': datetime.now().isoformat(),
        'environment_ready': True,
        'issues': [],
        'warnings': [],
        'recommendations': []
    }
    
    try:
        # Try to load configuration
        manager = create_production_config_manager()
        config = manager.config
        
        # Check critical components
        if config.trading.paper_trading:
            validation_results['warnings'].append("Paper trading is enabled - verify this is intentional for production")
        
        if config.debug_mode:
            validation_results['warnings'].append("Debug mode is enabled - should be disabled in production")
        
        if not config.monitoring.prometheus_enabled:
            validation_results['issues'].append("Prometheus monitoring is disabled")
            validation_results['environment_ready'] = False
        
        if not config.alerting.sentry_dsn and not config.alerting.notification_recipients:
            validation_results['warnings'].append("No alerting configured - errors will not be reported")
        
        if config.trading.max_position_size > 0.3:
            validation_results['warnings'].append(f"Max position size is high ({config.trading.max_position_size*100}%)")
        
        # Add recommendations
        if config.caching.redis_host == "localhost":
            validation_results['recommendations'].append("Consider using dedicated Redis instance for production")
        
        if config.database.connection_pool_size < 5:
            validation_results['recommendations'].append("Consider increasing database connection pool size")
        
        logger.info(f"Production environment validation: {'READY' if validation_results['environment_ready'] else 'NOT READY'}")
        
    except Exception as e:
        validation_results['environment_ready'] = False
        validation_results['issues'].append(f"Configuration loading failed: {str(e)}")
        logger.error(f"Production environment validation failed: {e}")
    
    return validation_results

# Example usage and testing
if __name__ == "__main__":
    # Set test environment variables
    os.environ['ALCHEMY_RPC_URL'] = 'https://solana-mainnet.g.alchemy.com/v2/test'
    os.environ['WALLET_ADDRESS'] = 'test_wallet_address'
    os.environ['MASTER_ENCRYPTION_KEY'] = 'test-encryption-key-32-characters-long'
    
    try:
        print("Testing Production Configuration Management")
        print("=" * 50)
        
        # Create configuration manager
        manager = create_production_config_manager("production.env.template")
        
        print(f"Configuration loaded successfully!")
        print(f"Environment: {manager.config.environment}")
        print(f"Service: {manager.config.service_name} v{manager.config.deployment_version}")
        print(f"Trading Mode: {'Paper' if manager.config.trading.paper_trading else 'Live'}")
        
        # Get configuration summary
        summary = manager.get_configuration_summary()
        print(f"\nConfiguration Summary:")
        for key, value in summary.items():
            if key != 'component_status':
                print(f"  {key}: {value}")
        
        print(f"\nComponent Status:")
        for component, status in summary['component_status'].items():
            print(f"  {component}: {status}")
        
        # Validate production environment
        print(f"\nValidating Production Environment...")
        validation = validate_production_environment()
        
        print(f"Environment Ready: {validation['environment_ready']}")
        if validation['issues']:
            print(f"Issues: {len(validation['issues'])}")
            for issue in validation['issues']:
                print(f"  - {issue}")
        
        if validation['warnings']:
            print(f"Warnings: {len(validation['warnings'])}")
            for warning in validation['warnings']:
                print(f"  - {warning}")
        
        print("\nProduction configuration management test completed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        raise