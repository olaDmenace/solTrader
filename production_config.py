#!/usr/bin/env python3
"""
Production Configuration Manager
===============================

Manages production-specific configuration with environment validation,
security hardening, and performance optimization.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class ProductionConfig:
    """Production configuration with validation and security"""
    
    # Environment
    environment: str = "production"
    version: str = "v2.0.0"
    deployment_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Security Configuration
    enable_https: bool = True
    ssl_cert_path: str = "/etc/ssl/certs/soltrader.crt"
    ssl_key_path: str = "/etc/ssl/private/soltrader.key"
    api_key_rotation_enabled: bool = True
    api_key_rotation_interval_hours: int = 24
    security_audit_enabled: bool = True
    
    # Performance Configuration
    max_concurrent_connections: int = 100
    connection_timeout_seconds: int = 30
    request_timeout_seconds: int = 60
    max_retries: int = 3
    rate_limit_per_minute: int = 1000
    burst_rate_limit: int = 50
    
    # Database Configuration
    database_pool_size: int = 20
    database_pool_max_overflow: int = 30
    database_query_timeout_seconds: int = 30
    database_backup_enabled: bool = True
    database_backup_interval_hours: int = 6
    
    # Caching Configuration
    redis_enabled: bool = True
    redis_cluster_enabled: bool = False
    cache_ttl_seconds: int = 300
    cache_max_memory_mb: int = 512
    cache_eviction_policy: str = "allkeys-lru"
    
    # Monitoring Configuration
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    grafana_enabled: bool = True
    grafana_port: int = 3000
    log_level: str = "INFO"
    log_retention_days: int = 30
    metrics_collection_interval_seconds: int = 15
    
    # Trading Configuration
    trading_enabled: bool = True
    paper_trading_mode: bool = False  # CRITICAL: Should be False for live trading
    emergency_controls_enabled: bool = True
    position_size_validation: bool = True
    risk_monitoring_enabled: bool = True
    
    # API Configuration
    api_health_checks_enabled: bool = True
    api_health_check_interval_seconds: int = 30
    api_failover_enabled: bool = True
    api_circuit_breaker_enabled: bool = True
    api_quota_monitoring_enabled: bool = True
    
    # Resource Limits
    max_memory_usage_mb: int = 1024
    max_cpu_usage_percent: int = 70
    max_disk_usage_gb: int = 50
    max_network_connections: int = 200
    
    # Backup and Recovery
    backup_enabled: bool = True
    backup_encryption_enabled: bool = True
    backup_retention_days: int = 90
    disaster_recovery_enabled: bool = True
    
    def __post_init__(self):
        """Validate production configuration"""
        self._validate_security_settings()
        self._validate_performance_settings()
        self._validate_resource_limits()
    
    def _validate_security_settings(self):
        """Validate security configuration"""
        if not self.enable_https:
            logger.warning("HTTPS is disabled - this is not recommended for production")
        
        if not self.api_key_rotation_enabled:
            logger.warning("API key rotation is disabled - security risk")
        
        if not self.security_audit_enabled:
            logger.warning("Security auditing is disabled")
    
    def _validate_performance_settings(self):
        """Validate performance configuration"""
        if self.rate_limit_per_minute < 100:
            logger.warning(f"Rate limit {self.rate_limit_per_minute}/min may be too low")
        
        if self.connection_timeout_seconds > 60:
            logger.warning(f"Connection timeout {self.connection_timeout_seconds}s is high")
        
        if self.database_pool_size < 10:
            logger.warning(f"Database pool size {self.database_pool_size} may be too small")
    
    def _validate_resource_limits(self):
        """Validate resource limits"""
        if self.max_memory_usage_mb > 2048:
            logger.warning(f"Memory limit {self.max_memory_usage_mb}MB is very high")
        
        if self.max_cpu_usage_percent > 80:
            logger.warning(f"CPU limit {self.max_cpu_usage_percent}% is high")

class ProductionConfigManager:
    """Manages production configuration loading and validation"""
    
    def __init__(self, config_path: str = "production_config.json"):
        self.config_path = config_path
        self.config: Optional[ProductionConfig] = None
        
    def load_config(self) -> ProductionConfig:
        """Load production configuration from file and environment"""
        
        # Load from file if exists
        file_config = {}
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.error(f"Failed to load config file: {e}")
        
        # Override with environment variables
        env_config = self._load_from_environment()
        
        # Merge configurations (env overrides file)
        merged_config = {**file_config, **env_config}
        
        # Create production config
        self.config = ProductionConfig(**merged_config)
        
        logger.info(f"Production configuration loaded - Environment: {self.config.environment}")
        return self.config
    
    def _load_from_environment(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        
        env_mapping = {
            # Security
            'ENABLE_HTTPS': ('enable_https', bool),
            'SSL_CERT_PATH': ('ssl_cert_path', str),
            'SSL_KEY_PATH': ('ssl_key_path', str),
            'API_KEY_ROTATION_ENABLED': ('api_key_rotation_enabled', bool),
            'SECURITY_AUDIT_ENABLED': ('security_audit_enabled', bool),
            
            # Performance
            'MAX_CONCURRENT_CONNECTIONS': ('max_concurrent_connections', int),
            'CONNECTION_TIMEOUT': ('connection_timeout_seconds', int),
            'REQUEST_TIMEOUT': ('request_timeout_seconds', int),
            'RATE_LIMIT_PER_MINUTE': ('rate_limit_per_minute', int),
            
            # Database
            'DB_POOL_SIZE': ('database_pool_size', int),
            'DB_QUERY_TIMEOUT': ('database_query_timeout_seconds', int),
            'DB_BACKUP_ENABLED': ('database_backup_enabled', bool),
            
            # Caching
            'REDIS_ENABLED': ('redis_enabled', bool),
            'CACHE_TTL_SECONDS': ('cache_ttl_seconds', int),
            'CACHE_MAX_MEMORY_MB': ('cache_max_memory_mb', int),
            
            # Monitoring
            'PROMETHEUS_ENABLED': ('prometheus_enabled', bool),
            'PROMETHEUS_PORT': ('prometheus_port', int),
            'LOG_LEVEL': ('log_level', str),
            
            # Trading
            'TRADING_ENABLED': ('trading_enabled', bool),
            'PAPER_TRADING': ('paper_trading_mode', bool),
            'EMERGENCY_CONTROLS_ENABLED': ('emergency_controls_enabled', bool),
            
            # Resource Limits
            'MAX_MEMORY_MB': ('max_memory_usage_mb', int),
            'MAX_CPU_PERCENT': ('max_cpu_usage_percent', int),
            'MAX_DISK_GB': ('max_disk_usage_gb', int),
            
            # Backup
            'BACKUP_ENABLED': ('backup_enabled', bool),
            'BACKUP_ENCRYPTION_ENABLED': ('backup_encryption_enabled', bool),
            'BACKUP_RETENTION_DAYS': ('backup_retention_days', int)
        }
        
        config = {}
        for env_var, (config_key, config_type) in env_mapping.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    if config_type == bool:
                        config[config_key] = env_value.lower() in ('true', '1', 'yes', 'on')
                    elif config_type == int:
                        config[config_key] = int(env_value)
                    else:
                        config[config_key] = env_value
                except ValueError as e:
                    logger.error(f"Invalid value for {env_var}: {env_value} ({e})")
        
        return config
    
    def save_config(self, config: ProductionConfig = None):
        """Save current configuration to file"""
        if config is None:
            config = self.config
        
        if config is None:
            raise ValueError("No configuration to save")
        
        try:
            # Convert to dict, excluding computed fields
            config_dict = {
                k: v for k, v in config.__dict__.items() 
                if not k.startswith('_')
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def validate_production_readiness(self) -> Dict[str, Any]:
        """Validate system is ready for production deployment"""
        if not self.config:
            raise ValueError("No configuration loaded")
        
        checks = {
            'security_enabled': self.config.enable_https and self.config.security_audit_enabled,
            'monitoring_enabled': self.config.prometheus_enabled and self.config.grafana_enabled,
            'backup_enabled': self.config.backup_enabled and self.config.backup_encryption_enabled,
            'resource_limits_set': (
                self.config.max_memory_usage_mb > 0 and 
                self.config.max_cpu_usage_percent > 0
            ),
            'api_protection_enabled': (
                self.config.api_health_checks_enabled and
                self.config.api_circuit_breaker_enabled
            ),
            'database_optimized': (
                self.config.database_pool_size >= 10 and
                self.config.database_backup_enabled
            )
        }
        
        warnings = []
        if not self.config.api_key_rotation_enabled:
            warnings.append("API key rotation is disabled")
        
        if self.config.paper_trading_mode:
            warnings.append("Paper trading mode is enabled - not live trading")
        
        if self.config.log_level == "DEBUG":
            warnings.append("Debug logging is enabled - may impact performance")
        
        passed_checks = sum(checks.values())
        total_checks = len(checks)
        readiness_score = (passed_checks / total_checks) * 100
        
        return {
            'readiness_score': readiness_score,
            'checks_passed': passed_checks,
            'total_checks': total_checks,
            'detailed_checks': checks,
            'warnings': warnings,
            'production_ready': readiness_score >= 80 and not warnings
        }
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get current environment information"""
        return {
            'environment': self.config.environment if self.config else 'unknown',
            'version': self.config.version if self.config else 'unknown',
            'deployment_time': self.config.deployment_timestamp if self.config else 'unknown',
            'python_version': os.sys.version,
            'working_directory': os.getcwd(),
            'config_file': self.config_path,
            'config_loaded': self.config is not None
        }

def create_production_config() -> ProductionConfig:
    """Create optimized production configuration"""
    
    return ProductionConfig(
        # Security hardened
        enable_https=True,
        api_key_rotation_enabled=True,
        security_audit_enabled=True,
        
        # Performance optimized
        max_concurrent_connections=100,
        connection_timeout_seconds=30,
        rate_limit_per_minute=1000,
        
        # Database optimized
        database_pool_size=20,
        database_backup_enabled=True,
        
        # Monitoring enabled
        prometheus_enabled=True,
        grafana_enabled=True,
        log_level="INFO",
        
        # Trading configuration
        trading_enabled=True,
        paper_trading_mode=False,  # LIVE TRADING
        emergency_controls_enabled=True,
        
        # Resource limits
        max_memory_usage_mb=1024,
        max_cpu_usage_percent=70,
        
        # Backup enabled
        backup_enabled=True,
        backup_encryption_enabled=True
    )

if __name__ == "__main__":
    # Example usage
    manager = ProductionConfigManager()
    config = manager.load_config()
    
    print("Production Configuration Summary:")
    print(f"Environment: {config.environment}")
    print(f"Version: {config.version}")
    print(f"HTTPS Enabled: {config.enable_https}")
    print(f"Paper Trading: {config.paper_trading_mode}")
    print(f"Rate Limit: {config.rate_limit_per_minute}/min")
    print(f"Memory Limit: {config.max_memory_usage_mb}MB")
    
    # Validate production readiness
    readiness = manager.validate_production_readiness()
    print(f"\nProduction Readiness: {readiness['readiness_score']:.1f}%")
    print(f"Checks Passed: {readiness['checks_passed']}/{readiness['total_checks']}")
    
    if readiness['warnings']:
        print("Warnings:")
        for warning in readiness['warnings']:
            print(f"  - {warning}")