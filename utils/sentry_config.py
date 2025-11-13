"""
Sentry configuration for error tracking and monitoring.
Professional error management replacing custom error handling.
"""
import os
import sentry_sdk
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
from typing import Optional
import logging
from dotenv import load_dotenv

# Configure Sentry logging integration
sentry_logging = LoggingIntegration(
    level=logging.INFO,        # Capture info and above as breadcrumbs
    event_level=logging.ERROR  # Send error and above as events
)

def init_sentry(
    environment: str = "development",
    debug: bool = False,
    fallback_to_console: bool = True
) -> bool:
    """
    Initialize Sentry error tracking.
    
    Args:
        environment: Environment name (development/production)
        debug: Enable debug mode
        fallback_to_console: Fall back to console logging if Sentry fails
        
    Returns:
        bool: True if Sentry initialized successfully, False otherwise
    """
    # Load environment variables
    load_dotenv()
    sentry_dsn = os.getenv('SENTRY_DSN')
    
    if not sentry_dsn:
        if fallback_to_console:
            logging.warning("SENTRY_DSN not configured, falling back to console logging")
        return False
    
    try:
        sentry_sdk.init(
            dsn=sentry_dsn,
            environment=environment,
            debug=debug,
            # Sample rate for performance monitoring
            traces_sample_rate=0.1,
            # Send session data
            auto_session_tracking=True,
            # Integrations
            integrations=[
                SqlalchemyIntegration(),
                sentry_logging
            ],
            # Release tracking
            release=os.getenv('APP_VERSION', 'unknown'),
            # Send default PII (user data)
            send_default_pii=True
        )
        
        # Set user context after initialization
        with sentry_sdk.configure_scope() as scope:
            scope.set_user({"wallet": os.getenv('WALLET_ADDRESS', 'unknown')})
        logging.info(f"Sentry initialized successfully for environment: {environment}")
        return True
        
    except Exception as e:
        if fallback_to_console:
            logging.error(f"Failed to initialize Sentry: {e}")
            logging.warning("Falling back to console logging")
        return False

def capture_trading_error(error: Exception, context: dict = None):
    """
    Capture trading-related errors with context.
    
    Args:
        error: The exception to capture
        context: Additional context about the trading operation
    """
    with sentry_sdk.configure_scope() as scope:
        if context:
            for key, value in context.items():
                scope.set_context(key, value)
        
        # Add trading-specific tags
        scope.set_tag("error_type", "trading")
        scope.set_tag("component", context.get("component", "unknown") if context else "unknown")
        
        sentry_sdk.capture_exception(error)

def capture_api_error(error: Exception, api_name: str, endpoint: str = None, context: dict = None):
    """
    Capture API-related errors with context.
    
    Args:
        error: The exception to capture
        api_name: Name of the API (Jupiter, Alchemy, etc.)
        endpoint: Specific endpoint that failed
        context: Additional context
    """
    with sentry_sdk.configure_scope() as scope:
        if context:
            for key, value in context.items():
                scope.set_context(key, value)
        
        scope.set_tag("error_type", "api")
        scope.set_tag("api_name", api_name)
        if endpoint:
            scope.set_tag("endpoint", endpoint)
        
        sentry_sdk.capture_exception(error)

def test_sentry_connection() -> bool:
    """
    Test Sentry connection by sending a test message.
    
    Returns:
        bool: True if test successful, False otherwise
    """
    try:
        sentry_sdk.capture_message("SolTrader Sentry test - MCP setup", level="info")
        return True
    except Exception as e:
        logging.error(f"Sentry test failed: {e}")
        return False