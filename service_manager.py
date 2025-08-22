#!/usr/bin/env python3
"""
SolTrader Service Manager
Manages all production services and deployments automatically
"""
import subprocess
import time
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
import logging
import sys
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/service_manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SolTraderServiceManager:
    """Manages all SolTrader production services"""
    
    def __init__(self):
        self.services = {}
        self.service_configs = {
            'trading_bot': {
                'command': ['python', 'main.py'],
                'restart_on_failure': True,
                'max_restarts': 5,
                'restart_delay': 30,
                'health_check_interval': 60,
                'description': 'Main trading bot engine'
            },
            'dashboard': {
                'command': ['python', 'create_monitoring_dashboard.py'],
                'restart_on_failure': True,
                'max_restarts': 3,
                'restart_delay': 15,
                'health_check_interval': 30,
                'description': 'Responsive real-time monitoring dashboard (mobile-optimized)'
            },
            'performance_charts': {
                'command': ['python', 'generate_charts.py'],
                'restart_on_failure': False,
                'schedule_interval': 300,  # Every 5 minutes
                'description': 'Performance chart generation'
            },
            'log_rotation': {
                'command': ['python', 'setup_log_rotation.py'],
                'restart_on_failure': False,
                'schedule_interval': 3600,  # Every hour
                'description': 'Log rotation and cleanup'
            },
            'performance_reports': {
                'command': ['python', 'src/utils/performance_reports.py'],
                'restart_on_failure': False,
                'schedule_interval': 1800,  # Every 30 minutes
                'description': 'Generate performance reports'
            },
            'health_monitor': {
                'command': ['python', '-c', 'from src.monitoring.health_monitor import HealthMonitor; HealthMonitor().run_health_check()'],
                'restart_on_failure': False,
                'schedule_interval': 180,  # Every 3 minutes
                'description': 'System health monitoring'
            }
        }
        self.running = False
        
    def start_service(self, service_name):
        """Start a specific service"""
        if service_name not in self.service_configs:
            logger.error(f"Unknown service: {service_name}")
            return False
            
        config = self.service_configs[service_name]
        
        try:
            logger.info(f"Starting {service_name}: {config['description']}")
            
            # For scheduled services, don't keep process running
            if 'schedule_interval' in config:
                self.services[service_name] = {
                    'type': 'scheduled',
                    'config': config,
                    'last_run': None,
                    'next_run': datetime.now(),
                    'status': 'scheduled'
                }
            else:
                # For persistent services, keep process running
                process = subprocess.Popen(
                    config['command'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                self.services[service_name] = {
                    'type': 'persistent',
                    'process': process,
                    'config': config,
                    'start_time': datetime.now(),
                    'restart_count': 0,
                    'status': 'running'
                }
                
            logger.info(f"✅ {service_name} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start {service_name}: {e}")
            return False
    
    def stop_service(self, service_name):
        """Stop a specific service"""
        if service_name not in self.services:
            logger.warning(f"Service {service_name} is not running")
            return
            
        service = self.services[service_name]
        
        if service['type'] == 'persistent' and 'process' in service:
            try:
                service['process'].terminate()
                service['process'].wait(timeout=10)
                logger.info(f"✅ {service_name} stopped successfully")
            except subprocess.TimeoutExpired:
                service['process'].kill()
                logger.warning(f"⚠️ {service_name} force killed")
            except Exception as e:
                logger.error(f"❌ Error stopping {service_name}: {e}")
        
        service['status'] = 'stopped'
    
    def restart_service(self, service_name):
        """Restart a specific service"""
        logger.info(f"🔄 Restarting {service_name}")
        self.stop_service(service_name)
        time.sleep(2)
        return self.start_service(service_name)
    
    def check_service_health(self, service_name):
        """Check if a service is healthy"""
        if service_name not in self.services:
            return False
            
        service = self.services[service_name]
        
        if service['type'] == 'persistent':
            if 'process' not in service:
                return False
            return service['process'].poll() is None
        else:
            # Scheduled services are considered healthy if they're scheduled
            return service['status'] == 'scheduled'
    
    def run_scheduled_service(self, service_name):
        """Run a scheduled service once"""
        if service_name not in self.services:
            return
            
        service = self.services[service_name]
        config = service['config']
        
        try:
            logger.info(f"🔄 Running scheduled service: {service_name}")
            
            result = subprocess.run(
                config['command'],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                logger.info(f"✅ {service_name} completed successfully")
                service['status'] = 'completed'
            else:
                logger.error(f"❌ {service_name} failed: {result.stderr}")
                service['status'] = 'failed'
                
            service['last_run'] = datetime.now()
            service['next_run'] = datetime.now() + timedelta(seconds=config['schedule_interval'])
            
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ {service_name} timed out")
            service['status'] = 'timeout'
        except Exception as e:
            logger.error(f"❌ Error running {service_name}: {e}")
            service['status'] = 'error'
    
    def monitor_services(self):
        """Monitor all services and handle restarts"""
        while self.running:
            try:
                current_time = datetime.now()
                
                for service_name, service in self.services.items():
                    config = service['config']
                    
                    if service['type'] == 'persistent':
                        # Check persistent services
                        if not self.check_service_health(service_name):
                            if config.get('restart_on_failure', False):
                                restart_count = service.get('restart_count', 0)
                                max_restarts = config.get('max_restarts', 3)
                                
                                if restart_count < max_restarts:
                                    logger.warning(f"⚠️ {service_name} unhealthy, restarting...")
                                    service['restart_count'] = restart_count + 1
                                    
                                    # Wait before restart
                                    restart_delay = config.get('restart_delay', 30)
                                    time.sleep(restart_delay)
                                    
                                    self.restart_service(service_name)
                                else:
                                    logger.error(f"❌ {service_name} exceeded max restarts ({max_restarts})")
                                    service['status'] = 'failed_permanently'
                    
                    elif service['type'] == 'scheduled':
                        # Check scheduled services
                        if current_time >= service['next_run']:
                            threading.Thread(
                                target=self.run_scheduled_service,
                                args=(service_name,),
                                daemon=True
                            ).start()
                
                # Sleep before next check
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in service monitor: {e}")
                time.sleep(60)
    
    def get_service_status(self):
        """Get status of all services"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'services': {}
        }
        
        for service_name, service in self.services.items():
            service_status = {
                'type': service['type'],
                'status': service['status'],
                'description': service['config']['description']
            }
            
            if service['type'] == 'persistent':
                service_status.update({
                    'uptime': str(datetime.now() - service.get('start_time', datetime.now())),
                    'restart_count': service.get('restart_count', 0),
                    'pid': service.get('process', {}).pid if 'process' in service else None
                })
            else:
                service_status.update({
                    'last_run': service.get('last_run', {}).isoformat() if service.get('last_run') else None,
                    'next_run': service.get('next_run', {}).isoformat() if service.get('next_run') else None
                })
            
            status['services'][service_name] = service_status
        
        return status
    
    def save_status_report(self):
        """Save service status to file"""
        try:
            os.makedirs('reports/services', exist_ok=True)
            status = self.get_service_status()
            
            with open('reports/services/service_status.json', 'w') as f:
                json.dump(status, f, indent=2)
                
            logger.info("📊 Service status report saved")
        except Exception as e:
            logger.error(f"Error saving status report: {e}")
    
    def start_all_services(self):
        """Start all configured services"""
        logger.info("🚀 Starting SolTrader Service Manager")
        
        # Ensure directories exist
        os.makedirs('logs', exist_ok=True)
        os.makedirs('reports/services', exist_ok=True)
        
        # Start persistent services first
        persistent_services = ['trading_bot', 'dashboard']
        for service_name in persistent_services:
            self.start_service(service_name)
            time.sleep(5)  # Stagger startup
        
        # Setup scheduled services
        scheduled_services = ['performance_charts', 'log_rotation', 'performance_reports', 'health_monitor']
        for service_name in scheduled_services:
            self.start_service(service_name)
        
        # Start monitoring
        self.running = True
        monitor_thread = threading.Thread(target=self.monitor_services, daemon=True)
        monitor_thread.start()
        
        # Start status reporting
        status_thread = threading.Thread(target=self.status_reporter, daemon=True)
        status_thread.start()
        
        logger.info("✅ All services started successfully")
    
    def status_reporter(self):
        """Periodically save status reports"""
        while self.running:
            try:
                self.save_status_report()
                time.sleep(300)  # Every 5 minutes
            except Exception as e:
                logger.error(f"Error in status reporter: {e}")
                time.sleep(60)
    
    def stop_all_services(self):
        """Stop all services"""
        logger.info("🛑 Stopping all services")
        self.running = False
        
        for service_name in list(self.services.keys()):
            self.stop_service(service_name)
        
        logger.info("✅ All services stopped")
    
    def print_status(self):
        """Print current service status"""
        print("\n" + "="*60)
        print("SOLTRADER SERVICE MANAGER STATUS")
        print("="*60)
        
        status = self.get_service_status()
        
        for service_name, service in status['services'].items():
            print(f"\n📋 {service_name.upper()}")
            print(f"   Description: {service['description']}")
            print(f"   Status: {service['status']}")
            print(f"   Type: {service['type']}")
            
            if service['type'] == 'persistent':
                print(f"   Uptime: {service.get('uptime', 'N/A')}")
                print(f"   Restarts: {service.get('restart_count', 0)}")
                print(f"   PID: {service.get('pid', 'N/A')}")
            else:
                print(f"   Last Run: {service.get('last_run', 'Never')}")
                print(f"   Next Run: {service.get('next_run', 'N/A')}")
        
        print("\n" + "="*60)

def main():
    """Main service manager entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SolTrader Service Manager')
    parser.add_argument('action', choices=['start', 'stop', 'restart', 'status'], 
                       help='Action to perform')
    parser.add_argument('--service', help='Specific service to manage')
    
    args = parser.parse_args()
    
    manager = SolTraderServiceManager()
    
    try:
        if args.action == 'start':
            if args.service:
                manager.start_service(args.service)
            else:
                manager.start_all_services()
                print("\n🎯 SolTrader Service Manager is running")
                print("Press Ctrl+C to stop all services")
                
                try:
                    while True:
                        time.sleep(10)
                        if not manager.running:
                            break
                except KeyboardInterrupt:
                    print("\n🛑 Shutdown requested...")
                finally:
                    manager.stop_all_services()
        
        elif args.action == 'stop':
            if args.service:
                manager.stop_service(args.service)
            else:
                manager.stop_all_services()
        
        elif args.action == 'restart':
            if args.service:
                manager.restart_service(args.service)
            else:
                manager.stop_all_services()
                time.sleep(5)
                manager.start_all_services()
        
        elif args.action == 'status':
            manager.print_status()
    
    except Exception as e:
        logger.error(f"Service manager error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()