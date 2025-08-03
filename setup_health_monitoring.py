#!/usr/bin/env python3
"""
Health Monitoring Setup Script
Sets up cron jobs, systemd services, and configuration for health monitoring
"""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

def setup_cron_job():
    """Setup cron job for health checking (Linux/Mac)"""
    try:
        if platform.system() not in ['Linux', 'Darwin']:
            print("⚠️ Cron jobs only supported on Linux/macOS")
            return False
        
        # Create cron job entry
        script_path = Path(__file__).parent / "health_checker.py"
        python_path = Path(__file__).parent / "venv/bin/python"
        
        cron_entry = f"*/5 * * * * {python_path} {script_path} --config health_checker_config.json >> logs/health_checker.log 2>&1\n"
        
        # Add to crontab
        result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
        existing_cron = result.stdout if result.returncode == 0 else ""
        
        if 'health_checker.py' not in existing_cron:
            new_cron = existing_cron + cron_entry
            process = subprocess.Popen(['crontab', '-'], stdin=subprocess.PIPE, text=True)
            process.communicate(input=new_cron)
            
            if process.returncode == 0:
                print("✅ Cron job added successfully (runs every 5 minutes)")
                return True
            else:
                print("❌ Failed to add cron job")
                return False
        else:
            print("✅ Cron job already exists")
            return True
            
    except Exception as e:
        print(f"❌ Error setting up cron job: {e}")
        return False

def setup_windows_task():
    """Setup Windows Task Scheduler task"""
    try:
        if platform.system() != 'Windows':
            print("⚠️ Windows Task Scheduler only supported on Windows")
            return False
        
        script_path = Path(__file__).parent / "health_checker.py"
        python_path = Path(__file__).parent / "venv/Scripts/python.exe"
        
        # Create task XML
        task_xml = f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <Triggers>
    <TimeTrigger>
      <Repetition>
        <Interval>PT5M</Interval>
        <StopAtDurationEnd>false</StopAtDurationEnd>
      </Repetition>
      <StartBoundary>2025-01-01T00:00:00</StartBoundary>
      <Enabled>true</Enabled>
    </TimeTrigger>
  </Triggers>
  <Actions>
    <Exec>
      <Command>{python_path}</Command>
      <Arguments>{script_path} --config health_checker_config.json</Arguments>
      <WorkingDirectory>{Path(__file__).parent}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>"""
        
        # Save task XML
        task_file = Path(__file__).parent / "soltrader_health_check.xml"
        with open(task_file, 'w') as f:
            f.write(task_xml)
        
        # Create task
        cmd = [
            'schtasks', '/create', '/tn', 'SolTrader Health Check',
            '/xml', str(task_file), '/f'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Windows Task Scheduler task created successfully")
            task_file.unlink()  # Clean up XML file
            return True
        else:
            print(f"❌ Failed to create Windows task: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error setting up Windows task: {e}")
        return False

def create_systemd_service():
    """Create systemd service file (Linux)"""
    try:
        if platform.system() != 'Linux':
            print("⚠️ Systemd only supported on Linux")
            return False
        
        bot_path = Path(__file__).parent
        python_path = bot_path / "venv/bin/python"
        main_script = bot_path / "main.py"
        
        service_content = f"""[Unit]
Description=SolTrader Bot
After=network.target

[Service]
Type=simple
User={os.getenv('USER', 'ubuntu')}
WorkingDirectory={bot_path}
ExecStart={python_path} {main_script}
Restart=always
RestartSec=10
Environment=PYTHONPATH={bot_path}

[Install]
WantedBy=multi-user.target
"""
        
        service_file = Path("/etc/systemd/system/soltrader.service")
        
        print(f"📝 Systemd service file content:")
        print(service_content)
        print(f"\n💡 To install the service, run as root:")
        print(f"sudo tee {service_file} <<EOF")
        print(service_content)
        print("EOF")
        print("sudo systemctl daemon-reload")
        print("sudo systemctl enable soltrader")
        print("sudo systemctl start soltrader")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating systemd service: {e}")
        return False

def setup_email_config():
    """Setup email configuration for alerts"""
    try:
        print("\n📧 Setting up email configuration for health alerts...")
        
        # Get email settings from user
        email_user = input("Enter your email address: ").strip()
        if not email_user:
            print("⚠️ Skipping email configuration")
            return True
        
        email_password = input("Enter your email password (or app password): ").strip()
        if not email_password:
            print("⚠️ Skipping email configuration")
            return True
        
        email_to = input(f"Enter alert recipient email (or press Enter for {email_user}): ").strip()
        if not email_to:
            email_to = email_user
        
        # Update health checker config
        config_file = "health_checker_config.json"
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            config['email']['username'] = email_user
            config['email']['password'] = email_password
            config['email']['to_address'] = email_to
            config['email']['enabled'] = True
            
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            print("✅ Email configuration updated")
            return True
        else:
            print("❌ Health checker config file not found")
            return False
            
    except Exception as e:
        print(f"❌ Error setting up email config: {e}")
        return False

def test_health_checker():
    """Test the health checker"""
    try:
        print("\n🧪 Testing health checker...")
        
        script_path = Path(__file__).parent / "health_checker.py"
        if platform.system() == 'Windows':
            python_path = Path(__file__).parent / "venv/Scripts/python.exe"
        else:
            python_path = Path(__file__).parent / "venv/bin/python"
        
        cmd = [str(python_path), str(script_path), "--no-restart", "--log-level", "INFO"]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode in [0, 1]:  # 0 = healthy, 1 = unhealthy but working
            print("✅ Health checker test completed")
            if result.stdout:
                print("📊 Health check results:")
                try:
                    health_data = json.loads(result.stdout)
                    print(f"  Healthy: {health_data.get('healthy', 'unknown')}")
                    print(f"  Issues: {len(health_data.get('issues', []))}")
                except:
                    print(result.stdout)
            return True
        else:
            print(f"❌ Health checker test failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing health checker: {e}")
        return False

def main():
    """Main setup function"""
    print("🛡️ SolTrader Health Monitoring Setup")
    print("=" * 50)
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create health reports directory
    health_dir = Path("health_reports")
    health_dir.mkdir(exist_ok=True)
    
    print("✅ Created necessary directories")
    
    # Setup based on platform
    system = platform.system()
    print(f"\n🖥️ Detected system: {system}")
    
    success_count = 0
    total_steps = 0
    
    if system == "Linux":
        print("\n📅 Setting up cron job...")
        total_steps += 1
        if setup_cron_job():
            success_count += 1
        
        print("\n⚙️ Creating systemd service template...")
        total_steps += 1
        if create_systemd_service():
            success_count += 1
            
    elif system == "Windows":
        print("\n📅 Setting up Windows Task Scheduler...")
        total_steps += 1
        if setup_windows_task():
            success_count += 1
            
    elif system == "Darwin":  # macOS
        print("\n📅 Setting up cron job...")
        total_steps += 1
        if setup_cron_job():
            success_count += 1
    
    # Email configuration
    print("\n📧 Email configuration...")
    total_steps += 1
    if setup_email_config():
        success_count += 1
    
    # Test health checker
    print("\n🧪 Testing health checker...")
    total_steps += 1
    if test_health_checker():
        success_count += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Setup Results: {success_count}/{total_steps} steps completed")
    
    if success_count == total_steps:
        print("🎉 Health monitoring setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Start your SolTrader bot")
        print("2. Monitor health reports in health_reports/ directory")
        print("3. Check logs in logs/ directory")
        print("4. Verify email alerts are working")
    else:
        print("⚠️ Some setup steps failed. Please review the errors above.")
    
    print("\n💡 Manual monitoring commands:")
    print(f"  Health check: python health_checker.py")
    print(f"  Daemon mode:  python health_checker.py --daemon --interval 300")
    print(f"  Test alerts:  python test_health_monitoring.py")

if __name__ == "__main__":
    main()