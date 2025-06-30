#!/usr/bin/env python3
"""
Console Bot Monitor - Real-time monitoring without Telegram
Displays bot activity, trades, and performance in a live console interface
"""

import os
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import sys

try:
    import colorama
    from colorama import Fore, Back, Style
    colorama.init()
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False
    # Fallback color codes
    class Fore:
        GREEN = RED = YELLOW = CYAN = MAGENTA = BLUE = WHITE = RESET = ""
    class Back:
        BLACK = ""
    class Style:
        BRIGHT = RESET_ALL = ""

class ConsoleMonitor:
    """Real-time console monitoring for SolTrader"""
    
    def __init__(self, log_file="logs/trading.log"):
        self.log_file = log_file
        self.last_position = 0
        self.stats = {
            "bot_start": None,
            "trades_today": 0,
            "pnl_today": 0.0,
            "last_trade": None,
            "positions_open": 0,
            "tokens_scanned": 0,
            "errors_today": 0
        }
        
    def clear_screen(self):
        """Clear console screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self):
        """Print dashboard header"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}🦍 SOLTRADER APE BOT - LIVE MONITOR{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Time: {now}{Style.RESET_ALL}")
        print()
    
    def print_stats(self):
        """Print current statistics"""
        print(f"{Fore.MAGENTA}📊 PERFORMANCE STATS{Style.RESET_ALL}")
        print(f"{Fore.CYAN}├─{Style.RESET_ALL} Trades Today: {Fore.YELLOW}{self.stats['trades_today']}{Style.RESET_ALL}")
        
        pnl_color = Fore.GREEN if self.stats['pnl_today'] >= 0 else Fore.RED
        print(f"{Fore.CYAN}├─{Style.RESET_ALL} P&L Today: {pnl_color}${self.stats['pnl_today']:.2f}{Style.RESET_ALL}")
        
        print(f"{Fore.CYAN}├─{Style.RESET_ALL} Open Positions: {Fore.BLUE}{self.stats['positions_open']}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}├─{Style.RESET_ALL} Tokens Scanned: {Fore.GREEN}{self.stats['tokens_scanned']}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}└─{Style.RESET_ALL} Errors Today: {Fore.RED}{self.stats['errors_today']}{Style.RESET_ALL}")
        print()
    
    def print_recent_activity(self, lines):
        """Print recent log activity"""
        print(f"{Fore.MAGENTA}📱 RECENT ACTIVITY (Last 10 events){Style.RESET_ALL}")
        
        if not lines:
            print(f"{Fore.YELLOW}   No recent activity...{Style.RESET_ALL}")
            return
            
        for line in lines[-10:]:
            timestamp = line[:19] if len(line) > 19 else ""
            message = line[19:].strip()
            
            # Color code based on message type
            if "ERROR" in line:
                print(f"{Fore.RED}🔴 {timestamp} {message}{Style.RESET_ALL}")
            elif "Position opened" in line or "bought" in line.lower():
                print(f"{Fore.GREEN}🟢 {timestamp} {message}{Style.RESET_ALL}")
            elif "Position closed" in line or "sold" in line.lower():
                print(f"{Fore.BLUE}🔵 {timestamp} {message}{Style.RESET_ALL}")
            elif "New token" in line or "token found" in line.lower():
                print(f"{Fore.YELLOW}🟡 {timestamp} {message}{Style.RESET_ALL}")
            elif "WARNING" in line:
                print(f"{Fore.MAGENTA}🟣 {timestamp} {message}{Style.RESET_ALL}")
            else:
                print(f"{Fore.WHITE}⚪ {timestamp} {message}{Style.RESET_ALL}")
    
    def parse_log_updates(self):
        """Parse new log entries and update stats"""
        if not os.path.exists(self.log_file):
            return []
            
        try:
            with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(self.last_position)
                new_lines = f.readlines()
                self.last_position = f.tell()
            
            # Update statistics based on new lines
            for line in new_lines:
                self.update_stats_from_line(line)
                
            return [line.strip() for line in new_lines if line.strip()]
            
        except Exception as e:
            return [f"Error reading log: {str(e)}"]
    
    def update_stats_from_line(self, line):
        """Update statistics from log line"""
        try:
            if "trading started" in line.lower():
                self.stats["bot_start"] = datetime.now()
                
            elif "position opened" in line.lower() or "bought" in line.lower():
                self.stats["trades_today"] += 1
                self.stats["positions_open"] += 1
                
            elif "position closed" in line.lower() or "sold" in line.lower():
                self.stats["positions_open"] = max(0, self.stats["positions_open"] - 1)
                # Try to extract P&L from line
                if "profit" in line.lower() or "loss" in line.lower():
                    # Simple regex-like extraction
                    words = line.split()
                    for i, word in enumerate(words):
                        if word.startswith('$') or word.startswith('+') or word.startswith('-'):
                            try:
                                pnl = float(word.replace('$', '').replace('+', ''))
                                self.stats["pnl_today"] += pnl
                                break
                            except:
                                pass
                                
            elif "new token" in line.lower() or "scanning" in line.lower():
                self.stats["tokens_scanned"] += 1
                
            elif "ERROR" in line:
                self.stats["errors_today"] += 1
                
        except Exception:
            pass  # Ignore parsing errors
    
    def print_controls(self):
        """Print control instructions"""
        print(f"{Fore.CYAN}{'─' * 60}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}🎮 CONTROLS: Press Ctrl+C to stop monitoring{Style.RESET_ALL}")
        print(f"{Fore.WHITE}📊 Updates every 2 seconds automatically{Style.RESET_ALL}")
        print(f"{Fore.WHITE}📁 Log file: {self.log_file}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'─' * 60}{Style.RESET_ALL}")
    
    def run(self):
        """Main monitoring loop"""
        print(f"{Fore.GREEN}🚀 Starting SolTrader Console Monitor...{Style.RESET_ALL}")
        time.sleep(1)
        
        try:
            while True:
                # Get new log entries
                new_lines = self.parse_log_updates()
                
                # Clear and redraw screen
                self.clear_screen()
                self.print_header()
                self.print_stats()
                self.print_recent_activity(new_lines)
                self.print_controls()
                
                # Wait before next update
                time.sleep(2)
                
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}👋 Monitor stopped by user{Style.RESET_ALL}")
        except Exception as e:
            print(f"\n{Fore.RED}❌ Monitor error: {str(e)}{Style.RESET_ALL}")

def check_bot_running():
    """Check if bot is currently running"""
    try:
        # Try to find python process running main.py
        if os.name == 'nt':  # Windows
            result = subprocess.run(['tasklist'], capture_output=True, text=True)
            return 'python' in result.stdout.lower()
        else:  # Unix-like
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            return 'main.py' in result.stdout
    except:
        return False

def main():
    """Main entry point"""
    print(f"{Fore.CYAN}🦍 SolTrader APE Bot - Console Monitor{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Real-time monitoring without Telegram{Style.RESET_ALL}")
    print()
    
    # Check if bot is running
    if check_bot_running():
        print(f"{Fore.GREEN}✅ Bot appears to be running{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}⚠️  Bot doesn't appear to be running{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Start the bot with: python main.py{Style.RESET_ALL}")
    
    print()
    
    # Find log file
    possible_logs = [
        "logs/trading.log",
        "trading.log", 
        "bot.log",
        "soltrader.log"
    ]
    
    log_file = None
    for log_path in possible_logs:
        if os.path.exists(log_path):
            log_file = log_path
            break
    
    if not log_file:
        print(f"{Fore.RED}❌ No log file found. Creating logs directory...{Style.RESET_ALL}")
        os.makedirs("logs", exist_ok=True)
        log_file = "logs/trading.log"
        
        # Create empty log file
        with open(log_file, 'w') as f:
            f.write(f"{datetime.now().isoformat()} - Monitor - Log file created\n")
    
    print(f"{Fore.GREEN}📁 Using log file: {log_file}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Press Enter to start monitoring...{Style.RESET_ALL}")
    input()
    
    # Start monitoring
    monitor = ConsoleMonitor(log_file)
    monitor.run()

if __name__ == "__main__":
    main()