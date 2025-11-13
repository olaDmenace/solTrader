#!/usr/bin/env python3
"""
Claude API Server - Secure command execution and monitoring endpoint
Allows Claude to remotely monitor and troubleshoot the SolTrader bot

Security Features:
- API key authentication
- Command whitelist
- Rate limiting
- Execution timeout
- Detailed logging
"""
import os
import sys
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
from functools import wraps
from flask import Flask, request, jsonify, Response
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Configuration
BOT_DIR = Path(__file__).parent.resolve()
LOG_FILE = BOT_DIR / "logs" / "trading.log"
STATUS_FILE = BOT_DIR / "bot_data.json"
API_LOG_FILE = BOT_DIR / "logs" / "claude_api.log"

# Create logs directory if it doesn't exist
(BOT_DIR / "logs").mkdir(exist_ok=True)

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Rate limiting (100 requests per hour per IP)
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

# API Key validation
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        expected_key = os.getenv('CLAUDE_API_KEY')

        if not expected_key:
            log_api_access("ERROR", "API key not configured in environment")
            return jsonify({"error": "Server not configured"}), 500

        if not api_key:
            log_api_access("UNAUTHORIZED", "Missing API key")
            return jsonify({"error": "Missing API key"}), 401

        if api_key != expected_key:
            log_api_access("UNAUTHORIZED", f"Invalid API key: {api_key[:8]}...")
            return jsonify({"error": "Invalid API key"}), 401

        return f(*args, **kwargs)
    return decorated_function

def log_api_access(level, message, extra_data=None):
    """Log API access attempts and actions"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "level": level,
        "message": message,
        "ip": request.remote_addr if request else "N/A",
        "endpoint": request.endpoint if request else "N/A"
    }
    if extra_data:
        log_entry["data"] = extra_data

    with open(API_LOG_FILE, 'a') as f:
        f.write(json.dumps(log_entry) + "\n")

# Whitelisted commands for security
ALLOWED_COMMANDS = [
    'tail', 'head', 'cat', 'grep', 'ls', 'pwd', 'wc',
    'python', 'python3', 'pip', 'git',
    'ps', 'top', 'df', 'du', 'free',
    'systemctl', 'journalctl',
    'find', 'which', 'echo'
]

def is_command_allowed(command):
    """Check if command is in whitelist"""
    cmd_parts = command.strip().split()
    if not cmd_parts:
        return False

    base_command = cmd_parts[0]

    # Allow commands from whitelist
    if base_command in ALLOWED_COMMANDS:
        return True

    # Allow relative paths to bot directory
    if base_command.startswith('./') or base_command.startswith('../'):
        return True

    return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint - no authentication required"""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "server": "claude-api",
        "version": "1.0.0",
        "bot_directory": str(BOT_DIR),
        "log_file_exists": LOG_FILE.exists(),
        "status_file_exists": STATUS_FILE.exists()
    })

@app.route('/execute', methods=['POST'])
@require_api_key
@limiter.limit("30 per minute")
def execute_command():
    """Execute a whitelisted command"""
    try:
        data = request.get_json()
        command = data.get('command')
        timeout = data.get('timeout', 30)  # Default 30 seconds

        if not command:
            return jsonify({"error": "Missing command parameter"}), 400

        if not is_command_allowed(command):
            log_api_access("BLOCKED", f"Command not allowed: {command}")
            return jsonify({
                "error": "Command not allowed",
                "allowed_commands": ALLOWED_COMMANDS
            }), 403

        log_api_access("EXECUTE", f"Running command: {command}")

        # Execute command with timeout
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=min(timeout, 60),  # Max 60 seconds
            cwd=str(BOT_DIR)
        )

        response = {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "command": command,
            "timestamp": datetime.now().isoformat()
        }

        log_api_access("SUCCESS", f"Command executed: {command}", {
            "returncode": result.returncode,
            "output_length": len(result.stdout)
        })

        return jsonify(response)

    except subprocess.TimeoutExpired:
        log_api_access("TIMEOUT", f"Command timeout: {command}")
        return jsonify({"error": "Command timeout"}), 408
    except Exception as e:
        log_api_access("ERROR", f"Execution error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/logs', methods=['GET'])
@require_api_key
@limiter.limit("60 per minute")
def get_logs():
    """Get trading logs"""
    try:
        lines = request.args.get('lines', 1000, type=int)
        lines = min(lines, 5000)  # Max 5000 lines

        if not LOG_FILE.exists():
            return jsonify({"error": "Log file not found"}), 404

        with open(LOG_FILE, 'r') as f:
            all_lines = f.readlines()
            log_lines = all_lines[-lines:]

        log_api_access("LOGS", f"Retrieved {len(log_lines)} log lines")

        return Response(
            ''.join(log_lines),
            mimetype='text/plain'
        )

    except Exception as e:
        log_api_access("ERROR", f"Log retrieval error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/logs/stream', methods=['GET'])
@require_api_key
def stream_logs():
    """Stream logs in real-time (Server-Sent Events)"""
    def generate():
        try:
            if not LOG_FILE.exists():
                yield f"data: Log file not found\n\n"
                return

            with open(LOG_FILE, 'r') as f:
                # Start from end of file
                f.seek(0, 2)

                while True:
                    line = f.readline()
                    if not line:
                        time.sleep(0.5)
                        continue
                    yield f"data: {line}\n\n"

        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"

    log_api_access("STREAM", "Started log streaming")
    return Response(generate(), mimetype='text/event-stream')

@app.route('/status', methods=['GET'])
@require_api_key
@limiter.limit("120 per minute")
def get_status():
    """Get bot status from bot_data.json"""
    try:
        if not STATUS_FILE.exists():
            return jsonify({"error": "Status file not found"}), 404

        with open(STATUS_FILE, 'r') as f:
            status_data = json.load(f)

        log_api_access("STATUS", "Retrieved bot status")
        return jsonify(status_data)

    except Exception as e:
        log_api_access("ERROR", f"Status retrieval error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/files', methods=['GET'])
@require_api_key
@limiter.limit("30 per minute")
def list_files():
    """List files in bot directory"""
    try:
        path = request.args.get('path', '.')
        # Prevent directory traversal
        full_path = (BOT_DIR / path).resolve()

        if not str(full_path).startswith(str(BOT_DIR)):
            return jsonify({"error": "Access denied"}), 403

        if not full_path.exists():
            return jsonify({"error": "Path not found"}), 404

        if full_path.is_file():
            return jsonify({
                "type": "file",
                "path": str(full_path.relative_to(BOT_DIR)),
                "size": full_path.stat().st_size,
                "modified": datetime.fromtimestamp(full_path.stat().st_mtime).isoformat()
            })

        files = []
        for item in full_path.iterdir():
            files.append({
                "name": item.name,
                "type": "directory" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else None,
                "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
            })

        log_api_access("FILES", f"Listed directory: {path}")
        return jsonify({
            "path": str(full_path.relative_to(BOT_DIR)),
            "files": sorted(files, key=lambda x: (x['type'] != 'directory', x['name']))
        })

    except Exception as e:
        log_api_access("ERROR", f"File listing error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/file', methods=['GET'])
@require_api_key
@limiter.limit("30 per minute")
def read_file():
    """Read a specific file"""
    try:
        file_path = request.args.get('path')
        if not file_path:
            return jsonify({"error": "Missing path parameter"}), 400

        full_path = (BOT_DIR / file_path).resolve()

        # Prevent directory traversal
        if not str(full_path).startswith(str(BOT_DIR)):
            return jsonify({"error": "Access denied"}), 403

        if not full_path.exists():
            return jsonify({"error": "File not found"}), 404

        if not full_path.is_file():
            return jsonify({"error": "Not a file"}), 400

        # Max file size 1MB
        if full_path.stat().st_size > 1024 * 1024:
            return jsonify({"error": "File too large (max 1MB)"}), 413

        with open(full_path, 'r') as f:
            content = f.read()

        log_api_access("FILE_READ", f"Read file: {file_path}")
        return Response(content, mimetype='text/plain')

    except Exception as e:
        log_api_access("ERROR", f"File read error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/bot/start', methods=['POST'])
@require_api_key
@limiter.limit("10 per hour")
def start_bot():
    """Start the trading bot"""
    try:
        # Check if already running
        result = subprocess.run(
            "ps aux | grep 'python.*main.py' | grep -v grep",
            shell=True,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            return jsonify({
                "status": "already_running",
                "message": "Bot is already running",
                "processes": result.stdout
            })

        # Start bot in background
        subprocess.Popen(
            ["nohup", "python3", "main.py"],
            cwd=str(BOT_DIR),
            stdout=open(BOT_DIR / 'bot.log', 'a'),
            stderr=subprocess.STDOUT
        )

        time.sleep(2)  # Wait for startup

        log_api_access("BOT_START", "Started trading bot")
        return jsonify({
            "status": "started",
            "message": "Bot started successfully",
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        log_api_access("ERROR", f"Bot start error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/bot/stop', methods=['POST'])
@require_api_key
@limiter.limit("10 per hour")
def stop_bot():
    """Stop the trading bot"""
    try:
        result = subprocess.run(
            "pkill -SIGTERM -f 'python.*main.py'",
            shell=True,
            capture_output=True,
            text=True
        )

        log_api_access("BOT_STOP", "Stopped trading bot")
        return jsonify({
            "status": "stopped",
            "message": "Bot stop signal sent",
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        log_api_access("ERROR", f"Bot stop error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/bot/status', methods=['GET'])
@require_api_key
def bot_running_status():
    """Check if bot is currently running"""
    try:
        result = subprocess.run(
            "ps aux | grep 'python.*main.py' | grep -v grep",
            shell=True,
            capture_output=True,
            text=True
        )

        is_running = result.returncode == 0

        return jsonify({
            "running": is_running,
            "processes": result.stdout if is_running else None,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.errorhandler(429)
def ratelimit_handler(e):
    """Rate limit exceeded"""
    log_api_access("RATELIMIT", f"Rate limit exceeded: {str(e)}")
    return jsonify({
        "error": "Rate limit exceeded",
        "message": str(e)
    }), 429

if __name__ == '__main__':
    # Check for API key
    if not os.getenv('CLAUDE_API_KEY'):
        print("ERROR: CLAUDE_API_KEY environment variable not set!")
        print("Please set it in your .env file or export it:")
        print("  export CLAUDE_API_KEY='your-secret-key-here'")
        sys.exit(1)

    print("=" * 60)
    print("Claude API Server Starting...")
    print("=" * 60)
    print(f"Bot Directory: {BOT_DIR}")
    print(f"Log File: {LOG_FILE}")
    print(f"API Key: {'*' * (len(os.getenv('CLAUDE_API_KEY', '')) - 4)}{os.getenv('CLAUDE_API_KEY', '')[-4:]}")
    print(f"Listening on: 0.0.0.0:8080")
    print("=" * 60)
    print("\nEndpoints:")
    print("  GET  /health          - Health check (no auth)")
    print("  POST /execute         - Execute command")
    print("  GET  /logs            - Get logs")
    print("  GET  /logs/stream     - Stream logs")
    print("  GET  /status          - Bot status")
    print("  GET  /files           - List files")
    print("  GET  /file            - Read file")
    print("  POST /bot/start       - Start bot")
    print("  POST /bot/stop        - Stop bot")
    print("  GET  /bot/status      - Check if running")
    print("=" * 60)

    # Run server
    app.run(
        host='0.0.0.0',
        port=8080,
        debug=False,
        threaded=True
    )
