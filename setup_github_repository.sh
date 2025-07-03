#!/bin/bash
# Setup GitHub Repository for SolTrader

echo "📁 Setting up GitHub repository for SolTrader..."
echo "==============================================="

cd /home/trader/solTrader

# Initialize git repository
echo "1. Initializing git repository..."
git init

# Create .gitignore
echo "2. Creating .gitignore..."
cat > .gitignore << 'EOF'
# Environment files (CRITICAL - Don't commit private keys!)
.env
*.env
.env.local
.env.production

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Logs
logs/
*.log

# Backups
*_backup*
*.backup

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Temporary files
temp_*
fix_*
*_temp*
EOF

# Add files to git
echo "3. Adding files to git..."
git add .
git commit -m "Initial commit: SolTrader APE Bot - Fully functional trading bot

Features:
- ✅ 24/7 automated trading on VPS
- ✅ Real-time token scanning with Jupiter API
- ✅ Paper trading with 100 SOL balance
- ✅ Web dashboard at https://bot.technicity.digital
- ✅ Momentum-based APE strategy
- ✅ Risk management and position limits
- ✅ Systemd service integration
- ✅ Nginx reverse proxy with SSL

Current Status: 95% functional, actively finding trading signals"

echo "4. Repository ready for GitHub!"
echo ""
echo "🔗 Next steps:"
echo "1. Go to github.com and create a new repository called 'soltrader-ape-bot'"
echo "2. Copy the repository URL"
echo "3. Run these commands:"
echo ""
echo "   git remote add origin https://github.com/YOUR_USERNAME/soltrader-ape-bot.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "⚠️  IMPORTANT: Your .env file is protected and won't be uploaded!"
echo "   This keeps your API keys and private keys safe."
echo ""
echo "✅ Your bot code will be safely backed up on GitHub"
