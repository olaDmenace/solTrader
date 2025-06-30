# 🔧 Dependency Installation Fix Guide

You're encountering a build error because some packages need to be compiled for Python 3.12. Here are multiple solutions:

## 🚀 Quick Solution (Recommended)

### Use newer compatible versions:

```cmd
# First, upgrade pip
pip install --upgrade pip

# Install compatible versions that have pre-built wheels
pip install aiohttp>=3.9.0
pip install anchorpy>=0.19.0
pip install base58>=2.1.1
pip install python-dotenv>=1.0.0
pip install solana>=0.32.0
pip install solders>=0.20.0
pip install pytest>=7.4.3
pip install pytest-asyncio>=0.21.1
pip install numpy>=1.24.3
pip install pandas>=2.0.3
pip install scipy>=1.11.3
pip install async-timeout>=4.0.0
pip install python-telegram-bot>=20.0
pip install backoff>=2.2.1
```

## 🛠️ Alternative Solution 1: Install Visual Studio Build Tools

If you want to use the exact versions in requirements.txt:

1. **Download Microsoft C++ Build Tools:**
   - Go to: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Download "Build Tools for Visual Studio 2022"
   - Install with "C++ build tools" workload selected

2. **After installation, try again:**
   ```cmd
   pip install -r requirements.txt
   ```

## 🐍 Alternative Solution 2: Use Python 3.11

Python 3.11 has better compatibility with pre-built wheels:

1. **Download Python 3.11:**
   - Go to: https://www.python.org/downloads/release/python-3118/
   - Download and install Python 3.11.8

2. **Recreate virtual environment:**
   ```cmd
   # Remove old environment
   rmdir /s venv
   
   # Create new with Python 3.11
   py -3.11 -m venv venv
   venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## 📦 Alternative Solution 3: Use pre-compiled wheels

Try installing from pre-compiled wheels:

```cmd
# Install from wheel files (usually available for common packages)
pip install --only-binary=all aiohttp
pip install --only-binary=all -r requirements.txt
```

## ✅ Recommended Approach (Step by Step)

### Step 1: Update requirements.txt

Replace your current requirements.txt with this updated version:

```txt
aiohttp>=3.9.0
anchorpy>=0.19.0
base58>=2.1.1
python-dotenv>=1.0.0
solana>=0.32.0
solders>=0.20.0
pytest>=7.4.3
pytest-asyncio>=0.21.1
numpy>=1.24.3
pandas>=2.0.3
scipy>=1.11.3
async-timeout>=4.0.0
python-telegram-bot>=20.0
backoff>=2.2.1
```

### Step 2: Install updated packages

```cmd
# Make sure virtual environment is activated
venv\Scripts\activate

# Upgrade pip first
python -m pip install --upgrade pip

# Install packages
pip install -r requirements.txt
```

### Step 3: Verify installation

```cmd
python verify_setup.py
```

## 🔍 Troubleshooting

### If you still get build errors:

1. **Check Python version:**
   ```cmd
   python --version
   ```

2. **Clear pip cache:**
   ```cmd
   pip cache purge
   ```

3. **Try individual package installation:**
   ```cmd
   pip install aiohttp --no-cache-dir
   ```

4. **Use conda instead of pip (if you have it):**
   ```cmd
   conda install aiohttp
   ```

### Common Windows Issues:

1. **Long path names:** Enable long path support in Windows
2. **Permissions:** Run command prompt as Administrator
3. **Antivirus:** Temporarily disable real-time scanning

## 🎯 Quick Test

After installation, test if imports work:

```cmd
python -c "import aiohttp; print('aiohttp:', aiohttp.__version__)"
python -c "import solana; print('solana: OK')"
python -c "import numpy; print('numpy: OK')"
```

## 🆘 If Nothing Works

Create a minimal environment with just essential packages:

```cmd
pip install requests>=2.31.0
pip install websockets>=11.0
pip install base58>=2.1.1
pip install python-dotenv>=1.0.0
pip install numpy>=1.24.3
```

Then modify the bot to use `requests` instead of `aiohttp` for HTTP calls.

---

**The quickest solution is usually updating to newer package versions that have pre-built wheels for Python 3.12. Try the "Quick Solution" first!**