# ⚡ Quick Setup Guide

Get OCR API running in 10 minutes!

---

## 📦 What You Need

- Python 3.8 or higher
- MongoDB
- Tesseract OCR

---

## 🪟 Windows Setup

### Step 1: Install Python
1. Download from [python.org](https://www.python.org/downloads/)
2. ✅ Check "Add Python to PATH" during installation
3. Test: Open CMD and type `python --version`

### Step 2: Install MongoDB
1. Download from [mongodb.com](https://www.mongodb.com/try/download/community)
2. Install with default settings
3. Start: `net start MongoDB`

### Step 3: Install Tesseract
1. Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
2. Install to: `C:\Program Files\Tesseract-OCR`

### Step 4: Install Poppler
1. Download from [poppler-windows](http://blog.alivate.com.au/poppler-windows/)
2. Extract to: `C:\Program Files\poppler`
3. Add to PATH: `C:\Program Files\poppler\Library\bin`

### Step 5: Setup Project
```bash
# Clone project
git clone https://github.com/yourusername/ocr-api.git
cd ocr-api

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install packages
pip install -r requirements.txt

# Create config file
copy .env.example .env
notepad .env
```

**Edit .env:**
```ini
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DATABASE=ocr_database
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
PORT=8000
```

### Step 6: Run
```bash
python server.py
```

**Open:** http://localhost:8000

---

## 🐧 Linux Setup (Ubuntu/Debian)

### One Command Install
```bash
# Install everything
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv mongodb tesseract-ocr poppler-utils

# Clone project
git clone https://github.com/yourusername/ocr-api.git
cd ocr-api

# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
nano .env
```

**Edit .env:**
```ini
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DATABASE=ocr_database
TESSERACT_PATH=/usr/bin/tesseract
PORT=8000
```

### Run
```bash
# Start MongoDB
sudo systemctl start mongod

# Run API
python server.py
```

---

## 🍎 Mac Setup

### Install with Homebrew
```bash
# Install Homebrew (if needed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install requirements
brew install python@3.9 mongodb-community tesseract poppler

# Start MongoDB
brew services start mongodb-community

# Clone project
git clone https://github.com/yourusername/ocr-api.git
cd ocr-api

# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
nano .env
```

**Edit .env:**
```ini
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DATABASE=ocr_database
TESSERACT_PATH=/usr/local/bin/tesseract
PORT=8000
```

### Run
```bash
python server.py
```

---

## ✅ Verify Installation

### Test All Components
```bash
# Python
python --version

# MongoDB
mongosh --version

# Tesseract
tesseract --version

# Poppler
pdftoppm -v

# API
curl http://localhost:8000/health
```

**All working?** You're ready to go! 🎉

---

## 🔧 Troubleshooting

### MongoDB Won't Start

**Windows:**
```bash
net start MongoDB
```

**Linux:**
```bash
sudo systemctl start mongod
sudo systemctl status mongod
```

**Mac:**
```bash
brew services start mongodb-community
```

### Tesseract Not Found

Update `.env` with correct path:
```ini
# Find tesseract location
where tesseract    # Windows
which tesseract    # Linux/Mac

# Update .env
TESSERACT_PATH=/path/from/above/command
```

### Port Already in Use

Change port in `.env`:
```ini
PORT=8001
```

### Import Errors

```bash
# Activate virtual environment first!
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Then install again
pip install -r requirements.txt
```

---

## 🎯 Next Steps

1. ✅ Everything installed? Test with a document!
2. 📖 Read API docs: [API_DOCS.md](API_DOCS.md)
3. 🚀 Deploy online: [DEPLOYMENT.md](DEPLOYMENT.md)

---

## 📞 Need Help?

**Something not working?**

1. Check error messages carefully
2. Google the error
3. Ask on [GitHub Issues](https://github.com/yourusername/ocr-api/issues)

**Common Problems:**
- Path issues → Check .env file
- Permission errors → Run as admin (Windows) or use sudo (Linux)
- Module not found → Activate venv and reinstall

---

**Setup Done!** Start scanning documents! 🚀
