# 📄 OCR API - Document Scanner

Extract data from Indian identity documents and marksheets automatically.

---

## ✨ What It Does

Upload a document → Get structured data instantly!

**Supports:**
- 🆔 PAN Card
- 🎴 Aadhaar Card  
- 🗳️ Voter ID
- 🚗 Driving Licence
- 📚 Marksheet

**Formats:** PDF, JPG, PNG (Max 16MB)

---

## 🚀 Quick Start (5 Minutes)

### 1. Install Requirements

```bash
# Install Python 3.8+
# Install MongoDB
# Install Tesseract OCR
```

### 2. Setup Project

```bash
# Clone
git clone https://github.com/yourusername/ocr-api.git
cd ocr-api

# Install
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your MongoDB URI and Tesseract path

# Run
python server.py
```

### 3. Test It

```bash
# Open browser
http://localhost:8000

# Or test with curl
curl -X POST -F "file=@document.pdf" http://localhost:8000/api/scan
```

---

## 📖 Usage

### Scan Document

```python
import requests

# Upload file
files = {'file': open('aadhaar.pdf', 'rb')}
response = requests.post('http://localhost:8000/api/scan', files=files)

# Get results
result = response.json()
print(result['fields'])  # {'name': 'John Doe', 'aadhaar_number': '1234...'}
```

### JavaScript

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/api/scan', {
  method: 'POST',
  body: formData
})
.then(res => res.json())
.then(data => console.log(data.fields));
```

---

## 🔌 Main Endpoints

| Endpoint | What It Does | Example |
|----------|--------------|---------|
| `POST /api/scan` | Scan document | Upload file, get data |
| `POST /api/rescan/<id>` | Rescan for better accuracy | If confidence < 80% |
| `POST /api/submit/<id>` | Submit verified data | Save final data |
| `GET /api/scan/<id>` | Get scan details | Retrieve scan info |
| `GET /api/scans` | List all scans | Get all documents |

---

## 📊 What You Get

### PAN Card
```json
{
  "pan": "ABCDE1234F",
  "name": "JOHN DOE",
  "father_name": "FATHER NAME",
  "dob": "01/01/1990"
}
```

### Aadhaar Card
```json
{
  "aadhaar_number": "1234 5678 9012",
  "name": "JOHN DOE",
  "dob": "01/01/1990",
  "gender": "Male",
  "address": "123 Street, City"
}
```

### Marksheet
```json
{
  "student_name": "JOHN DOE",
  "roll_no": "12345678",
  "cgpa": "9.5",
  "subjects": [
    {"subject": "Math", "grade": "A1", "marks": "95"}
  ]
}
```

---

## ⚙️ Configuration

Edit `.env` file:

```ini
# MongoDB
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DATABASE=ocr_database

# Tesseract Path
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe  # Windows
# TESSERACT_PATH=/usr/bin/tesseract  # Linux/Mac

# Server
PORT=8000
```

---

## 🐛 Common Issues

**MongoDB not connecting?**
```bash
# Start MongoDB
net start MongoDB  # Windows
sudo systemctl start mongod  # Linux
```

**Tesseract not found?**
```bash
# Update TESSERACT_PATH in .env file
TESSERACT_PATH=/correct/path/to/tesseract
```

**Port 8000 in use?**
```bash
# Change port in .env
PORT=8001
```

---

## 📁 Project Files

```
ocr-api/
├── server.py              # Main server
├── config.py              # Configuration
├── requirements.txt       # Dependencies
├── .env                   # Your settings (don't commit!)
├── routes/
│   └── ocr_routes.py     # API endpoints
└── services/
    ├── extractor.py      # OCR logic
    └── database.py       # MongoDB
```

---

## 🔒 Security

**Before deploying:**
- [ ] Change `SECRET_KEY` in `.env`
- [ ] Use strong MongoDB password
- [ ] Enable HTTPS
- [ ] Set `FLASK_DEBUG=False`

---

## 📞 Help

- **Docs:** See `API_DOCS.md` for detailed API reference
- **Issues:** [GitHub Issues](https://github.com/yourusername/ocr-api/issues)
- **Email:** support@example.com

---

## 📝 License

MIT License - Feel free to use and modify!

---

## 🎯 Next Steps

1. ✅ Got it running? Test with sample documents
2. 📖 Read full docs: [API_DOCS.md](API_DOCS.md)
3. 🚀 Deploy to production: [DEPLOYMENT.md](DEPLOYMENT.md)
4. 🤝 Contribute: [CONTRIBUTING.md](CONTRIBUTING.md)

---

**Made with ❤️ for easy document processing**

**Star ⭐ this repo if you find it helpful!**
