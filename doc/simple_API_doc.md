# 📡 API Documentation - Simple Guide

Easy-to-use API reference for developers.

---

## 🌐 Base URL

```
http://localhost:8000
```

All endpoints start with `/api`

---

## 🚀 Quick Example

```javascript
// Upload and scan
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('http://localhost:8000/api/scan', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(result.fields); // Extracted data
```

---

## 📌 Main Endpoints

### 1️⃣ Scan Document

**Upload a document and extract data**

```http
POST /api/scan
```

**Request:**
```bash
curl -X POST -F "file=@document.pdf" http://localhost:8000/api/scan
```

**Response:**
```json
{
  "success": true,
  "scan_id": "abc-123",
  "document_type": "Aadhaar",
  "fields": {
    "aadhaar_number": "1234 5678 9012",
    "name": "JOHN DOE",
    "dob": "01/01/1990"
  },
  "confidence": 92.5
}
```

---

### 2️⃣ Rescan Document

**Re-scan if accuracy is low**

```http
POST /api/rescan/<scan_id>
```

**Request:**
```bash
curl -X POST -F "file=@better_image.pdf" \
  http://localhost:8000/api/rescan/abc-123
```

**When to use:** If confidence < 80%

---

### 3️⃣ Submit Data

**Save verified/corrected data**

```http
POST /api/submit/<scan_id>
```

**Request:**
```bash
curl -X POST http://localhost:8000/api/submit/abc-123 \
  -H "Content-Type: application/json" \
  -d '{
    "verified_fields": {
      "name": "John Doe",
      "aadhaar_number": "1234 5678 9012"
    },
    "user_corrections": {
      "name": "Corrected Name"
    }
  }'
```

**Simple submit (auto-uses scan data):**
```bash
curl -X POST http://localhost:8000/api/submit/abc-123
```

---

### 4️⃣ Get Scan Details

```http
GET /api/scan/<scan_id>
```

```bash
curl http://localhost:8000/api/scan/abc-123
```

---

### 5️⃣ List All Scans

```http
GET /api/scans?limit=10&skip=0
```

```bash
curl http://localhost:8000/api/scans
```

---

### 6️⃣ Batch Process

**Upload multiple files at once**

```http
POST /api/batch/scan
```

```javascript
const formData = new FormData();
files.forEach(file => formData.append('files', file));

fetch('http://localhost:8000/api/batch/scan', {
  method: 'POST',
  body: formData
});
```

---

## 📄 Document Types & Fields

### PAN Card
```json
{
  "document_type": "PAN",
  "fields": {
    "pan": "ABCDE1234F",
    "name": "JOHN DOE",
    "father_name": "FATHER NAME",
    "dob": "01/01/1990"
  }
}
```

### Aadhaar Card
```json
{
  "document_type": "Aadhaar",
  "fields": {
    "aadhaar_number": "1234 5678 9012",
    "name": "JOHN DOE",
    "dob": "01/01/1990",
    "gender": "Male",
    "father_name": "FATHER NAME",
    "address": "123 Street, City, State - 123456"
  }
}
```

### Voter ID
```json
{
  "document_type": "Voter ID",
  "fields": {
    "voter_id": "ABC1234567",
    "name": "JOHN DOE",
    "father_name": "FATHER NAME",
    "dob": "01/01/1990",
    "gender": "Male",
    "address": "123 Street"
  }
}
```

### Driving Licence
```json
{
  "document_type": "Driving Licence",
  "fields": {
    "dl_number": "TS0612345678901",
    "name": "JOHN DOE",
    "dob": "01/01/1990",
    "issue_date": "01/01/2020",
    "valid_till": "01/01/2040",
    "address": "123 Street"
  }
}
```

### Marksheet
```json
{
  "document_type": "Marksheet",
  "fields": {
    "student_name": "JOHN DOE",
    "roll_no": "12345678",
    "school_name": "XYZ HIGH SCHOOL",
    "cgpa": "9.5"
  },
  "table": [
    {
      "subject": "Mathematics",
      "grade": "A1",
      "marks": "95"
    }
  ]
}
```

---

## ⚠️ Error Responses

```json
{
  "success": false,
  "error": "No file uploaded",
  "message": "Please upload a file"
}
```

**Common Errors:**
- `400` - Bad request (no file, wrong format)
- `404` - Not found (scan_id doesn't exist)
- `413` - File too large (max 16MB)
- `500` - Server error

---

## 💻 Code Examples

### Python
```python
import requests

# Scan
files = {'file': open('document.pdf', 'rb')}
r = requests.post('http://localhost:8000/api/scan', files=files)
data = r.json()

print(data['scan_id'])
print(data['fields'])
```

### JavaScript (Fetch)
```javascript
async function scanDocument(file) {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('http://localhost:8000/api/scan', {
    method: 'POST',
    body: formData
  });
  
  return await response.json();
}
```

### React
```jsx
function DocumentScanner() {
  const [result, setResult] = useState(null);
  
  const handleUpload = async (e) => {
    const file = e.target.files[0];
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('http://localhost:8000/api/scan', {
      method: 'POST',
      body: formData
    });
    
    const data = await response.json();
    setResult(data);
  };
  
  return (
    <div>
      <input type="file" onChange={handleUpload} />
      {result && <pre>{JSON.stringify(result.fields, null, 2)}</pre>}
    </div>
  );
}
```

### cURL
```bash
# Scan
curl -X POST -F "file=@document.pdf" \
  http://localhost:8000/api/scan

# Get scan
curl http://localhost:8000/api/scan/abc-123

# Submit
curl -X POST http://localhost:8000/api/submit/abc-123 \
  -H "Content-Type: application/json" \
  -d '{"verified_fields": {...}}'
```

---

## 🎯 Complete Workflow

```javascript
// 1. Scan document
const scanResponse = await fetch('/api/scan', {
  method: 'POST',
  body: formData
});
const scan = await scanResponse.json();

// 2. Check confidence
if (scan.confidence < 80) {
  // Rescan with better image
  const rescanResponse = await fetch(`/api/rescan/${scan.scan_id}`, {
    method: 'POST',
    body: betterFormData
  });
}

// 3. Review and edit fields
// User reviews and corrects data in UI

// 4. Submit
await fetch(`/api/submit/${scan.scan_id}`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    verified_fields: editedFields,
    user_corrections: corrections
  })
});
```

---

## 📊 Response Structure

**Success:**
```json
{
  "success": true,
  "scan_id": "...",
  "fields": {...},
  "confidence": 92.5,
  "message": "Success"
}
```

**Error:**
```json
{
  "success": false,
  "error": "Error type",
  "message": "Error description"
}
```

---

## 🔒 File Requirements

- **Formats:** PDF, PNG, JPG, JPEG
- **Max Size:** 16MB
- **Quality:** 300+ DPI recommended
- **Orientation:** Straight, not rotated

---

## 💡 Tips

### Better Accuracy
- ✅ Use high-quality scans (300+ DPI)
- ✅ Ensure document is well-lit
- ✅ Keep document flat and straight
- ✅ Avoid shadows and glare
- ✅ Use rescan if confidence < 80%

### Performance
- Use batch processing for multiple files
- Compress large images before upload
- Cache scan results on frontend

### Error Handling
```javascript
try {
  const response = await fetch('/api/scan', {
    method: 'POST',
    body: formData
  });
  
  if (!response.ok) {
    throw new Error('Upload failed');
  }
  
  const data = await response.json();
  
  if (!data.success) {
    throw new Error(data.message);
  }
  
  // Process data
} catch (error) {
  console.error('Error:', error.message);
  alert('Please try again');
}
```

---

## 📞 Support

**API Not Working?**
1. Check if server is running: `http://localhost:8000/health`
2. Verify file format and size
3. Check browser console for errors
4. Review server logs: `logs/server.log`

**Need Help?**
- GitHub Issues: [Link](https://github.com/yourusername/ocr-api/issues)
- Email: support@example.com

---

## 🚀 Quick Test

```bash
# Test health
curl http://localhost:8000/health

# Test scan
curl -X POST -F "file=@test.pdf" \
  http://localhost:8000/api/scan

# Test stats
curl http://localhost:8000/api/statistics
```

---

**Ready to integrate?** Start building! 🎉

**Questions?** Check the full docs or open an issue.
