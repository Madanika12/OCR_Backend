"""
Complete OCR Extractor Service - Enhanced Version
Includes all logic from notebook with YOLO support and advanced extraction
"""

import io
import re
import os
import tempfile
from typing import List, Dict, Any, Tuple, Optional
import pytesseract
from PIL import Image
import numpy as np
from config import Config

# Set Tesseract path
from config import Config
pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_PATH
# ==================== OPTIONAL IMPORTS ====================

# PDF processing
try:
    from pdf2image import convert_from_bytes
    HAVE_PDF2IMAGE = True
except ImportError:
    HAVE_PDF2IMAGE = False
    print("⚠️  pdf2image not available")

try:
    import pdfplumber
    HAVE_PDFPLUMBER = True
except ImportError:
    HAVE_PDFPLUMBER = False
    print("⚠️  pdfplumber not available")

# Advanced OCR
try:
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
    ocr_model = ocr_predictor(pretrained=True).to("cpu")
    HAVE_DOCTR = True
except ImportError:
    HAVE_DOCTR = False
    ocr_model = None
    print("⚠️  doctr not available")

# YOLO for text detection (optional)
try:
    from ultralytics import YOLO
    HAVE_YOLO = Config.ENABLE_YOLO
    if HAVE_YOLO:
        yolo_model = YOLO(Config.YOLO_WEIGHTS)
        print(f"✅ YOLO model loaded: {Config.YOLO_WEIGHTS}")
    else:
        yolo_model = None
except ImportError:
    HAVE_YOLO = False
    yolo_model = None
    print("⚠️  YOLO not available")

# OpenCV for image processing
try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False
    print("⚠️  opencv not available")

# ==================== HELPER FUNCTIONS ====================

def classify_document_type(text: str) -> str:
    """Classify document based on text content with enhanced logic"""
    if not text or len(text.strip()) == 0:
        return "Unknown"
    
    txt = text.upper()
    
    # PAN Card - Check FIRST (most distinctive pattern)
    if re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", txt):
        return "PAN"
    if any(keyword in txt for keyword in ["INCOME TAX", "PERMANENT ACCOUNT", "GOVT. OF INDIA INCOME TAX"]):
        return "PAN"
    
    # Aadhaar - 12 digit pattern with keywords
    if re.search(r"\b\d{4}\s*\d{4}\s*\d{4}\b", txt):
        if any(keyword in txt for keyword in ["AADHAAR", "AADHAR", "UNIQUE IDENTIFICATION", "UIDAI", "VID", "GOVT. OF INDIA"]):
            return "Aadhaar"
    
    # Driving Licence
    if any(keyword in txt for keyword in ["DRIVING LICENCE", "DRIVING LICENSE", "TRANSPORT AUTHORITY", "MOTOR VEHICLES", "FORM OF DRIVING"]):
        return "Driving Licence"
    if re.search(r"\bDL\s*NO\b", txt) or (re.search(r"\bISSUE\s*DATE\b", txt) and re.search(r"\bVALID", txt)):
        return "Driving Licence"
    
    # Voter ID
    if any(keyword in txt for keyword in ["ELECTION COMMISSION", "ELECTOR", "EPIC NO", "VOTER"]):
        return "Voter ID"
    if re.search(r"\b[A-Z]{3}[0-9]{7}\b", txt):
        return "Voter ID"
    
    # Marksheet
    if any(keyword in txt for keyword in ["MARKSHEET", "MARKS MEMO", "GRADE POINT", "CGPA", "BOARD OF", "S.S.C", "H.S.C", "EXAMINATION", "CUMULATIVE GRADE"]):
        return "Marksheet"
    
    return "Unknown"


def safe_split_lines(text: str) -> List[str]:
    """Split text into lines safely"""
    return [ln.strip() for ln in re.split(r"[\r\n]+", text or "") if ln.strip()]


def normalize_name(s: str) -> Optional[str]:
    """Normalize name by removing extra spaces and punctuation"""
    if not s:
        return None
    s = re.sub(r'\s+', ' ', s).strip()
    return re.sub(r'[:\-]+$', '', s).strip()


def clean_value(val: str) -> Optional[str]:
    """Remove unwanted punctuation/spaces"""
    if not val:
        return None
    return re.sub(r"^[\s,.:;_\-]+|[\s,.:;_\-]+$", "", val).strip()


def is_probable_name(text: str) -> bool:
    """Check if text looks like a person's name"""
    if not text:
        return False
    return (
        bool(re.fullmatch(r"[A-Za-z .'-]+", text)) and
        3 < len(text) < 50 and
        not any(kw in text.lower() for kw in ['government', 'india', 'authority', 
                'unique', 'identification', 'number', 'aadhaar', 'address', 
                'pin', 'code', 'signature', 'enrolment', 'mobile'])
    )


def is_probable_address_line(text: str) -> bool:
    """Check if text looks like an address line"""
    if not text or len(text) <= 4:
        return False
    return (
        re.search(r'[A-Za-z]', text) and
        not re.fullmatch(r'[0-9 /:,.-]+', text) and
        not any(kw in text.lower() for kw in ['aadhaar', 'signature', 'mobile', 
                'government', 'unique', 'identification', 'enrolment', 
                'your aadhaar no', 'vid', 'pin code'])
    )


def flatten_doctr_blocks(blocks: List[List[str]]) -> List[str]:
    """Flatten doctr block structure to list of lines"""
    out = []
    for block in blocks or []:
        if isinstance(block, (list, tuple)):
            out.extend([line.strip() for line in block if line and isinstance(line, str)])
        elif isinstance(block, str):
            out.append(block.strip())
    return out


# ==================== OCR FUNCTIONS ====================

def extract_image_ocr(img_bytes: bytes) -> Tuple[str, Optional[dict], Image.Image, List[str]]:
    """
    Perform OCR on image bytes
    Returns: (tesseract_text, tesseract_data, pil_image, doctr_lines)
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    # Tesseract OCR
    try:
        tess_text = pytesseract.image_to_string(img)
        tess_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    except Exception as e:
        print(f"Tesseract OCR Error: {e}")
        tess_text, tess_data = "", None

    # doctr OCR (if available)
    doctr_lines = []
    if HAVE_DOCTR and ocr_model is not None:
        try:
            doc = DocumentFile.from_images([img_bytes])
            result = ocr_model(doc)
            blocks = []
            for page in result.pages:
                for block in page.blocks:
                    blocks.append([" ".join([w.value for w in line.words]) for line in block.lines])
            doctr_lines = flatten_doctr_blocks(blocks)
        except Exception as e:
            print(f"Doctr OCR Error: {e}")
            doctr_lines = []

    return tess_text, tess_data, img, doctr_lines


def pdf_bytes_to_images(pdf_bytes: bytes, dpi=300) -> List[Tuple[bytes, int]]:
    """
    Convert PDF to images
    Returns: List of (image_bytes, page_number)
    """
    if not HAVE_PDF2IMAGE:
        raise RuntimeError("pdf2image not installed")
    
    images = []
    try:
        pil_pages = convert_from_bytes(pdf_bytes, dpi=dpi)
        for i, pil in enumerate(pil_pages):
            bio = io.BytesIO()
            pil.save(bio, format='PNG')
            images.append((bio.getvalue(), i + 1))
    except Exception as e:
        print(f"PDF conversion error: {e}")
        raise
    
    return images


def extract_pdf_content(pdf_bytes: bytes) -> Tuple[str, List[List[str]]]:
    """
    Extract text and tables from PDF using pdfplumber
    Returns: (text, tables)
    """
    if not HAVE_PDFPLUMBER:
        return "", []
    
    tempf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        tempf.write(pdf_bytes)
        tempf.flush()
        tempf.close()
        
        text = ""
        tables = []
        
        with pdfplumber.open(tempf.name) as pdf:
            page_texts = []
            for page in pdf.pages:
                page_texts.append(page.extract_text() or "")
                try:
                    for pt in page.extract_tables():
                        tables.append(pt)
                except Exception:
                    continue
            text = "\n".join(page_texts)
        
        return text, tables
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return "", []
    finally:
        try:
            os.unlink(tempf.name)
        except:
            pass


# ==================== YOLO INTEGRATION ====================

def run_yolo_on_pil_image(pil_img: Image.Image, conf_threshold: float = 0.25) -> List[Dict[str, Any]]:
    """
    Run YOLO on PIL Image and return detected boxes
    Returns: List of {'x1', 'y1', 'x2', 'y2', 'conf', 'cls'}
    """
    if not HAVE_YOLO or yolo_model is None:
        return []
    
    try:
        arr = np.array(pil_img)[:,:,::-1]  # RGB to BGR
        results = yolo_model(arr, imgsz=1024, conf=conf_threshold)
        boxes_out = []
        
        r = results[0]
        if hasattr(r, 'boxes') and r.boxes is not None:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                boxes_out.append({
                    'x1': int(xyxy[0]),
                    'y1': int(xyxy[1]),
                    'x2': int(xyxy[2]),
                    'y2': int(xyxy[3]),
                    'conf': conf,
                    'cls': cls
                })
        return boxes_out
    except Exception as e:
        print(f"YOLO error: {e}")
        return []


def expand_box(x1, y1, x2, y2, img_w, img_h, pad=0.02):
    """Expand box by padding and clip to image boundaries"""
    bw = x2 - x1
    bh = y2 - y1
    pad_w = int(bw * pad)
    pad_h = int(bh * pad)
    nx1 = max(0, x1 - pad_w)
    ny1 = max(0, y1 - pad_h)
    nx2 = min(img_w, x2 + pad_w)
    ny2 = min(img_h, y2 + pad_h)
    return nx1, ny1, nx2, ny2


def sort_boxes_reading_order(boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort boxes top-to-bottom, left-to-right"""
    if not boxes:
        return []
    return sorted(boxes, key=lambda b: (b['y1'], b['x1']))


def ocr_crop(crop_img: np.ndarray) -> Tuple[str, dict]:
    """Run OCR on cropped image"""
    try:
        pil_crop = Image.fromarray(crop_img[:,:,::-1] if crop_img.shape[2] == 3 else crop_img)
    except:
        try:
            pil_crop = Image.fromarray(crop_img)
        except:
            return "", None
    
    try:
        text = pytesseract.image_to_string(pil_crop)
        tess_data = pytesseract.image_to_data(pil_crop, output_type=pytesseract.Output.DICT)
        return text, tess_data
    except:
        return "", None


def get_right_text(box, boxes, max_y_diff=40) -> str:
    """Find nearest text box to the right of a given label box"""
    x1, y1, x2, y2 = box['box']
    right_texts = []
    
    for other in boxes:
        if 'text' in other:
            ox1, oy1, ox2, oy2 = other['box']
            if ox1 > x1 and abs(oy1 - y1) < max_y_diff:
                right_texts.append((ox1, other['text']))
    
    right_texts.sort(key=lambda x: x[0])
    if right_texts:
        return clean_value(right_texts[0][1])
    return None


# ==================== FIELD EXTRACTORS ====================

def extract_aadhaar_fields(text: str, yolo_output: dict = None) -> dict:
    """Extract Aadhaar card fields"""
    import re
    import string

    fields = {
        "aadhaar_number": None,
        "name": None,
        "dob": None,
        "gender": None,
        "father_name": None,
        "address": None,
        "mobile": None,
    }

    def clean_candidate(s):
        return s.strip(" .-–—")

    def is_address_junk(line):
        if not line or not re.search(r'[A-Za-z0-9]', line):
            return True
        if re.search(r'\b\d{4}\b', line):
            if len(line.strip()) == 4:
                return True
        if re.search(r'@', line):
            return True
        if re.search(r'www\.', line):
            return True
        if line.lower().startswith('help@') or line.lower().startswith('info@'):
            return True
        if all(c in string.punctuation for c in line.strip()):
            return True
        return False

    def is_really_likely_name(candidate):
        parts = [p for p in candidate.split() if len(p) > 2]
        if len(parts) < 2:
            return False
        for p in parts:
            if not p[0].isupper():
                return False
        if re.fullmatch(r"(.)\1*", candidate.replace(" ", "")):
            return False
        vowel_count = sum(1 for c in candidate.lower() if c in 'aeiou')
        if vowel_count < 2:
            return False
        if candidate.lower() in ['wwr ore', 'unknown', 'sample']:
            return False
        return True

    def is_probable_name(text: str) -> bool:
        if not text:
            return False
        if re.fullmatch(r"[-–—]+", text):
            return False
        if len(text.strip(" .'-")) < 3:
            return False
        return (
            bool(re.fullmatch(r"[A-Za-z .'-]+", text)) and
            3 < len(text) < 50 and
            not any(kw in text.lower() for kw in ['government', 'india', 'authority', 
                    'unique', 'identification', 'number', 'aadhaar', 'address', 
                    'pin', 'code', 'signature', 'enrolment', 'mobile'])
        )

    all_texts = []
    if yolo_output:
        for page in yolo_output.get("pages", []):
            for crop in page.get('crops', []):
                text_crop = crop.get('text', '').strip()
                if text_crop:
                    all_texts.append(text_crop)

    combined_text = '\n'.join(all_texts) + '\n' + (text or '')
    lines = [ln.strip() for ln in combined_text.splitlines() if ln.strip()]

    if not lines:
        return fields

    # Aadhaar Number
    for line in lines:
        m = re.search(r'\b(\d{4})\s*(\d{4})\s*(\d{4})\b', line)
        if m:
            fields['aadhaar_number'] = ''.join(m.groups())
            break
        m = re.search(r'\b(\d{12})\b', line)
        if m:
            fields['aadhaar_number'] = m.group(1)
            break

    # DOB
    for line in lines:
        m = re.search(r'\b(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4})\b', line)
        if m:
            fields['dob'] = m.group(1)
            break

    # Gender
    for line in lines:
        if re.search(r'\b(male|female|transgender)\b', line, re.I):
            m = re.search(r'\b(male|female|transgender)\b', line, re.I)
            fields['gender'] = m.group(1).title()
            break

    # Mobile
    for line in lines:
        m = re.search(r'(?:^|[^\d])([6-9]\d{9})(?:[^\d]|$)', line)
        if m:
            fields['mobile'] = m.group(1)
            break
    if not fields['mobile']:
        for line in lines:
            m = re.search(r'([6-9][0-9]{2}\s*[0-9]{3}\s*[0-9]{4})', line.replace(' ', ''))
            if m:
                fields['mobile'] = m.group(1).replace(' ', '')
                break

    # Name and Father Name extraction
    relationship_idx = None
    relationship_line = None

    for i, line in enumerate(lines):
        if re.search(r'\b(D/O|S/O|W/O|C/O)\b', line, re.I):
            relationship_idx = i
            relationship_line = line

            patterns = [
                r'(?:D/O|S/O|W/O|C/O)\s*[:\-]?\s*([A-Za-z\s.]{3,80})',
                r'(?:D/O|S/O|W/O|C/O)[:\s]+([A-Z][A-Za-z\s.]+)'
            ]
            for pat in patterns:
                m = re.search(pat, line, re.I)
                if m:
                    father_candidate = m.group(1).strip(" ,.:;")
                    father_candidate = re.sub(r'\s+', ' ', father_candidate)
                    father_candidate = re.sub(r'[^A-Za-z\s.]', '', father_candidate)
                    father_candidate = clean_candidate(father_candidate)
                    if len(father_candidate) > 3 and is_probable_name(father_candidate) and not re.fullmatch(r"[-–—]+", father_candidate) and is_really_likely_name(father_candidate):
                        fields['father_name'] = father_candidate
                        break
            break

    if relationship_idx is not None:
        for offset in range(1, 4):
            check_idx = relationship_idx - offset
            if check_idx >= 0:
                candidate = clean_candidate(lines[check_idx])
                if (is_probable_name(candidate) and 
                    len(candidate) > 3 and 
                    not re.fullmatch(r"[-–—]+", candidate) and
                    not any(kw in candidate.lower() for kw in ['year', 'birth', 'male', 'female', 'enrol', 'vid', 'government']) and
                    is_really_likely_name(candidate)):
                    fields['name'] = candidate
                    break

    if not fields['name'] and fields['dob']:
        for i, line in enumerate(lines):
            if fields['dob'] in line:
                for j in range(max(0, i - 3), i):
                    candidate = clean_candidate(lines[j])
                    if is_probable_name(candidate) and len(candidate) > 3 and not re.fullmatch(r"[-–—]+", candidate) and is_really_likely_name(candidate):
                        fields['name'] = candidate
                        break
                break

    if not fields['name'] and fields['aadhaar_number']:
        for i, line in enumerate(lines):
            if fields['aadhaar_number'] in line.replace(" ", ""):
                for j in range(max(0, i - 3), i):
                    candidate = clean_candidate(lines[j])
                    if is_probable_name(candidate) and len(candidate) > 3 and not re.fullmatch(r"[-–—]+", candidate) and is_really_likely_name(candidate):
                        fields['name'] = candidate
                        break
                break

    if not fields['name']:
        skip_keywords = ['government', 'india', 'authority', 'unique', 'identification', 'aadhaar', 'year', 'birth']
        for line in lines:
            candidate = clean_candidate(line)
            if (is_probable_name(candidate) and 
                len(candidate) > 3 and 
                not re.fullmatch(r"[-–—]+", candidate) and
                not any(s in candidate.lower() for s in skip_keywords) and
                is_really_likely_name(candidate)):
                fields['name'] = candidate
                break

    if not fields['father_name'] and fields['name']:
        name_indices = [i for i, l in enumerate(lines) if fields['name'].strip().lower() == clean_candidate(l).strip().lower()]
        for ni in name_indices:
            for offset in range(1, 4):
                if ni+offset < len(lines):
                    possible = clean_candidate(lines[ni+offset])
                    if (is_probable_name(possible) and 
                        possible.lower() != fields['name'].lower() and
                        len(possible) > 3 and not re.fullmatch(r"[-–—]+", possible) and is_really_likely_name(possible)):
                        fields['father_name'] = possible
                        break
            if fields['father_name']:
                break

    if relationship_idx is not None:
        addr_parts = []
        for i in range(relationship_idx + 1, len(lines)):
            line = clean_candidate(lines[i])
            if not line or re.fullmatch(r"[-–—]+", line):
                continue
            if fields['mobile'] and fields['mobile'] in line:
                break
            if re.search(r'\b(signature|aadhaar\s+no|vid|government|enrolment|download)', line, re.I):
                break
            clean_line = line
            if fields['father_name'] and fields['father_name'] in clean_line:
                clean_line = clean_line.replace(fields['father_name'], '').strip(' ,')
            if (len(clean_line) > 3 and 
                re.search(r'[A-Za-z]', clean_line) and
                not re.search(r'\b(male|female|gender|dob|birth)\b', clean_line, re.I)):
                addr_parts.append(clean_line)
        addr_parts = [l for l in addr_parts if not is_address_junk(l)]
        if addr_parts:
            fields['address'] = ', '.join(addr_parts).strip(' ,')

    if not fields['address']:
        vtc_indices = [i for i, line in enumerate(lines) if re.search(r'\bVTC\b', line, re.I)]
        for vtc_idx in vtc_indices:
            address_candidates = []
            for j in range(max(0, vtc_idx-5), vtc_idx):
                l = clean_candidate(lines[j])
                if (len(l) > 3 and 
                    re.search(r'[A-Za-z]', l) and
                    not re.search(r'\b(aadhaar|male|female|government|dob)\b', l, re.I) and
                    not re.fullmatch(r"[-–—]+", l) and
                    not is_address_junk(l)):
                    address_candidates.append(l)
            if address_candidates:
                fields['address'] = ', '.join(address_candidates).strip(' ,')
                break

    return fields


def extract_pan_fields(text: str, rawdata: bool = False) -> dict:
    """Extract PAN card fields"""
    import re

    fields = {"pan": None, "name": None, "father_name": None, "dob": None}

    def clean_candidate(s):
        return s.strip(" .-–—")

    def is_probable_name(text: str) -> bool:
        if not text:
            return False
        if re.fullmatch(r"[-–—]+", text):
            return False
        if len(text.strip(" .'-")) < 3:
            return False
        return (
            bool(re.fullmatch(r"[A-Za-z .'-]+", text)) and
            3 < len(text) < 50 and
            not any(kw in text.lower() for kw in [
                'government', 'india', 'authority', 'unique', 'identification',
                'number', 'aadhaar', 'address', 'pin', 'code', 'signature',
                'enrolment', 'mobile'])
        )

    def normalize_name(s: str):
        if not s:
            return None
        s = re.sub(r'[^\w\s]', '', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def is_pan_real_name(candidate):
        words = candidate.split()
        # Allow names up to 6 words (previously max 3)
        if len(words) < 1 or len(words) > 6:
            return False
        # Require at least one reasonably long token (>=3 chars)
        if not any(len(w) >= 3 for w in words):
            return False
        if not any(w[0].isupper() for w in words if len(w) > 1):
            return False
        if not any(w.isalpha() for w in words):
            return False
        if re.fullmatch(r"(.)\1*", candidate.replace(" ", "")):
            return False
        if all(len(w) <= 2 for w in words):
            return False
        if candidate.lower() in ['unknown', 'sample']:
            return False
        return True

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    upper_lines = [ln.upper() for ln in lines]
    up = text.upper()

    # PAN Number
    m = re.search(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b", up)
    if m:
        fields["pan"] = m.group(1)

    # DOB
    m = re.search(r"(DOB|DATE OF BIRTH)[:\s]*([0-9]{2}[/-][0-9]{2}[/-][0-9]{4})", up)
    if m:
        fields["dob"] = m.group(2)
    else:
        m = re.search(r"\b[0-9]{2}[/-][0-9]{2}[/-][0-9]{4}\b", text)
        if m:
            fields["dob"] = m.group(0)

    # Name extraction
    name_found = False
    for i, ln in enumerate(lines):
        if re.search(r'name', ln, re.I) and not re.search(r'father', ln, re.I) and not name_found:
            if i+1 < len(lines):
                candidate = normalize_name(lines[i+1])
                candidate = clean_candidate(candidate)
                if candidate and len(candidate) > 2 and is_probable_name(candidate) and is_pan_real_name(candidate):
                    fields["name"] = candidate
                    name_found = True
            m = re.search(r'name\s*[:\-\/]?\s*(.+)', ln, re.I)
            if m and not name_found:
                candidate = normalize_name(m.group(1))
                candidate = clean_candidate(candidate)
                if candidate and len(candidate) > 2 and is_probable_name(candidate) and is_pan_real_name(candidate):
                    fields["name"] = candidate
                    name_found = True

    # Father's Name extraction (robust to splits, slashes, next line, etc)
    father_found = False
    for i, ln in enumerate(upper_lines):
        if "FATHER" in ln and not father_found:
            # Try direct next line first
            if i+1 < len(lines):
                next_candidate = normalize_name(lines[i+1])
                next_candidate = clean_candidate(next_candidate)
                if (
                    next_candidate
                    and len(next_candidate) > 2
                    and is_probable_name(next_candidate)
                    and is_pan_real_name(next_candidate)
                    and next_candidate != fields.get("name")
                ):
                    fields["father_name"] = next_candidate
                    father_found = True
            # Try to extract from current line, even if extra tokens before/after label
            if not father_found:
                idx = ln.find("FATHER")
                after_father = lines[i][idx:]
                m = re.search(r"FATHER'?S?\s*NAME\s*[:\-\/]?\s*(.+)", after_father, re.I)
                if m:
                    candidate = normalize_name(m.group(1))
                    candidate = clean_candidate(candidate)
                    if (
                        candidate
                        and len(candidate) > 2
                        and is_probable_name(candidate)
                        and is_pan_real_name(candidate)
                        and candidate != fields.get("name")
                    ):
                        fields["father_name"] = candidate
                        father_found = True
            # Try from current line mixing e.g. "... Father's Name <name>"
            if not father_found:
                m2 = re.search(r"FATHER'?S?\s*NAME\s*[:\-\/]?\s*(.+)", lines[i], re.I)
                if m2:
                    candidate = normalize_name(m2.group(1))
                    candidate = clean_candidate(candidate)
                    if (
                        candidate
                        and len(candidate) > 2
                        and is_probable_name(candidate)
                        and is_pan_real_name(candidate)
                        and candidate != fields.get("name")
                    ):
                        fields["father_name"] = candidate
                        father_found = True
            # Try split line where name might be on same line after "/ Father's Name"
            if not father_found and '/' in lines[i]:
                s = lines[i].split('/')
                for fragment_idx, fragment in enumerate(s):
                    if "father" in fragment.lower():
                        idxf = fragment_idx
                        if idxf + 1 < len(s):
                            candidate = normalize_name(s[idxf+1])
                            candidate = clean_candidate(candidate)
                            if (
                                candidate
                                and len(candidate) > 2
                                and is_probable_name(candidate)
                                and is_pan_real_name(candidate)
                                and candidate != fields.get("name")
                            ):
                                fields["father_name"] = candidate
                                father_found = True
                                break
            # Try window search for father's name within next 3 lines if not found
            if not father_found and i+3 < len(lines):
                for j in range(i+1, min(i+4, len(lines))):
                    candidate = normalize_name(lines[j])
                    candidate = clean_candidate(candidate)
                    if (
                        candidate
                        and len(candidate) > 2
                        and is_probable_name(candidate)
                        and is_pan_real_name(candidate)
                        and candidate != fields.get("name")
                    ):
                        fields["father_name"] = candidate
                        father_found = True
                        break

    # Positional fallback
    if not name_found or not father_found:
        dob_idx = None
        pan_idx = None

        for i, ln in enumerate(lines):
            if fields["dob"] and fields["dob"] in ln:
                dob_idx = i
            if fields["pan"] and fields["pan"] in ln.upper():
                pan_idx = i

        anchor = dob_idx if dob_idx is not None else pan_idx

        if anchor is not None:
            if not name_found and anchor >= 2:
                candidate = normalize_name(lines[anchor-2])
                candidate = clean_candidate(candidate)
                if candidate and len(candidate) > 2 and is_probable_name(candidate) and is_pan_real_name(candidate):
                    fields["name"] = candidate

            if not father_found and anchor >= 1:
                candidate = normalize_name(lines[anchor-1])
                candidate = clean_candidate(candidate)
                if (candidate and len(candidate) > 2 and
                    is_probable_name(candidate) and
                    is_pan_real_name(candidate) and
                    candidate != fields.get("name")):
                    fields["father_name"] = candidate

    # Final fallback
    if not fields["name"]:
        income_tax_idx = None
        father_idx = None
        for i, ln in enumerate(upper_lines):
            if "INCOME TAX" in ln and income_tax_idx is None:
                income_tax_idx = i
            if "FATHER" in ln and father_idx is None:
                father_idx = i
        if income_tax_idx is not None and father_idx is not None and father_idx > income_tax_idx:
            for i in range(income_tax_idx+1, father_idx):
                candidate = normalize_name(lines[i])
                candidate = clean_candidate(candidate)
                if candidate and len(candidate) > 2 and is_probable_name(candidate) and is_pan_real_name(candidate):
                    fields["name"] = candidate
                    break

    if rawdata:
        fields['rawdata'] = lines

    return fields


def extract_voter_fields(text: str, yolo_output: dict = None, rawdata: bool = False) -> dict:
    """Extract Voter ID fields"""
    import re

    fields = {
        "voter_id": None,
        "name": None,
        "father_name": None,
        "husband_name": None,
        "dob": None,
        "gender": None,
        "address": None,
    }

    def clean_candidate(s):
        return s.strip(" .-–—")

    def is_probable_name(text: str) -> bool:
        if not text:
            return False
        if re.fullmatch(r"[-–—]+", text):
            return False
        if len(text.strip(" .'-")) < 3:
            return False
        return (
            bool(re.fullmatch(r"[A-Za-z .'-]+", text)) and
            3 < len(text) < 50 and
            not any(kw in text.lower() for kw in [
                'government', 'india', 'authority', 'unique', 'identification',
                'number', 'aadhaar', 'address', 'pin', 'code', 'signature',
                'enrolment', 'mobile'])
        )

    def normalize_name(s: str):
        if not s:
            return None
        s = re.sub(r'[^\w\s]', '', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    if yolo_output:
        yolo_boxes = []
        for page in yolo_output.get("pages", []):
            if 'crops' in page:
                yolo_boxes.extend(page['crops'])

        if yolo_boxes:
            for box in yolo_boxes:
                text_lower = box.get('text', '').lower()
                if not text_lower:
                    continue

                if 'name' in text_lower and 'father' not in text_lower and 'husband' not in text_lower:
                    fields['name'] = fields['name'] or get_right_text(box, yolo_boxes)
                elif 'father' in text_lower:
                    fields['father_name'] = fields['father_name'] or get_right_text(box, yolo_boxes)
                elif 'husband' in text_lower:
                    fields['husband_name'] = fields['husband_name'] or get_right_text(box, yolo_boxes)
                elif 'birth' in text_lower:
                    fields['dob'] = fields['dob'] or get_right_text(box, yolo_boxes)
                elif 'gender' in text_lower:
                    fields['gender'] = fields['gender'] or get_right_text(box, yolo_boxes)
                elif 'epic no' in text_lower or ('epic' in text_lower and 'no' in text_lower):
                    fields['voter_id'] = fields['voter_id'] or get_right_text(box, yolo_boxes)

            if any(fields.values()):
                if rawdata:
                    fields['rawdata'] = [box.get('text', '') for box in yolo_boxes]
                return fields

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if rawdata:
        fields['rawdata'] = lines

    # Voter ID
    m = re.search(r"\b([A-Z]{3,4}[0-9]{6,10})\b", text)
    if not m:
        m = re.search(r"Epic no\.?\s*[:\-]?\s*([A-Z0-9]{6,20})", text, re.I)
    if m:
        fields["voter_id"] = m.group(1)
    if not fields["voter_id"]:
        for ln in lines:
            m = re.search(r"Epic.*?([A-Z0-9]{6,20})", ln, re.I)
            if m:
                fields["voter_id"] = m.group(1)
                break
            m = re.search(r"\b([A-Z]{3,4}[0-9]{6,10})\b", ln)
            if m:
                fields["voter_id"] = m.group(1)
                break

    # Name extraction
    m = re.search(r"Name[ ,:/-]*([A-Za-z .'-]+)", text, re.I)
    if m:
        candidate = normalize_name(m.group(1))
        candidate = clean_candidate(candidate)
        if candidate and is_probable_name(candidate):
            fields["name"] = candidate
    if not fields["name"]:
        m = re.search(r"Elector'?s Name\s*[:;+\-_]*\s*([A-Za-z .'-]+)", text, re.I)
        if m:
            candidate = normalize_name(m.group(1))
            candidate = clean_candidate(candidate)
            if candidate and is_probable_name(candidate):
                fields["name"] = candidate
        if not fields["name"]:
            for ln in lines:
                m = re.search(r"Elector'?s Name\s*[:;+\-_]*\s*([A-Za-z .'-]+)", ln, re.I)
                if m:
                    candidate = normalize_name(m.group(1))
                    candidate = clean_candidate(candidate)
                    if candidate and is_probable_name(candidate):
                        fields["name"] = candidate
                        break

    # Father's Name extraction
    if not fields["father_name"]:
        for ln in lines:
            m = re.search(r"Father'?s Name\s*[:;+\-_]*\s*([A-Za-z .'-]+)", ln, re.I)
            if m:
                candidate = m.group(1)
                candidate = re.sub(r'^[^A-Za-z]+', '', candidate)
                words = [w for w in candidate.split() if w.isalpha() and len(w) > 1]
                if words:
                    name_val = " ".join(words[:3])
                    name_val = normalize_name(name_val)
                    name_val = clean_candidate(name_val)
                    if is_probable_name(name_val):
                        fields["father_name"] = name_val
                        break
            if 'father' in ln.lower():
                split_parts = re.split(r"Father'?s Name", ln, flags=re.I)
                if len(split_parts) > 1:
                    after_label = split_parts[-1]
                    after_label = re.sub(r'^[^A-Za-z]+', '', after_label)
                    words = [w for w in after_label.split() if w.isalpha() and len(w) > 1]
                    if words:
                        name_val = " ".join(words[:3])
                        name_val = normalize_name(name_val)
                        name_val = clean_candidate(name_val)
                        if is_probable_name(name_val):
                            fields["father_name"] = name_val
                            break

    # Husband's Name
    m = re.search(r"Husband'?s Name[ ,:/-]*([A-Za-z .'-]+)", text, re.I)
    if m:
        candidate = normalize_name(m.group(1))
        candidate = clean_candidate(candidate)
        if candidate and is_probable_name(candidate):
            fields["husband_name"] = candidate
    if not fields["husband_name"]:
        m = re.search(r"Husband'?s Name\s*[:;+\-_]*\s*([A-Za-z .'-]+)", text, re.I)
        if m:
            candidate = normalize_name(m.group(1))
            candidate = clean_candidate(candidate)
            if candidate and is_probable_name(candidate):
                fields["husband_name"] = candidate
        if not fields["husband_name"]:
            for ln in lines:
                m = re.search(r"Husband'?s Name\s*[:;+\-_]*\s*([A-Za-z .'-]+)", ln, re.I)
                if m:
                    candidate = normalize_name(m.group(1))
                    candidate = clean_candidate(candidate)
                    if candidate and is_probable_name(candidate):
                        fields["husband_name"] = candidate
                        break

    # DOB extraction
    dob_patterns = [
        r"Date of Birth[ /:]*([0-9]{2}[-/][0-9]{2}[-/][0-9]{4})",
        r"([0-9]{2}[-/][0-9]{2}[-/][0-9]{4})"
    ]
    for pat in dob_patterns:
        m = re.search(pat, text, re.I)
        if m:
            fields["dob"] = m.group(1)
            break
    if not fields["dob"]:
        for ln in lines:
            m = re.search(r"Date of Birth\s*[:;+\-_]*\s*([0-9]{2}[-/][0-9]{2}[-/][0-9]{4})", ln, re.I)
            if m:
                fields["dob"] = m.group(1)
                break

    # Gender
    m = re.search(r"Gender.*?(Female|Male|Other)", text, re.I)
    if m:
        fields["gender"] = m.group(1).capitalize()
    if not fields["gender"]:
        m = re.search(r"(Sex|Gender)\s*[:;+\-_]*\s*(Male|Female|Other)", text, re.I)
        if m:
            fields["gender"] = m.group(2).capitalize()
        if not fields["gender"]:
            for ln in lines:
                m = re.search(r"(Sex|Gender)\s*[:;+\-_]*\s*(Male|Female|Other)", ln, re.I)
                if m:
                    fields["gender"] = m.group(2).capitalize()
                    break

    # Address extraction
    address_lines = []
    start_collect = False
    for i, ln in enumerate(lines):
        if ln.lower().startswith('address'):
            start_collect = True
            addr_part = re.sub(r'^address\s*[:\-]?\s*', '', ln, flags=re.I)
            if addr_part:
                address_lines.append(addr_part)
        elif start_collect:
            if re.match(r'^(name|father|husband|dob|gender|epic no|sex)', ln, re.I) or not ln:
                break
            address_lines.append(ln)

    if address_lines:
        address_lines = [clean_candidate(line) for line in address_lines if len(clean_candidate(line)) > 3]
        if address_lines:
            fields["address"] = ", ".join(address_lines).strip()

    return fields


def extract_dl_fields(text: str) -> Dict[str, Any]:
    """Extract Driving Licence fields"""
    fields = {
        "dl_number": None,
        "name": None,
        "dob": None,
        "issue_date": None,
        "valid_till": None,
        "father_name": None,
        "address": None,
    }
    
    if not text:
        return fields
    
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # DL Number
    for ln in lines:
        candidate = ln.replace(" ", "")
        m = re.search(r"\b([A-Z]{2}[0O]?\d{6,20})\b", candidate)
        if m:
            fields["dl_number"] = m.group(1)
            break
    
    if not fields["dl_number"]:
        m = re.search(r"\b([A-Z]{2}[0O]?\s*\d[\d\s]{5,20})\b", " ".join(lines))
        if m:
            fields["dl_number"] = m.group(1).replace(" ", "")

    # Name
    for ln in lines:
        if re.search(r"\bNAME\b", ln, re.I):
            m = re.search(r"Name\s*[:\-]?\s*(.+)", ln, re.I)
            if m:
                val = re.sub(r"Holder.?s Signature", "", m.group(1), flags=re.I).strip()
                fields["name"] = normalize_name(val)
                break

    # Father/Guardian
    for ln in lines:
        if re.search(r"\b(S/O|D/O|W/O|FATHER['']S NAME|SON OF|DAUGHTER OF|WIFE OF)\b", ln, re.I):
            m = re.search(r"(?:S/O|D/O|W/O|FATHER['']S NAME|SON OF|DAUGHTER OF|WIFE OF)[:\-]?\s*(.+)", ln, re.I)
            if m:
                fields["father_name"] = normalize_name(m.group(1))
                break

    # Address
    address_lines = []
    address_start_index = -1
    for i, ln in enumerate(lines):
        if "ADDRESS" in ln.upper():
            address_start_index = i
            addr_part = re.sub(r".*ADDRESS\s*[:\-]?\s*", "", ln, flags=re.I).strip()
            if addr_part:
                address_lines.append(addr_part)
            break
    
    if address_start_index != -1:
        for i in range(address_start_index + 1, len(lines)):
            line = lines[i].strip()
            if line:
                address_lines.append(line)
            elif address_lines:
                break
    
    if address_lines:
        fields["address"] = ", ".join(address_lines).strip()

    # Improved Date Extraction
    all_dates = re.findall(r"(\d{2}[/-]\d{2}[/-]\d{4})", text)
    unique_dates = sorted(set(all_dates), key=all_dates.index)

    # DOB with label
    dob_match = re.search(r"(?:Date of Birth|DOB)[\s:]*(\d{2}[/-]\d{2}[/-]\d{4})", text, re.I)
    if dob_match:
        fields["dob"] = dob_match.group(1)
        if fields["dob"] in unique_dates:
            unique_dates.remove(fields["dob"])

    # Issue Date with label
    issue_match = re.search(r"(?:Issue Date|Date of First Issue)[\s:]*(\d{2}[/-]\d{2}[/-]\d{4})", text, re.I)
    if issue_match:
        fields["issue_date"] = issue_match.group(1)
        if fields["issue_date"] in unique_dates:
            unique_dates.remove(fields["issue_date"])

    # Valid Till with label
    valid_match = re.search(r"(?:Validity|Valid Till)[\s\(\)A-Z]*[:\s]*(\d{2}[/-]\d{2}[/-]\d{4})", text, re.I)
    if valid_match:
        fields["valid_till"] = valid_match.group(1)
        if fields["valid_till"] in unique_dates:
            unique_dates.remove(fields["valid_till"])

    # Assign remaining dates
    if not fields["issue_date"] and unique_dates:
        fields["issue_date"] = unique_dates.pop(0)
    if not fields["valid_till"] and unique_dates:
        fields["valid_till"] = unique_dates.pop(0)

    return fields


def clean_subject(subj: str) -> str:
    """Clean subject name from marksheet"""
    if not subj:
        return None
    s = subj.upper()
    s = re.sub(r"\b(FIRST|SECOND|THIRD|FOURTH|FIFTH|LANGUAGE|CURRICULAR|CO-CURRICULAR|AREA|VALUE|EDUCATION|WORK|&|AND|THE|SUBJECT|SUBJECTS|GRADE|POINT|CODE)\b", ' ', s)
    s = re.sub(r"[\(\)\:\-\|,\.\\/]", ' ', s)
    s = re.sub(r"\s+", ' ', s).strip()
    tokens = [t for t in s.split(' ') if t]
    for tok in reversed(tokens):
        if len(tok) >= 3 and tok.isalpha():
            return tok.title()
    return s.title() if s else None


def parse_table_from_pdf_tables(pdf_tables: List[List[List[str]]]) -> List[Dict[str, Any]]:
    """Parse subject tables from PDF tables"""
    results = []
    if not pdf_tables:
        return results
    
    for tbl in pdf_tables:
        if not tbl or not isinstance(tbl, list):
            continue
        
        header_idx = None
        for i, row in enumerate(tbl[:5]):
            joined = " ".join([str(c).upper() for c in row if c])
            if any(h in joined for h in ['SUBJECT', 'GRADE', 'MARKS', 'POINT', 'SCORE', 'GRADE POINT']):
                header_idx = i
                break
        
        if header_idx is not None:
            headers = [str(c).strip().lower() for c in tbl[header_idx]]
            col_map = {}
            for ci, h in enumerate(headers):
                if 'subject' in h or 'course' in h or 'paper' in h:
                    col_map['subject'] = ci
                elif 'grade' in h:
                    col_map['grade'] = ci
                elif 'point' in h or 'marks' in h or 'score' in h or 'total' in h:
                    col_map['marks'] = ci
                elif 'max' in h and 'marks' not in col_map:
                    col_map['max_marks'] = ci
            
            for row in tbl[header_idx+1:]:
                try:
                    subj = row[col_map['subject']].strip() if 'subject' in col_map and len(row) > col_map['subject'] else ''
                    grade = row[col_map['grade']].strip() if 'grade' in col_map and len(row) > col_map['grade'] else ''
                    marks = row[col_map['marks']].strip() if 'marks' in col_map and len(row) > col_map['marks'] else ''
                    if subj or grade or marks:
                        results.append({
                            'subject': clean_subject(subj),
                            'grade': grade,
                            'marks': marks
                        })
                except:
                    continue
        else:
            for row in tbl:
                row_join = ' '.join([str(c) for c in row if c])
                grade_m = re.search(r"\bA[1-4]\b|\bA1\b|\bA2\b|\bB\b|\bC\b|\bD\b|\bE\b|\bF\b", row_join, re.I)
                marks_m = re.search(r"\b[0-9]{1,3}\b", row_join)
                if grade_m and marks_m:
                    subj = row_join[:grade_m.start()].strip()
                    results.append({
                        'subject': clean_subject(subj),
                        'grade': grade_m.group(0),
                        'marks': marks_m.group(0)
                    })
    
    return results


def parse_table_from_lines(lines: List[str]) -> List[Dict[str, Any]]:
    """Parse subject tables from text lines"""
    results = []
    if not lines:
        return results
    
    up_lines = [ln.upper() for ln in lines]
    grade_regex = re.compile(r"\bA[1-4]\b|\bA1\b|\bA2\b|\bB\b|\bC\b|\bD\b|\bE\b|\bF\b", re.I)
    marks_regex = re.compile(r"\b([0-9]{1,3})(?:\.\d+)?\b")
    used_indices = set()
    n = len(lines)
    
    for i in range(n):
        if i in used_indices:
            continue
        
        window = ' '.join([lines[j] for j in range(i, min(i+4, n))])
        up_window = window.upper()
        g = grade_regex.search(up_window)
        m = marks_regex.search(up_window)
        
        if g and m:
            subj_text = window[:g.start()].strip()
            subj_clean = clean_subject(subj_text)
            grade = g.group(0).strip()
            marks = m.group(1).strip()
            if subj_clean:
                results.append({
                    'subject': subj_clean,
                    'grade': grade,
                    'marks': marks
                })
                for k in range(i, min(i+4, n)):
                    used_indices.add(k)
                continue
        
        if i+2 < n:
            g2 = grade_regex.search(up_lines[i+1])
            m2 = marks_regex.search(up_lines[i+2])
            if g2 and m2:
                subj_clean = clean_subject(lines[i])
                grade = g2.group(0).strip()
                marks = m2.group(1).strip()
                if subj_clean:
                    results.append({
                        'subject': subj_clean,
                        'grade': grade,
                        'marks': marks
                    })
                    used_indices.update([i, i+1, i+2])
                    continue
    
    seen = set()
    dedup = []
    for r in results:
        key = (r.get('subject', '').upper(), r.get('grade', ''), r.get('marks', ''))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)
    
    return dedup


def extract_marksheet_fields(text: str, filename: str = None, meta: Dict[str, Any] = None) -> Dict[str, Any]:
    """Extract Marksheet fields"""
    import re
    fields = {
        "student_name": None,
        "father_name": None,
        "mother_name": None,
        "school_name": None,
        "dob": None,
        "roll_no": None,
        "year": None,
        "cgpa": None,
        "subjects": []
    }
    
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # School Name
    for ln in lines:
        m = re.match(r"SCHOOL\s*[:\-–—]?\s*(.+)", ln, re.I)
        if m:
            fields["school_name"] = m.group(0).strip()
            break
    
    if not fields["school_name"]:
        for ln in lines:
            if re.search(r'\b(SCHOOL|INSTITUTE|COLLEGE|ACADEMY|HIGHER\s*SECONDARY|SR\.?\s*SEC\.?)\b', ln, re.I):
                fields["school_name"] = ln.strip()
                break

    # Roll Number
    for i, ln in enumerate(lines):
        if re.search(r"\bROLL\s*NO\b|\bROLL\b", ln, re.I):
            m = re.search(r"\bROLL\s*(?:NO)?\.\s*[:\-–—]?\s*([0-9]{7,12})\b", ln, re.I)
            if m and "\\" not in m.group(1) and "/" not in m.group(1):
                fields["roll_no"] = m.group(1)
                break
            elif i + 1 < len(lines):
                candidate = lines[i + 1].strip()
                if re.match(r"^[0-9]{7,12}$", candidate) and "\\" not in candidate and "/" not in candidate:
                    fields["roll_no"] = candidate
                    break
    
    if not fields["roll_no"]:
        for ln in lines:
            m = re.search(r"\b([0-9]{8,12})\b", ln)
            if m and "\\" not in m.group(1) and "/" not in m.group(1):
                fields["roll_no"] = m.group(1)
                break

    # DOB
    dob_patterns = [
        r"\b(?:DOB|D\.O\.B\.?|DATE\s*OF\s*BIRTH|BIRTH\s*DATE)[\s:\-–—]*([0-3]?\d[\/\-.][01]?\d[\/\-.]\d{4})\b",
        r"\b([0-3]?\d[\/\-.][01]?\d[\/\-.]\d{4})\b"
    ]
    for pat in dob_patterns:
        m = re.search(pat, text, re.I)
        if m:
            fields["dob"] = m.group(1)
            break

    # Year
    for ln in lines:
        m = re.search(r"EXAMINATION\s+held\s+in\s+\w+-?(20\d{2})", ln, re.I)
        if m:
            fields["year"] = m.group(1)
            break
    
    if not fields["year"]:
        for ln in lines:
            m = re.search(r"(\b20\d{2}\b)", ln)
            if m and "DATE OF BIRTH" not in ln.upper() and "DOB" not in ln.upper():
                fields["year"] = m.group(1)
                break

    # CGPA
    cgpa_match = re.search(r"(CGPA|C\.G\.P\.A|GPA|G\.P\.A|GRADE\s*POINT)[\s\.:;\-–—]*([0-9]{1,2}\.[0-9]{1,2})", text, re.I)
    if cgpa_match:
        fields["cgpa"] = cgpa_match.group(2)
    else:
        cgpa_match2 = re.search(r"\b([0-9]{1,2}\.[0-9]{1,2})\b", text)
        if cgpa_match2:
            fields["cgpa"] = cgpa_match2.group(1)

    # Name Extraction
    for i, line in enumerate(lines):
        if re.search(r"\b(REGULAR|ROLL|PC\/)\b", line, re.I):
            if i + 1 < len(lines) and re.search(r"CERTIFIED\s+THAT\s+([A-Z\s]+)", lines[i+1], re.I):
                m = re.search(r"CERTIFIED\s+THAT\s+([A-Z\s]+)", lines[i+1], re.I)
                fields["student_name"] = m.group(1).strip() if m else lines[i+1].strip()
            elif i + 1 < len(lines):
                fields["student_name"] = lines[i+1].strip()
            
            if i + 2 < len(lines) and re.search(r"FATHER'?S\s+NAME\s+([A-Z\s]+)", lines[i+2], re.I):
                m = re.search(r"FATHER'?S\s+NAME\s+([A-Z\s]+)", lines[i+2], re.I)
                fields["father_name"] = m.group(1).strip() if m else lines[i+2].strip()
            elif i + 2 < len(lines):
                fields["father_name"] = lines[i+2].strip()
            
            if i + 3 < len(lines) and re.search(r"MOTHER'?S\s+NAME\s+([A-Z\s]+)", lines[i+3], re.I):
                m = re.search(r"MOTHER'?S\s+NAME\s+([A-Z\s]+)", lines[i+3], re.I)
                fields["mother_name"] = m.group(1).strip() if m else lines[i+3].strip()
            elif i + 3 < len(lines):
                fields["mother_name"] = lines[i+3].strip()
            break

    if not fields["student_name"]:
        for ln in lines:
            m = re.search(r"CERTIFIED THAT\s*:?\s*([A-Z][A-Z\s]+)", ln, re.I)
            if m:
                candidate = m.group(1).strip()
                if len(candidate) > 2 and not candidate.startswith("FATHER") and not candidate.startswith("MOTHER"):
                    fields["student_name"] = candidate
                    break

    return fields


# ==================== MAIN PROCESSOR ====================

def process_document(filename: str, content_bytes: bytes) -> Dict[str, Any]:
    """Main function to process uploaded document"""
    result = {
        'filename': filename,
        'document_type': None,
        'fields': {},
        'table': [],
        'raw_text_preview': "",
        'confidence': 0.0,
        'meta': {}
    }
    
    try:
        is_pdf = filename.lower().endswith('.pdf')
        
        # PDF PROCESSING
        if is_pdf:
            full_text = ""
            all_doctr_lines = []
            pdf_tables = []
            yolo_meta = {'pages': []}
            
            if HAVE_PDFPLUMBER:
                try:
                    text_from_pdf, tables_from_pdf = extract_pdf_content(content_bytes)
                    full_text = text_from_pdf or ""
                    if tables_from_pdf:
                        pdf_tables = tables_from_pdf
                except Exception as e:
                    print(f"PDF extraction error: {e}")
            
            if HAVE_PDF2IMAGE:
                try:
                    page_images = pdf_bytes_to_images(content_bytes, dpi=Config.OCR_DPI)
                    all_text_pages = []
                    
                    for img_bytes, page_no in page_images:
                        tess_text, tess_data, pil_img, doctr_lines = extract_image_ocr(img_bytes)
                        all_text_pages.append(tess_text or "")
                        if doctr_lines:
                            all_doctr_lines.extend(doctr_lines)
                        
                        if HAVE_YOLO:
                            try:
                                boxes = run_yolo_on_pil_image(pil_img, conf_threshold=0.2)
                                crops_info = []
                                img_w, img_h = pil_img.size
                                img_bgr = np.array(pil_img)[:,:,::-1]
                                
                                for b in sort_boxes_reading_order(boxes):
                                    x1, y1, x2, y2 = int(b['x1']), int(b['y1']), int(b['x2']), int(b['y2'])
                                    nx1, ny1, nx2, ny2 = expand_box(x1, y1, x2, y2, img_w, img_h, pad=0.03)
                                    
                                    crop = img_bgr[ny1:ny2, nx1:nx2].copy()
                                    ttxt, td = ocr_crop(crop)
                                    
                                    crops_info.append({
                                        'page': page_no,
                                        'box': {'x1': nx1, 'y1': ny1, 'x2': nx2, 'y2': ny2},
                                        'conf': float(b.get('conf', 0.0)),
                                        'text': ttxt
                                    })
                                
                                yolo_meta['pages'].append({
                                    'page_no': page_no,
                                    'boxes': boxes,
                                    'crops': crops_info
                                })
                            except Exception as e:
                                print(f"YOLO error on page {page_no}: {e}")
                    
                    ocr_text = "\n".join(all_text_pages)
                    if len(ocr_text) > len(full_text):
                        full_text = ocr_text
                    
                except Exception as e:
                    print(f"PDF to image conversion error: {e}")
            
            lines = (all_doctr_lines or []) + safe_split_lines(full_text)
        
        # IMAGE PROCESSING
        else:
            tess_text, tess_data, img, doctr_lines = extract_image_ocr(content_bytes)
            full_text = tess_text or ""
            lines = (doctr_lines or []) + safe_split_lines(tess_text)
            pdf_tables = []
            yolo_meta = {'pages': []}
            
            if HAVE_YOLO:
                try:
                    boxes = run_yolo_on_pil_image(img, conf_threshold=0.2)
                    crops_info = []
                    img_w, img_h = img.size
                    img_bgr = np.array(img)[:,:,::-1]
                    
                    for b in sort_boxes_reading_order(boxes):
                        x1, y1, x2, y2 = int(b['x1']), int(b['y1']), int(b['x2']), int(b['y2'])
                        nx1, ny1, nx2, ny2 = expand_box(x1, y1, x2, y2, img_w, img_h, pad=0.03)
                        
                        crop = img_bgr[ny1:ny2, nx1:nx2].copy()
                        ttxt, td = ocr_crop(crop)
                        
                        crops_info.append({
                            'page': 1,
                            'box': {'x1': nx1, 'y1': ny1, 'x2': nx2, 'y2': ny2},
                            'conf': float(b.get('conf', 0.0)),
                            'text': ttxt
                        })
                    
                    yolo_meta['pages'].append({
                        'page_no': 1,
                        'boxes': boxes,
                        'crops': crops_info
                    })
                except Exception as e:
                    print(f"YOLO error: {e}")
        
        # DEBUG: Print OCR text preview
        print(f"\n=== OCR TEXT PREVIEW ===")
        print(full_text[:500] if full_text else "NO TEXT EXTRACTED")
        print(f"=== END PREVIEW ===\n")
        
        # DOCUMENT CLASSIFICATION
        doc_type = classify_document_type(full_text)
        print(f"🔍 Detected Document Type: {doc_type}")
        result['document_type'] = doc_type
        
        # FIELD EXTRACTION
        if doc_type == "PAN":
            result['fields'] = extract_pan_fields(full_text)
        
        elif doc_type == "Aadhaar":
            result['fields'] = extract_aadhaar_fields(full_text, yolo_meta if HAVE_YOLO else None)
        
        elif doc_type == "Voter ID":
            result['fields'] = extract_voter_fields(full_text, yolo_meta if HAVE_YOLO else None)
        
        elif doc_type == "Driving Licence":
            result['fields'] = extract_dl_fields(full_text)
        
        elif doc_type == "Marksheet":
            meta = {"pdf_tables": pdf_tables} if pdf_tables else {}
            result['fields'] = extract_marksheet_fields(full_text, filename, meta)
            
            table_entries = parse_table_from_pdf_tables(pdf_tables) if pdf_tables else []
            if not table_entries:
                table_entries = parse_table_from_lines(lines)
            if table_entries:
                result['table'] = table_entries
        
        else:
            # Unknown - try marksheet parser as fallback
            meta = {"pdf_tables": pdf_tables} if pdf_tables else {}
            result['fields'] = extract_marksheet_fields(full_text, filename, meta)
        
        # Calculate confidence
        total_fields = len(result['fields'])
        filled_fields = sum(1 for v in result['fields'].values() if v)
        result['confidence'] = (filled_fields / total_fields * 100) if total_fields > 0 else 0
        
        # Add text preview
        result['raw_text_preview'] = "\n".join(full_text.splitlines()[:30])
        
        # Store full raw text
        result['raw_text_full'] = full_text
        
        # Add YOLO meta
        if HAVE_YOLO and yolo_meta['pages']:
            result['meta']['yolo'] = yolo_meta
        
        return result
    
    except Exception as e:
        print(f"❌ Processing error: {e}")
        import traceback
        traceback.print_exc()
        result['error'] = str(e)
        return result


if __name__ == "__main__":
    print("✅ Complete Extractor Service loaded successfully")
    print(f"   - PDF2Image: {'✅' if HAVE_PDF2IMAGE else '❌'}")
    print(f"   - PDFPlumber: {'✅' if HAVE_PDFPLUMBER else '❌'}")
    print(f"   - docTR: {'✅' if HAVE_DOCTR else '❌'}")
    print(f"   - YOLO: {'✅' if HAVE_YOLO else '❌'}")
    print(f"   - OpenCV: {'✅' if HAVE_CV2 else '❌'}")