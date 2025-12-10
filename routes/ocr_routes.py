"""
OCR Routes - Split Architecture Version
Light API tries first, calls Heavy API if needed
"""
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from services.extractor import process_document
from services.database import get_db
from services.file_storage import get_storage
from services.confidence_calculator import process_with_confidence, add_extraction_summary
from services.models import ScanResponse, RescanResponse, SubmissionResponse
from config import Config
import requests
import os

ocr_blueprint = Blueprint("ocr", __name__)

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

# Heavy API URL (will be set via environment variable)
HEAVY_API_URL = os.getenv('HEAVY_API_URL', None)
APP_MODE = os.getenv('APP_MODE', 'light')  # 'light' or 'heavy'

# Confidence threshold for calling heavy API
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '70'))

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def call_heavy_api(file_bytes, filename, auto_submit=False):
    """Call heavy API for better extraction"""
    if not HEAVY_API_URL:
        print("⚠️  Heavy API URL not configured")
        return None
    
    try:
        print(f"🔄 Calling Heavy API: {HEAVY_API_URL}")
        
        files = {'file': (filename, file_bytes, 'application/pdf')}
        data = {'auto_submit': 'true' if auto_submit else 'false'}
        
        response = requests.post(
            f"{HEAVY_API_URL}/api/scan",
            files=files,
            data=data,
            timeout=60  # 60 second timeout
        )
        
        if response.status_code == 200:
            print("✅ Heavy API returned success")
            return response.json()
        else:
            print(f"❌ Heavy API error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Heavy API call failed: {e}")
        return None


# ==================== SCAN ENDPOINT ====================

@ocr_blueprint.route("/scan", methods=["POST"])
def scan_document():
    """
    Scan document with smart routing:
    1. Try with light API (Tesseract only)
    2. If confidence < threshold, call heavy API
    """
    try:
        if "file" not in request.files:
            response = ScanResponse(success=False, error="No file uploaded")
            return jsonify(response.to_dict()), 400
        
        file = request.files["file"]
        
        if file.filename == "":
            response = ScanResponse(success=False, error="Empty filename")
            return jsonify(response.to_dict()), 400
        
        if not allowed_file(file.filename):
            response = ScanResponse(success=False, error=f"Invalid file type")
            return jsonify(response.to_dict()), 400
        
        auto_submit = request.form.get('auto_submit', 'false').lower() == 'true'
        
        # Read file bytes
        filename = secure_filename(file.filename)
        file_bytes = file.read()
        
        print(f"📄 Processing scan: {filename} (Mode: {APP_MODE})")
        
        # STEP 1: Try with current API (light or heavy)
        result = process_document(filename, file_bytes)
        result = process_with_confidence(result)
        result = add_extraction_summary(result)
        
        confidence = result.get("overall_confidence") or result.get("confidence", 0.0)
        
        print(f"📊 Confidence: {confidence}%")
        
        # STEP 2: If this is LIGHT mode and confidence is low, try HEAVY API
        if APP_MODE == 'light' and confidence < CONFIDENCE_THRESHOLD and HEAVY_API_URL:
            print(f"⚠️  Low confidence ({confidence}%), trying Heavy API...")
            
            heavy_result = call_heavy_api(file_bytes, filename, auto_submit)
            
            if heavy_result and heavy_result.get('success'):
                # Use heavy API result
                print("✅ Using Heavy API result")
                return jsonify(heavy_result), 200
            else:
                print("⚠️  Heavy API failed, using Light API result")
                # Continue with light result below
        
        # STEP 3: Save to database
        db = get_db()
        scan_id = db.save_scan(result)
        
        # Save file for rescan (if enabled)
        storage_metadata = {}
        if Config.ENABLE_FILE_STORAGE:
            storage = get_storage()
            storage_metadata = storage.save_file(scan_id, file_bytes, filename)
            db.update_scan(scan_id, {'storage_metadata': storage_metadata})
            print(f"💾 File stored for rescan: {storage_metadata.get('storage_mode')}")
        
        # Build response
        response = ScanResponse(
            success=True,
            scan_id=scan_id,
            filename=result.get("filename"),
            document_type=result.get("document_type"),
            fields=result.get("fields", {}),
            confidence=confidence,
            extraction_summary=result.get("extraction_summary", {}),
            message=f"Document scanned successfully ({APP_MODE} mode)"
        )
        
        # Auto-submit if requested
        if auto_submit and result.get("fields"):
            submission_data = {
                'scan_id': scan_id,
                'document_type': result.get("document_type"),
                'verified_fields': result.get("fields", {}),
                'user_corrections': {},
                'final_confidence': confidence,
                'extraction_summary': result.get("extraction_summary", {})
            }
            submission_id = db.save_submission(submission_data)
            response.submission_id = submission_id
            response.auto_submitted = True
            response.message = f"Document scanned and submitted automatically ({APP_MODE} mode)"
        
        return jsonify(response.to_dict()), 200
    
    except Exception as e:
        print(f"❌ Scan error: {e}")
        response = ScanResponse(success=False, error=str(e))
        return jsonify(response.to_dict()), 500


# ==================== RESCAN ENDPOINT ====================

@ocr_blueprint.route("/rescan/<scan_id>", methods=["POST"])
def rescan_document(scan_id: str):
    """Rescan document - uses stored file"""
    try:
        db = get_db()
        scan = db.get_scan(scan_id)
        if not scan:
            response = RescanResponse(success=False, error="Original scan not found")
            return jsonify(response.to_dict()), 404
        
        storage_metadata = scan.get('storage_metadata', {})
        
        # Get stored file or new upload
        if storage_metadata.get('stored') and "file" not in request.files:
            print(f"📄 Rescanning using stored file for: {scan_id}")
            storage = get_storage()
            file_bytes = storage.get_file(scan_id, storage_metadata)
            
            if not file_bytes:
                response = RescanResponse(
                    success=False, 
                    error="Stored file not found. Please upload file again."
                )
                return jsonify(response.to_dict()), 404
            
            filename = storage_metadata.get('filename', 'stored_file.pdf')
        
        elif "file" in request.files:
            print(f"📤 Rescanning with new uploaded file")
            file = request.files["file"]
            
            if file.filename == "":
                response = RescanResponse(success=False, error="Empty filename")
                return jsonify(response.to_dict()), 400
                
            if not allowed_file(file.filename):
                response = RescanResponse(success=False, error="Invalid file type")
                return jsonify(response.to_dict()), 400
            
            filename = secure_filename(file.filename)
            file_bytes = file.read()
        
        else:
            response = RescanResponse(
                success=False,
                error="No file available. Original file not stored or no new file uploaded."
            )
            return jsonify(response.to_dict()), 400
        
        auto_submit = request.form.get('auto_submit', 'false').lower() == 'true'
        
        # Process document
        print(f"📄 Reprocessing document...")
        result = process_document(filename, file_bytes)
        result = process_with_confidence(result)
        result = add_extraction_summary(result)
        
        confidence = result.get("overall_confidence") or result.get("confidence", 0.0)
        
        # Try heavy API if needed
        if APP_MODE == 'light' and confidence < CONFIDENCE_THRESHOLD and HEAVY_API_URL:
            print(f"⚠️  Low confidence on rescan ({confidence}%), trying Heavy API...")
            
            heavy_result = call_heavy_api(file_bytes, filename, auto_submit)
            
            if heavy_result and heavy_result.get('success'):
                print("✅ Using Heavy API result for rescan")
                return jsonify(heavy_result), 200
        
        # Save rescan
        rescan_id = db.save_rescan(result, scan_id)
        
        # Build response
        response = RescanResponse(
            success=True,
            rescan_id=rescan_id,
            scan_id=scan_id,
            filename=result.get("filename"),
            document_type=result.get("document_type"),
            fields=result.get("fields", {}),
            confidence=confidence,
            extraction_summary=result.get("extraction_summary", {}),
            message=f"Document rescanned successfully ({APP_MODE} mode)"
        )
        
        # Auto-submit if requested
        if auto_submit and result.get("fields"):
            submission_data = {
                'scan_id': scan_id,
                'rescan_id': rescan_id,
                'document_type': result.get("document_type"),
                'verified_fields': result.get("fields", {}),
                'user_corrections': {},
                'final_confidence': confidence,
                'extraction_summary': result.get("extraction_summary", {})
            }
            submission_id = db.save_submission(submission_data)
            response.submission_id = submission_id
            response.auto_submitted = True
            response.message = f"Document rescanned and submitted automatically ({APP_MODE} mode)"
        
        return jsonify(response.to_dict()), 200
    
    except Exception as e:
        print(f"❌ Rescan error: {e}")
        import traceback
        traceback.print_exc()
        response = RescanResponse(success=False, error=str(e))
        return jsonify(response.to_dict()), 500


# ==================== SUBMIT ENDPOINT ====================

@ocr_blueprint.route("/submit/<scan_id>", methods=["POST"])
def submit_document(scan_id: str):
    """Submit document and optionally cleanup stored file"""
    try:
        db = get_db()
        scan = db.get_scan(scan_id)
        
        if not scan:
            response = SubmissionResponse(success=False, error="Scan not found")
            return jsonify(response.to_dict()), 404
        
        data = request.get_json() or {}
        
        verified_fields = data.get('verified_fields') or scan.get('fields', {})
        extraction_summary = data.get('extraction_summary') or scan.get('extraction_summary', {})
        
        submission_data = {
            'scan_id': scan_id,
            'rescan_id': data.get('rescan_id'),
            'document_type': data.get('document_type') or scan.get('document_type'),
            'verified_fields': verified_fields,
            'table': data.get('table') or scan.get('table', []),
            'user_corrections': data.get('user_corrections', {}),
            'final_confidence': data.get('confidence') or scan.get('overall_confidence') or scan.get('confidence', 0.0),
            'extraction_summary': extraction_summary
        }
        
        submission_id = db.save_submission(submission_data)
        
        # Cleanup stored file (optional)
        cleanup = request.args.get('cleanup', 'true').lower() == 'true'
        if cleanup:
            storage_metadata = scan.get('storage_metadata', {})
            if storage_metadata.get('stored'):
                storage = get_storage()
                if storage.delete_file(scan_id, storage_metadata):
                    print(f"🗑️ Cleaned up stored file for: {scan_id}")
        
        print(f"✅ Submission saved: {submission_id}")
        
        response_data = {
            "success": True,
            "submission_id": submission_id,
            "scan_id": scan_id,
            "status": "submitted",
            "verified_fields": verified_fields,
            "table": scan.get('table', []),
            "document_type": submission_data['document_type'],
            "message": "Document submitted successfully"
        }
        
        return jsonify(response_data), 200
    
    except Exception as e:
        print(f"❌ Submit error: {e}")
        response = SubmissionResponse(success=False, error=str(e))
        return jsonify(response.to_dict()), 500


# ==================== DOCUMENTATION ====================

@ocr_blueprint.route("/docs", methods=["GET"])
def docs():
    """API Documentation"""
    return jsonify({
        "api_version": "3.0.0 - Split Architecture",
        "description": "OCR API with Smart Routing (Light + Heavy)",
        "mode": APP_MODE,
        "heavy_api_configured": HEAVY_API_URL is not None,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "workflow": {
            "light_mode": "Scans with Tesseract. If confidence < threshold, calls Heavy API.",
            "heavy_mode": "Uses YOLO + docTR for maximum accuracy."
        },
        "endpoints": {
            "/api/scan": {
                "method": "POST",
                "description": "Scan document with smart routing",
                "parameters": {
                    "file": "Document file (required)",
                    "auto_submit": "true/false (optional)"
                }
            },
            "/api/rescan/<scan_id>": {
                "method": "POST",
                "description": "Rescan document"
            },
            "/api/submit/<scan_id>": {
                "method": "POST",
                "description": "Submit verified data"
            }
        }
    }), 200


# ==================== HEALTH CHECK ====================

@ocr_blueprint.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    try:
        db = get_db()
        stats = db.get_statistics()
        
        return jsonify({
            "status": "healthy",
            "service": "OCR API",
            "version": "3.0.0",
            "mode": APP_MODE,
            "heavy_api": "configured" if HEAVY_API_URL else "not_configured",
            "database": {
                "status": "connected",
                "total_scans": stats.get('total_scans', 0),
                "total_submissions": stats.get('total_submissions', 0)
            },
            "features": {
                "yolo": Config.ENABLE_YOLO,
                "file_storage": Config.ENABLE_FILE_STORAGE
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500