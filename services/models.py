"""
MongoDB Document Models/Schemas - Updated with Extraction Summary
Defines the structure of data stored in MongoDB collections
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum


# ==================== ENUMS ====================

class DocumentType(str, Enum):
    """Supported document types"""
    PAN = "PAN"
    AADHAAR = "Aadhaar"
    VOTER_ID = "Voter ID"
    DRIVING_LICENCE = "Driving Licence"
    MARKSHEET = "Marksheet"
    UNKNOWN = "Unknown"


class ScanStatus(str, Enum):
    """Scan processing status"""
    SCANNED = "scanned"
    SUBMITTED = "submitted"
    PROCESSING = "processing"
    FAILED = "failed"


class SubmissionStatus(str, Enum):
    """Submission status"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    VERIFIED = "verified"
    REJECTED = "rejected"


# ==================== FIELD MODELS ====================

@dataclass
class FieldValue:
    """Individual field with confidence score"""
    value: Any
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "confidence": self.confidence
        }


@dataclass
class PanFields:
    """PAN Card fields"""
    pan: Optional[str] = None
    name: Optional[str] = None
    father_name: Optional[str] = None
    dob: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class AadhaarFields:
    """Aadhaar Card fields"""
    aadhaar_number: Optional[str] = None
    name: Optional[str] = None
    dob: Optional[str] = None
    gender: Optional[str] = None
    father_name: Optional[str] = None
    address: Optional[str] = None
    mobile: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class VoterIdFields:
    """Voter ID fields"""
    voter_id: Optional[str] = None
    name: Optional[str] = None
    father_name: Optional[str] = None
    husband_name: Optional[str] = None
    dob: Optional[str] = None
    gender: Optional[str] = None
    address: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class DrivingLicenceFields:
    """Driving Licence fields"""
    dl_number: Optional[str] = None
    name: Optional[str] = None
    dob: Optional[str] = None
    issue_date: Optional[str] = None
    valid_till: Optional[str] = None
    father_name: Optional[str] = None
    address: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class MarksheetFields:
    """Marksheet fields"""
    student_name: Optional[str] = None
    father_name: Optional[str] = None
    mother_name: Optional[str] = None
    school_name: Optional[str] = None
    dob: Optional[str] = None
    roll_no: Optional[str] = None
    year: Optional[str] = None
    cgpa: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class SubjectGrade:
    """Subject with grade and marks"""
    subject: Optional[str] = None
    grade: Optional[str] = None
    marks: Optional[str] = None
    max_marks: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


# ==================== MAIN DOCUMENT MODELS ====================

@dataclass
class ScanDocument:
    """
    Main Scan Document Model
    Stored in 'scans' collection
    """
    scan_id: str
    filename: str
    document_type: str
    fields: Dict[str, Any]
    confidence: float
    status: str = ScanStatus.SCANNED.value
    table: List[Dict[str, Any]] = field(default_factory=list)
    raw_text_preview: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    extraction_summary: Dict[str, Any] = field(default_factory=dict)
    rescan_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to MongoDB document"""
        return {
            "scan_id": self.scan_id,
            "filename": self.filename,
            "document_type": self.document_type,
            "fields": self.fields,
            "table": self.table,
            "confidence": self.confidence,
            "raw_text_preview": self.raw_text_preview,
            "meta": self.meta,
            "extraction_summary": self.extraction_summary,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "rescan_count": self.rescan_count
        }
    
    @classmethod
    def from_extraction(cls, scan_id: str, extraction_result: Dict[str, Any]) -> 'ScanDocument':
        """Create ScanDocument from extraction result"""
        return cls(
            scan_id=scan_id,
            filename=extraction_result.get('filename', ''),
            document_type=extraction_result.get('document_type', DocumentType.UNKNOWN.value),
            fields=extraction_result.get('fields', {}),
            table=extraction_result.get('table', []),
            confidence=extraction_result.get('confidence', 0.0),
            raw_text_preview=extraction_result.get('raw_text_preview', ''),
            meta=extraction_result.get('meta', {}),
            extraction_summary=extraction_result.get('extraction_summary', {})
        )


@dataclass
class RescanDocument:
    """
    Rescan Document Model
    Stored in 'rescans' collection
    """
    rescan_id: str
    original_scan_id: str
    filename: str
    document_type: str
    fields: Dict[str, Any]
    confidence: float
    table: List[Dict[str, Any]] = field(default_factory=list)
    raw_text_preview: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    extraction_summary: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to MongoDB document"""
        return {
            "rescan_id": self.rescan_id,
            "original_scan_id": self.original_scan_id,
            "filename": self.filename,
            "document_type": self.document_type,
            "fields": self.fields,
            "table": self.table,
            "confidence": self.confidence,
            "raw_text_preview": self.raw_text_preview,
            "meta": self.meta,
            "extraction_summary": self.extraction_summary,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_extraction(cls, rescan_id: str, original_scan_id: str, 
                       extraction_result: Dict[str, Any]) -> 'RescanDocument':
        """Create RescanDocument from extraction result"""
        return cls(
            rescan_id=rescan_id,
            original_scan_id=original_scan_id,
            filename=extraction_result.get('filename', ''),
            document_type=extraction_result.get('document_type', DocumentType.UNKNOWN.value),
            fields=extraction_result.get('fields', {}),
            table=extraction_result.get('table', []),
            confidence=extraction_result.get('confidence', 0.0),
            raw_text_preview=extraction_result.get('raw_text_preview', ''),
            meta=extraction_result.get('meta', {}),
            extraction_summary=extraction_result.get('extraction_summary', {})
        )


@dataclass
class SubmissionDocument:
    """
    Submission Document Model
    Stored in 'submissions' collection
    """
    submission_id: str
    scan_id: str
    document_type: str
    verified_fields: Dict[str, Any]
    final_confidence: float
    rescan_id: Optional[str] = None
    user_corrections: Dict[str, Any] = field(default_factory=dict)
    extraction_summary: Dict[str, Any] = field(default_factory=dict)
    status: str = SubmissionStatus.SUBMITTED.value
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to MongoDB document"""
        return {
            "submission_id": self.submission_id,
            "scan_id": self.scan_id,
            "rescan_id": self.rescan_id,
            "document_type": self.document_type,
            "verified_fields": self.verified_fields,
            "user_corrections": self.user_corrections,
            "final_confidence": self.final_confidence,
            "extraction_summary": self.extraction_summary,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_submission_data(cls, submission_id: str, submission_data: Dict[str, Any]) -> 'SubmissionDocument':
        """Create SubmissionDocument from submission data"""
        return cls(
            submission_id=submission_id,
            scan_id=submission_data.get('scan_id'),
            rescan_id=submission_data.get('rescan_id'),
            document_type=submission_data.get('document_type'),
            verified_fields=submission_data.get('verified_fields', {}),
            user_corrections=submission_data.get('user_corrections', {}),
            final_confidence=submission_data.get('final_confidence', 0.0),
            extraction_summary=submission_data.get('extraction_summary', {})
        )


# ==================== RESPONSE MODELS ====================

@dataclass
class ScanResponse:
    """API Response for scan operation"""
    success: bool
    scan_id: Optional[str] = None
    filename: Optional[str] = None
    document_type: Optional[str] = None
    fields: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None
    extraction_summary: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    error: Optional[str] = None
    submission_id: Optional[str] = None
    auto_submitted: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"success": self.success}
        if self.scan_id:
            result["scan_id"] = self.scan_id
        if self.filename:
            result["filename"] = self.filename
        if self.document_type:
            result["document_type"] = self.document_type
        if self.fields is not None:
            result["fields"] = self.fields
        if self.confidence is not None:
            result["confidence"] = self.confidence
        if self.extraction_summary is not None:
            result["extraction_summary"] = self.extraction_summary
        if self.message:
            result["message"] = self.message
        if self.error:
            result["error"] = self.error
        if self.submission_id:
            result["submission_id"] = self.submission_id
        if self.auto_submitted:
            result["auto_submitted"] = self.auto_submitted
        return result


@dataclass
class RescanResponse:
    """API Response for rescan operation"""
    success: bool
    rescan_id: Optional[str] = None
    scan_id: Optional[str] = None
    filename: Optional[str] = None
    document_type: Optional[str] = None
    fields: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None
    extraction_summary: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    error: Optional[str] = None
    submission_id: Optional[str] = None
    auto_submitted: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"success": self.success}
        if self.rescan_id:
            result["rescan_id"] = self.rescan_id
        if self.scan_id:
            result["scan_id"] = self.scan_id
        if self.filename:
            result["filename"] = self.filename
        if self.document_type:
            result["document_type"] = self.document_type
        if self.fields is not None:
            result["fields"] = self.fields
        if self.confidence is not None:
            result["confidence"] = self.confidence
        if self.extraction_summary is not None:
            result["extraction_summary"] = self.extraction_summary
        if self.message:
            result["message"] = self.message
        if self.error:
            result["error"] = self.error
        if self.submission_id:
            result["submission_id"] = self.submission_id
        if self.auto_submitted:
            result["auto_submitted"] = self.auto_submitted
        return result


@dataclass
class SubmissionResponse:
    """API Response for submission operation"""
    success: bool
    submission_id: Optional[str] = None
    scan_id: Optional[str] = None
    status: Optional[str] = None
    verified_fields: Optional[Dict[str, Any]] = None
    extraction_summary: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"success": self.success}
        if self.submission_id:
            result["submission_id"] = self.submission_id
        if self.scan_id:
            result["scan_id"] = self.scan_id
        if self.status:
            result["status"] = self.status
        if self.verified_fields is not None:
            result["verified_fields"] = self.verified_fields
        if self.extraction_summary is not None:
            result["extraction_summary"] = self.extraction_summary
        if self.message:
            result["message"] = self.message
        if self.error:
            result["error"] = self.error
        return result


# ==================== SCHEMA DOCUMENTATION ====================

MONGODB_SCHEMA = {
    "scans": {
        "description": "Initial document scans",
        "indexes": [
            {"keys": [("scan_id", 1)], "unique": True},
            {"keys": [("document_type", 1)]},
            {"keys": [("created_at", -1)]},
            {"keys": [("status", 1)]}
        ],
        "example": {
            "scan_id": "uuid-string",
            "filename": "aadhaar.pdf",
            "document_type": "Aadhaar",
            "fields": {
                "aadhaar_number": {"value": "1234 5678 9012", "confidence": 95},
                "name": {"value": "JOHN DOE", "confidence": 88}
            },
            "table": [],
            "confidence": 85,
            "raw_text_preview": "First 30 lines...",
            "meta": {"yolo": {}, "processing_time": 1.5},
            "extraction_summary": {
                "overall_confidence": 92.8,
                "high_confidence": {
                    "count": 7,
                    "fields": ["aadhaar_number", "name", "dob"]
                },
                "medium_confidence": {
                    "count": 3,
                    "fields": ["father_name"]
                },
                "low_confidence": {
                    "count": 1,
                    "fields": ["mobile"]
                }
            },
            "status": "scanned",
            "created_at": "2025-10-28T10:30:00Z",
            "updated_at": "2025-10-28T10:30:00Z",
            "rescan_count": 0
        }
    },
    "rescans": {
        "description": "Rescanned documents for improved accuracy",
        "indexes": [
            {"keys": [("rescan_id", 1)], "unique": True},
            {"keys": [("original_scan_id", 1)]},
            {"keys": [("created_at", -1)]}
        ],
        "example": {
            "rescan_id": "uuid-string",
            "original_scan_id": "original-uuid",
            "filename": "aadhaar_better.pdf",
            "document_type": "Aadhaar",
            "fields": {
                "aadhaar_number": {"value": "1234 5678 9012", "confidence": 98}
            },
            "table": [],
            "confidence": 92,
            "raw_text_preview": "Better scan...",
            "meta": {},
            "extraction_summary": {
                "overall_confidence": 95.5,
                "high_confidence": {"count": 8},
                "medium_confidence": {"count": 2},
                "low_confidence": {"count": 0}
            },
            "created_at": "2025-10-28T10:35:00Z"
        }
    },
    "submissions": {
        "description": "Final submitted and verified data",
        "indexes": [
            {"keys": [("submission_id", 1)], "unique": True},
            {"keys": [("scan_id", 1)]},
            {"keys": [("created_at", -1)]},
            {"keys": [("status", 1)]}
        ],
        "example": {
            "submission_id": "uuid-string",
            "scan_id": "scan-uuid",
            "rescan_id": "rescan-uuid",
            "document_type": "Aadhaar",
            "verified_fields": {
                "aadhaar_number": "1234 5678 9012",
                "name": "JOHN DOE"
            },
            "user_corrections": {
                "name": {"old": "JOHN D0E", "new": "JOHN DOE"}
            },
            "final_confidence": 95,
            "extraction_summary": {
                "overall_confidence": 95.5,
                "high_confidence": {"count": 8},
                "medium_confidence": {"count": 2},
                "low_confidence": {"count": 0}
            },
            "status": "submitted",
            "created_at": "2025-10-28T10:40:00Z",
            "updated_at": "2025-10-28T10:40:00Z"
        }
    }
}


# ==================== UTILITY FUNCTIONS ====================

def validate_document_type(doc_type: str) -> bool:
    """Validate if document type is supported"""
    return doc_type in [t.value for t in DocumentType]


def get_field_schema(doc_type: str) -> Dict[str, type]:
    """Get field schema for document type"""
    schemas = {
        DocumentType.PAN.value: PanFields,
        DocumentType.AADHAAR.value: AadhaarFields,
        DocumentType.VOTER_ID.value: VoterIdFields,
        DocumentType.DRIVING_LICENCE.value: DrivingLicenceFields,
        DocumentType.MARKSHEET.value: MarksheetFields
    }
    return schemas.get(doc_type)


def print_schema_documentation():
    """Print MongoDB schema documentation"""
    print("=" * 80)
    print("MONGODB SCHEMA DOCUMENTATION")
    print("=" * 80)
    for collection, info in MONGODB_SCHEMA.items():
        print(f"\nCollection: {collection}")
        print(f"Description: {info['description']}")
        print(f"Indexes: {len(info['indexes'])}")
        print(f"Example Document:")
        import json
        print(json.dumps(info['example'], indent=2, default=str))
        print("-" * 80)


if __name__ == "__main__":
    print_schema_documentation()