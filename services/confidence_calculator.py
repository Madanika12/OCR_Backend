"""
Field-level Confidence Calculator - Simplified
Calculates confidence scores for each extracted field (percentages)
"""

import re
from typing import Dict, Any, Optional

def calculate_field_confidence(field_name: str, field_value: Any, document_type: str, 
                                raw_text: str = "") -> int:
    """
    Calculate confidence score for a single field
    Returns: 0 to 100 (percentage)
    """
    if field_value is None or (isinstance(field_value, str) and not field_value.strip()):
        return 0
    
    value_str = str(field_value).strip()
    
    # Base confidence by field type (internally as fraction 0.0-1.0)
    confidence = 0.0
    
    # ========== NUMBER FIELDS (High confidence if valid pattern) ==========
    
    if field_name == "aadhaar_number":
        # Aadhaar: 12 digits
        if re.fullmatch(r'\d{12}', value_str.replace(' ', '')):
            confidence = 0.98
        elif re.fullmatch(r'\d{10,14}', value_str.replace(' ', '')):
            confidence = 0.70  # Close but not exact
        else:
            confidence = 0.30
    
    elif field_name == "pan":
        # PAN: 5 letters, 4 digits, 1 letter
        if re.fullmatch(r'[A-Z]{5}[0-9]{4}[A-Z]', value_str):
            confidence = 0.98
        elif re.fullmatch(r'[A-Z0-9]{10}', value_str):
            confidence = 0.65
        else:
            confidence = 0.25
    
    elif field_name == "voter_id":
        # Voter ID: 3-4 letters + 6-10 digits
        if re.fullmatch(r'[A-Z]{3,4}[0-9]{6,10}', value_str):
            confidence = 0.95
        elif re.fullmatch(r'[A-Z0-9]{9,15}', value_str):
            confidence = 0.60
        else:
            confidence = 0.30
    
    elif field_name == "dl_number":
        # DL: State code (2 letters) + digits
        if re.fullmatch(r'[A-Z]{2}[0-9O]{6,20}', value_str):
            confidence = 0.95
        elif re.search(r'[A-Z]{2}', value_str) and re.search(r'\d{6,}', value_str):
            confidence = 0.70
        else:
            confidence = 0.35
    
    elif field_name == "mobile":
        # Mobile: 10 digits starting with 6-9
        if re.fullmatch(r'[6-9]\d{9}', value_str):
            confidence = 0.97
        elif re.fullmatch(r'\d{10}', value_str):
            confidence = 0.60
        else:
            confidence = 0.25
    
    elif field_name == "roll_no":
        # Roll number: 7-12 digits
        if re.fullmatch(r'\d{7,12}', value_str):
            confidence = 0.92
        elif re.fullmatch(r'\d{5,15}', value_str):
            confidence = 0.70
        else:
            confidence = 0.40
    
    # ========== DATE FIELDS ==========
    
    elif field_name in ["dob", "issue_date", "valid_till"]:
        # Date: DD/MM/YYYY or DD-MM-YYYY
        if re.fullmatch(r'\d{1,2}[/-]\d{1,2}[/-]\d{4}', value_str):
            # Validate date ranges
            parts = re.split(r'[/-]', value_str)
            day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
            
            if 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2100:
                confidence = 0.95
            else:
                confidence = 0.50  # Format OK but values suspicious
        else:
            confidence = 0.30
    
    # ========== GENDER FIELD ==========
    
    elif field_name == "gender":
        if value_str.lower() in ["male", "female", "transgender", "m", "f"]:
            confidence = 0.99
        else:
            confidence = 0.40
    
    # ========== NAME FIELDS ==========
    
    elif field_name in ["name", "father_name", "mother_name", "student_name"]:
        # Name should be alphabetic with spaces
        # Length check
        if len(value_str) < 3:
            confidence = 0.20
        elif len(value_str) > 50:
            confidence = 0.40
        else:
            # Character composition check
            alpha_ratio = sum(c.isalpha() or c.isspace() for c in value_str) / len(value_str)
            
            if alpha_ratio >= 0.90:
                # Good name pattern
                word_count = len(value_str.split())
                if 2 <= word_count <= 5:
                    confidence = 0.88
                elif word_count == 1:
                    confidence = 0.70  # Single name less confident
                else:
                    confidence = 0.65  # Too many words
            elif alpha_ratio >= 0.70:
                confidence = 0.50  # Some non-alphabetic chars
            else:
                confidence = 0.25  # Too much noise
            
            # Penalize if contains common OCR errors
            if re.search(r'[|_\[\]{}]', value_str):
                confidence *= 0.70
            
            # Penalize very short fragments
            if any(len(word) == 1 for word in value_str.split()):
                confidence *= 0.80
    
    # ========== ADDRESS FIELD ==========
    
    elif field_name == "address":
        if len(value_str) < 10:
            confidence = 0.30  # Too short for address
        elif len(value_str) > 200:
            confidence = 0.50  # Suspiciously long
        else:
            # Good address should have mix of letters, numbers, spaces
            has_letters = bool(re.search(r'[A-Za-z]', value_str))
            has_numbers = bool(re.search(r'\d', value_str))
            has_comma = ',' in value_str
            
            if has_letters and has_numbers and has_comma:
                confidence = 0.85
            elif has_letters and (has_numbers or has_comma):
                confidence = 0.70
            elif has_letters:
                confidence = 0.55
            else:
                confidence = 0.30
    
    # ========== SCHOOL/INSTITUTION NAME ==========
    
    elif field_name == "school_name":
        if len(value_str) < 5:
            confidence = 0.30
        elif len(value_str) > 100:
            confidence = 0.40
        else:
            # Should contain keywords like SCHOOL, COLLEGE, etc.
            if re.search(r'\b(SCHOOL|COLLEGE|INSTITUTE|ACADEMY|UNIVERSITY)\b', value_str, re.I):
                confidence = 0.90
            else:
                confidence = 0.60
    
    # ========== CGPA/MARKS ==========
    
    elif field_name == "cgpa":
        # CGPA typically 0.0 to 10.0
        try:
            cgpa_val = float(value_str)
            if 0.0 <= cgpa_val <= 10.0:
                confidence = 0.92
            elif 0.0 <= cgpa_val <= 100.0:
                confidence = 0.60  # Might be percentage
            else:
                confidence = 0.30
        except:
            confidence = 0.20
    
    elif field_name == "year":
        # Year should be 4 digits, 1900-2100
        if re.fullmatch(r'(19|20)\d{2}', value_str):
            confidence = 0.95
        elif re.fullmatch(r'\d{4}', value_str):
            confidence = 0.60
        else:
            confidence = 0.25
    
    # ========== DEFAULT FOR OTHER FIELDS ==========
    
    else:
        # Generic confidence based on content
        if len(value_str) > 0:
            # Basic presence gives base confidence
            confidence = 0.60
            
            # Adjust based on length
            if len(value_str) < 2:
                confidence = 0.30
            elif len(value_str) > 100:
                confidence = 0.50
        else:
            confidence = 0.0
    
    # Convert fractional confidence (0.0-1.0) to percentage (0-100) and return integer
    return int(round(confidence * 100))


def add_confidence_to_fields(fields: Dict[str, Any], document_type: str, 
                              raw_text: str = "") -> Dict[str, Dict[str, Any]]:
    """
    Transform flat fields dict into confidence-annotated structure
    
    Input:  {"name": "John Doe", "dob": "01/01/1990"}
    Output: {
        "name": {
            "value": "John Doe",
            "confidence": 88
        },
        "dob": {
            "value": "01/01/1990",
            "confidence": 95
        }
    }
    """
    annotated_fields = {}
    
    for field_name, field_value in fields.items():
        confidence = calculate_field_confidence(field_name, field_value, document_type, raw_text)
        
        annotated_fields[field_name] = {
            "value": field_value,
            "confidence": confidence
        }
    
    return annotated_fields


def calculate_overall_confidence(annotated_fields: Dict[str, Dict[str, Any]]) -> int:
    """
    Calculate overall document confidence from individual field confidences
    Uses weighted average based on field importance

    Returns: overall confidence as percentage (0-100)
    """
    if not annotated_fields:
        return 0
    
    # Field importance weights (higher = more important)
    importance_weights = {
        # ID Numbers (most important)
        "aadhaar_number": 1.5,
        "pan": 1.5,
        "voter_id": 1.5,
        "dl_number": 1.5,
        
        # Personal Info (important)
        "name": 1.3,
        "student_name": 1.3,
        "dob": 1.2,
        "father_name": 1.0,
        "mother_name": 0.9,
        
        # Contact Info (medium importance)
        "mobile": 1.0,
        "address": 0.9,
        
        # Dates (medium)
        "issue_date": 0.8,
        "valid_till": 0.8,
        "year": 0.8,
        
        # Other fields (less critical)
        "gender": 0.7,
        "school_name": 0.8,
        "roll_no": 1.0,
        "cgpa": 0.7,
    }
    
    total_weighted_confidence = 0.0
    total_weight = 0.0
    
    for field_name, field_data in annotated_fields.items():
        confidence = field_data.get("confidence", 0)  # percentage 0-100
        value = field_data.get("value")
        
        # Only count fields that have values
        if value is not None and (not isinstance(value, str) or value.strip()):
            weight = importance_weights.get(field_name, 1.0)
            total_weighted_confidence += confidence * weight
            total_weight += weight
    
    if total_weight == 0:
        return 0
    
    overall = total_weighted_confidence / total_weight  # this is a percentage
    return int(round(overall))


def enhance_extraction_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhance extraction result with confidence scores
    
    Input: Standard extraction result with flat fields
    Output: Enhanced result with per-field confidence
    """
    if not result or "fields" not in result:
        return result
    
    document_type = result.get("document_type", "Unknown")
    raw_text = result.get("raw_text_preview", "")
    fields = result.get("fields", {})
    
    # Add confidence to each field
    annotated_fields = add_confidence_to_fields(fields, document_type, raw_text)
    
    # Calculate overall confidence
    overall_confidence = calculate_overall_confidence(annotated_fields)
    
    # Update result
    enhanced_result = result.copy()
    enhanced_result["fields"] = annotated_fields
    enhanced_result["overall_confidence"] = overall_confidence
    enhanced_result["confidence"] = overall_confidence  # Keep for backward compatibility
    
    return enhanced_result


def get_low_confidence_fields(annotated_fields: Dict[str, Dict[str, Any]], 
                               threshold: int = 70) -> list:
    """
    Get list of fields with confidence below threshold (threshold is percentage, e.g., 70)
    Useful for highlighting fields that need review
    
    Returns: List of field info dicts with low confidence
    """
    low_conf_fields = []
    
    for field_name, field_data in annotated_fields.items():
        confidence = field_data.get("confidence", 0)
        value = field_data.get("value")
        
        # Only flag fields that have values but low confidence
        if value is not None and confidence < threshold:
            low_conf_fields.append({
                "field": field_name,
                "value": value,
                "confidence": confidence
            })
    
    # Sort by confidence (lowest first)
    low_conf_fields.sort(key=lambda x: x["confidence"])
    
    return low_conf_fields


def should_suggest_rescan(overall_confidence: int, 
                          low_confidence_count: int) -> bool:
    """
    Determine if rescan should be suggested to user
    
    Args:
        overall_confidence: Overall document confidence (0 to 100)
        low_confidence_count: Number of fields with low confidence
    
    Returns: True if rescan recommended
    """
    # Suggest rescan if:
    # 1. Overall confidence is low (< 70%)
    # 2. Multiple fields have low confidence (>= 3 fields)
    
    if overall_confidence < 70:
        return True
    
    if low_confidence_count >= 3:
        return True
    
    return False


# ========== HELPER FUNCTION FOR ROUTES ==========

def process_with_confidence(extraction_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to be called from routes
    Adds confidence scores and metadata to extraction result
    """
    # Enhance with confidence
    enhanced = enhance_extraction_result(extraction_result)
    
    # Get low confidence fields (threshold as percentage)
    annotated_fields = enhanced.get("fields", {})
    low_conf = get_low_confidence_fields(annotated_fields, threshold=70)
    
    # Add metadata
    enhanced["metadata"] = {
        "low_confidence_fields": low_conf,
        "low_confidence_count": len(low_conf),
        "suggest_rescan": should_suggest_rescan(
            enhanced.get("overall_confidence", 0),
            len(low_conf)
        ),
        "reviewed": False  # Frontend should set this to True after user review
    }
    
    return enhanced


# ========== SIMPLIFIED EXTRACTION SUMMARY ==========

def add_extraction_summary(enhanced_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add extraction summary to enhanced result (simplified - no options)
    
    Args:
        enhanced_result: Result from process_with_confidence()
    
    Returns:
        Enhanced result with extraction_summary
    """
    from services.extraction_summary import add_extraction_summary_to_result
    
    return add_extraction_summary_to_result(enhanced_result)


if __name__ == "__main__":
    # Test the confidence calculator (percentages)
    test_fields = {
        "name": "KOTTANGI CHARAN",
        "aadhaar_number": "900839614949",
        "dob": "28/08/2004",
        "gender": "Male",
        "father_name": None,
        "address": None,
        "mobile": None
    }
    
    annotated = add_confidence_to_fields(test_fields, "Aadhaar")
    overall = calculate_overall_confidence(annotated)
    
    print("Annotated Fields:")
    for field, data in annotated.items():
        print(f"  {field}: {data['value']} (confidence: {data['confidence']})")
    
    print(f"\nOverall Confidence: {overall}")
    
    low_conf = get_low_confidence_fields(annotated)
    print(f"\nLow Confidence Fields: {low_conf}")