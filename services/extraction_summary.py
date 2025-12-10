"""
Extraction Summary Generator - Simplified (No Breakdown)
Generates confidence-based summary for OCR extraction results
"""

from typing import Dict, Any, List


def calculate_extraction_summary(annotated_fields: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate extraction summary with confidence classification
    
    Confidence Levels:
    - High Confidence (HC): >= 85%
    - Medium Confidence (MC): 68% to 84%
    - Low Confidence (LC): < 68%
    
    Args:
        annotated_fields: Dict with field names as keys, each containing:
                         {'value': ..., 'confidence': ...}
    
    Returns:
        Summary dict with counts only
    """
    if not annotated_fields:
        return {
            'overall_confidence': 0,
            'high_confidence': {'count': 0},
            'medium_confidence': {'count': 0},
            'low_confidence': {'count': 0}
        }
    
    # Field importance weights (higher = more important)
    importance_weights = {
        # ID Numbers (most important)
        "aadhaar_number": 1.5,
        "pan": 1.5,
        "voter_id": 1.5,
        "dl_number": 1.5,
        "roll_no": 1.3,
        
        # Personal Info (important)
        "name": 1.3,
        "student_name": 1.3,
        "dob": 1.2,
        "father_name": 1.0,
        "mother_name": 0.9,
        
        # Contact Info
        "mobile": 1.0,
        "address": 0.9,
        
        # Dates
        "issue_date": 0.8,
        "valid_till": 0.8,
        "year": 0.8,
        
        # Other fields
        "gender": 0.7,
        "school_name": 0.8,
        "cgpa": 0.7,
    }
    
    # Confidence thresholds
    HIGH_THRESHOLD = 85
    MEDIUM_THRESHOLD = 68
    
    # Initialize counters
    high_count = 0
    medium_count = 0
    low_count = 0
    
    total_weighted_confidence = 0.0
    total_weight = 0.0
    
    # Classify each field
    for field_name, field_data in annotated_fields.items():
        confidence = field_data.get("confidence", 0)
        value = field_data.get("value")
        
        # Only count fields that have values
        if value is not None and (not isinstance(value, str) or value.strip()):
            weight = importance_weights.get(field_name, 1.0)
            
            # Calculate weighted confidence
            total_weighted_confidence += confidence * weight
            total_weight += weight
            
            # Classify by confidence level
            if confidence >= HIGH_THRESHOLD:
                high_count += 1
            elif confidence >= MEDIUM_THRESHOLD:
                medium_count += 1
            else:
                low_count += 1
    
    # Calculate overall confidence
    overall_confidence = (total_weighted_confidence / total_weight) if total_weight > 0 else 0
    
    # Build summary (counts only)
    summary = {
        'overall_confidence': round(overall_confidence, 1),
        'high_confidence': {'count': high_count},
        'medium_confidence': {'count': medium_count},
        'low_confidence': {'count': low_count}
    }
    
    return summary


def should_suggest_rescan_v2(summary: Dict[str, Any]) -> bool:
    """
    Determine if rescan should be suggested based on summary
    """
    overall_conf = summary.get('overall_confidence', 0)
    low_count = summary.get('low_confidence', {}).get('count', 0)
    medium_count = summary.get('medium_confidence', {}).get('count', 0)
    
    # Suggest rescan if:
    # 1. Overall confidence is below 70%
    # 2. More than 2 low confidence fields
    # 3. More than 4 medium+low confidence fields
    
    if overall_conf < 70:
        return True
    
    if low_count > 2:
        return True
    
    if (medium_count + low_count) > 4:
        return True
    
    return False


def add_extraction_summary_to_result(enhanced_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add extraction summary to enhanced result
    
    Args:
        enhanced_result: Result from process_with_confidence()
    
    Returns:
        Enhanced result with extraction_summary
    """
    annotated_fields = enhanced_result.get("fields", {})
    
    # Calculate summary
    summary = calculate_extraction_summary(annotated_fields)
    
    # Add summary to result
    enhanced_result["extraction_summary"] = summary
    
    # Update suggest_rescan based on new logic
    if "metadata" not in enhanced_result:
        enhanced_result["metadata"] = {}
    
    enhanced_result["metadata"]["suggest_rescan"] = should_suggest_rescan_v2(summary)
    
    return enhanced_result


if __name__ == "__main__":
    # Example test
    sample_fields = {
        "aadhaar_number": {"value": "123456789012", "confidence": 98},
        "name": {"value": "JOHN DOE", "confidence": 88},
        "dob": {"value": "01/01/1990", "confidence": 95},
        "gender": {"value": "Male", "confidence": 99},
        "father_name": {"value": "FATHER NAME", "confidence": 75},
        "address": {"value": "123 Street, City", "confidence": 65},
        "mobile": {"value": "9876543210", "confidence": 80}
    }
    
    summary = calculate_extraction_summary(sample_fields)
    
    print("=" * 80)
    print("EXTRACTION SUMMARY TEST")
    print("=" * 80)
    import json
    print(json.dumps(summary, indent=2))
    print("=" * 80)