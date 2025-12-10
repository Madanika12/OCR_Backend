"""
MongoDB Database Service - Fixed to store table in submissions
"""

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, PyMongoError
from datetime import datetime
from typing import Dict, Any, List, Optional
from bson import ObjectId
import json
from config import Config
from services.models import (
    ScanDocument, RescanDocument, SubmissionDocument,
    DocumentType, ScanStatus, SubmissionStatus
)

class DatabaseService:
    def __init__(self):
        try:
            self.client = MongoClient(Config.MONGODB_URI)
            self.db = self.client[Config.MONGODB_DATABASE]
            
            # Collections
            self.scans = self.db.scans
            self.rescans = self.db.rescans
            self.submissions = self.db.submissions
            
            # Create indexes for better performance
            self._create_indexes()
            
            # Test connection
            self.client.admin.command('ping')
            print("✅ MongoDB connected successfully")
            
        except ConnectionFailure as e:
            print(f"❌ MongoDB connection failed: {e}")
            raise
    
    def _create_indexes(self):
        """Create database indexes"""
        try:
            # Scans collection indexes
            self.scans.create_index([("scan_id", ASCENDING)], unique=True)
            self.scans.create_index([("document_type", ASCENDING)])
            self.scans.create_index([("created_at", DESCENDING)])
            self.scans.create_index([("status", ASCENDING)])
            
            # Rescans collection indexes
            self.rescans.create_index([("rescan_id", ASCENDING)], unique=True)
            self.rescans.create_index([("original_scan_id", ASCENDING)])
            self.rescans.create_index([("created_at", DESCENDING)])
            
            # Submissions collection indexes
            self.submissions.create_index([("submission_id", ASCENDING)], unique=True)
            self.submissions.create_index([("scan_id", ASCENDING)])
            self.submissions.create_index([("created_at", DESCENDING)])
            self.submissions.create_index([("status", ASCENDING)])
            
        except Exception as e:
            print(f"⚠️ Index creation warning: {e}")
    
    # ==================== SCAN OPERATIONS ====================
    
    def save_scan(self, scan_data: Dict[str, Any]) -> str:
        """
        Save initial scan result using ScanDocument model
        Returns: scan_id
        """
        try:
            from uuid import uuid4
            scan_id = str(uuid4())
            
            # Create ScanDocument from extraction result
            scan_doc = ScanDocument.from_extraction(scan_id, scan_data)
            
            # Convert to dict and insert
            self.scans.insert_one(scan_doc.to_dict())
            print(f"✅ Scan saved: {scan_id}")
            return scan_id
            
        except PyMongoError as e:
            print(f"❌ Error saving scan: {e}")
            raise
    
    def get_scan(self, scan_id: str) -> Optional[Dict[str, Any]]:
        """Get scan by ID"""
        try:
            scan = self.scans.find_one({"scan_id": scan_id})
            if scan:
                scan['_id'] = str(scan['_id'])
            return scan
        except PyMongoError as e:
            print(f"❌ Error retrieving scan: {e}")
            return None
    
    def get_all_scans(self, limit: int = 100, skip: int = 0, 
            document_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all scans with pagination and filtering"""
        try:
            query = {}
            if document_type:
                query['document_type'] = document_type
            
            scans = list(self.scans.find(query)
                        .sort("created_at", DESCENDING)
                        .skip(skip)
                        .limit(limit))
            
            for scan in scans:
                scan['_id'] = str(scan['_id'])
            
            return scans
        except PyMongoError as e:
            print(f"❌ Error retrieving scans: {e}")
            return []
    
    def update_scan(self, scan_id: str, update_data: Dict[str, Any]) -> bool:
        """Update scan data"""
        try:
            update_data['updated_at'] = datetime.utcnow()
            result = self.scans.update_one(
                {"scan_id": scan_id},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except PyMongoError as e:
            print(f"❌ Error updating scan: {e}")
            return False
    
    def delete_scan(self, scan_id: str) -> bool:
        """Delete scan"""
        try:
            result = self.scans.delete_one({"scan_id": scan_id})
            return result.deleted_count > 0
        except PyMongoError as e:
            print(f"❌ Error deleting scan: {e}")
            return False
    
    # ==================== RESCAN OPERATIONS ====================
    
    def save_rescan(self, rescan_data: Dict[str, Any], original_scan_id: str) -> str:
        """
        Save rescan result using RescanDocument model
        Returns: rescan_id
        """
        try:
            from uuid import uuid4
            rescan_id = str(uuid4())
            
            # Create RescanDocument from extraction result
            rescan_doc = RescanDocument.from_extraction(
                rescan_id, 
                original_scan_id, 
                rescan_data
            )
            
            # Convert to dict and insert
            self.rescans.insert_one(rescan_doc.to_dict())
            
            # Update original scan's rescan count
            self.scans.update_one(
                {"scan_id": original_scan_id},
                {
                    "$inc": {"rescan_count": 1},
                    "$set": {"updated_at": datetime.utcnow()}
                }
            )
            
            print(f"✅ Rescan saved: {rescan_id}")
            return rescan_id
            
        except PyMongoError as e:
            print(f"❌ Error saving rescan: {e}")
            raise
    
    def get_rescan(self, rescan_id: str) -> Optional[Dict[str, Any]]:
        """Get rescan by ID"""
        try:
            rescan = self.rescans.find_one({"rescan_id": rescan_id})
            if rescan:
                rescan['_id'] = str(rescan['_id'])
            return rescan
        except PyMongoError as e:
            print(f"❌ Error retrieving rescan: {e}")
            return None
    
    def get_rescans_by_scan(self, scan_id: str) -> List[Dict[str, Any]]:
        """Get all rescans for a specific scan"""
        try:
            rescans = list(self.rescans.find({"original_scan_id": scan_id})
                .sort("created_at", DESCENDING))
            
            for rescan in rescans:
                rescan['_id'] = str(rescan['_id'])
            
            return rescans
        except PyMongoError as e:
            print(f"❌ Error retrieving rescans: {e}")
            return []
    
    # ==================== SUBMISSION OPERATIONS ====================
    
    def save_submission(self, submission_data: Dict[str, Any]) -> str:
        """
        Save final submission - NOW INCLUDES TABLE DATA
        Returns: submission_id
        """
        try:
            from uuid import uuid4
            submission_id = str(uuid4())
            
            # Build submission document with table
            submission_doc = {
                "submission_id": submission_id,
                "scan_id": submission_data.get('scan_id'),
                "rescan_id": submission_data.get('rescan_id'),
                "document_type": submission_data.get('document_type'),
                "verified_fields": submission_data.get('verified_fields', {}),
                "table": submission_data.get('table', []),  # 🆕 Store table data
                "user_corrections": submission_data.get('user_corrections', {}),
                "final_confidence": submission_data.get('final_confidence', 0.0),
                "status": SubmissionStatus.SUBMITTED.value,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            # Insert submission
            self.submissions.insert_one(submission_doc)
            
            # Update scan status
            if submission_data.get('scan_id'):
                self.scans.update_one(
                    {"scan_id": submission_data['scan_id']},
                    {
                        "$set": {
                            "status": ScanStatus.SUBMITTED.value, 
                            "updated_at": datetime.utcnow()
                        }
                    }
                )
            
            print(f"✅ Submission saved: {submission_id} (with {len(submission_doc.get('table', []))} table rows)")
            return submission_id
            
        except PyMongoError as e:
            print(f"❌ Error saving submission: {e}")
            raise
    
    def get_submission(self, submission_id: str) -> Optional[Dict[str, Any]]:
        """Get submission by ID"""
        try:
            submission = self.submissions.find_one({"submission_id": submission_id})
            if submission:
                submission['_id'] = str(submission['_id'])
            return submission
        except PyMongoError as e:
            print(f"❌ Error retrieving submission: {e}")
            return None
    
    def get_submissions_by_scan(self, scan_id: str) -> List[Dict[str, Any]]:
        """Get all submissions for a specific scan"""
        try:
            submissions = list(self.submissions.find({"scan_id": scan_id})
                            .sort("created_at", DESCENDING))
            
            for submission in submissions:
                submission['_id'] = str(submission['_id'])
            
            return submissions
        except PyMongoError as e:
            print(f"❌ Error retrieving submissions: {e}")
            return []
    
    def get_all_submissions(self, limit: int = 100, skip: int = 0,
                        status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all submissions with pagination and filtering"""
        try:
            query = {}
            if status:
                query['status'] = status
            
            submissions = list(self.submissions.find(query)
                    .sort("created_at", DESCENDING)
                    .skip(skip)
                    .limit(limit))
            
            for submission in submissions:
                submission['_id'] = str(submission['_id'])
            
            return submissions
        except PyMongoError as e:
            print(f"❌ Error retrieving submissions: {e}")
            return []
    
    # ==================== STATISTICS ====================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            stats = {
                "total_scans": self.scans.count_documents({}),
                "total_rescans": self.rescans.count_documents({}),
                "total_submissions": self.submissions.count_documents({}),
                "scans_by_type": {},
                "submissions_by_status": {},
                "recent_activity": []
            }
            
            # Scans by document type
            pipeline = [
                {"$group": {"_id": "$document_type", "count": {"$sum": 1}}}
            ]
            for doc in self.scans.aggregate(pipeline):
                stats['scans_by_type'][doc['_id']] = doc['count']
            
            # Submissions by status
            pipeline = [
                {"$group": {"_id": "$status", "count": {"$sum": 1}}}
            ]
            for doc in self.submissions.aggregate(pipeline):
                stats['submissions_by_status'][doc['_id']] = doc['count']
            
            # Recent activity (last 10 scans)
            recent = list(self.scans.find()
                        .sort("created_at", DESCENDING)
                        .limit(10))
            for item in recent:
                stats['recent_activity'].append({
                    "scan_id": item['scan_id'],
                    "document_type": item['document_type'],
                    "created_at": item['created_at'].isoformat()
                })
            
            return stats
        except PyMongoError as e:
            print(f"❌ Error getting statistics: {e}")
            return {}
    
    # ==================== UTILITY ====================
    
    def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            print("✅ MongoDB connection closed")

# Singleton instance
_db_service = None

def get_db() -> DatabaseService:
    """Get database service instance"""
    global _db_service
    if _db_service is None:
        _db_service = DatabaseService()
    return _db_service


if __name__ == "__main__":
    # Test database connection
    db = get_db()
    print("Database service initialized successfully")
    stats = db.get_statistics()
    print(f"Statistics: {json.dumps(stats, indent=2)}")

    #the end