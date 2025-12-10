"""
Hybrid File Storage Service
Supports both Database (GridFS) and Filesystem storage
Switch between modes via config
"""

import os
import gridfs
from pymongo import MongoClient
from typing import Optional, Tuple
from config import Config
from datetime import datetime, timedelta

class FileStorageService:
    def __init__(self):
        self.mode = Config.FILE_STORAGE_MODE
        self.uploads_folder = Config.UPLOAD_FOLDER
        
        # Initialize based on mode
        if self.mode == 'database':
            self.client = MongoClient(Config.MONGODB_URI)
            self.db = self.client[Config.MONGODB_DATABASE]
            self.fs = gridfs.GridFS(self.db)
            print(f"✅ File Storage: Database (GridFS)")
        else:
            os.makedirs(self.uploads_folder, exist_ok=True)
            print(f"✅ File Storage: Filesystem ({self.uploads_folder})")
    
    def save_file(self, scan_id: str, file_bytes: bytes, filename: str) -> dict:
        """
        Save file using configured storage mode
        Returns: Storage metadata
        """
        try:
            if self.mode == 'database':
                # Save to MongoDB GridFS
                file_id = self.fs.put(
                    file_bytes,
                    filename=filename,
                    scan_id=scan_id,
                    upload_date=datetime.utcnow()
                )
                return {
                    'stored': True,
                    'storage_mode': 'database',
                    'gridfs_id': str(file_id),
                    'filename': filename
                }
            else:
                # Save to filesystem
                file_path = os.path.join(self.uploads_folder, f"{scan_id}_{filename}")
                with open(file_path, 'wb') as f:
                    f.write(file_bytes)
                return {
                    'stored': True,
                    'storage_mode': 'filesystem',
                    'file_path': file_path,
                    'filename': filename
                }
        except Exception as e:
            print(f"❌ File save error: {e}")
            return {'stored': False, 'error': str(e)}
    
    def get_file(self, scan_id: str, storage_metadata: dict) -> Optional[bytes]:
        """
        Retrieve file bytes using storage metadata
        """
        try:
            storage_mode = storage_metadata.get('storage_mode', self.mode)
            
            if storage_mode == 'database':
                # Get from GridFS
                from bson import ObjectId
                gridfs_id = storage_metadata.get('gridfs_id')
                if gridfs_id:
                    file_data = self.fs.get(ObjectId(gridfs_id))
                    return file_data.read()
            else:
                # Get from filesystem
                file_path = storage_metadata.get('file_path')
                if file_path and os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        return f.read()
                else:
                    # Fallback: try to construct path
                    filename = storage_metadata.get('filename')
                    if filename:
                        file_path = os.path.join(self.uploads_folder, f"{scan_id}_{filename}")
                        if os.path.exists(file_path):
                            with open(file_path, 'rb') as f:
                                return f.read()
            
            return None
        except Exception as e:
            print(f"❌ File retrieval error: {e}")
            return None
    
    def delete_file(self, scan_id: str, storage_metadata: dict) -> bool:
        """
        Delete file from storage
        """
        try:
            storage_mode = storage_metadata.get('storage_mode', self.mode)
            
            if storage_mode == 'database':
                # Delete from GridFS
                from bson import ObjectId
                gridfs_id = storage_metadata.get('gridfs_id')
                if gridfs_id:
                    self.fs.delete(ObjectId(gridfs_id))
                    return True
            else:
                # Delete from filesystem
                file_path = storage_metadata.get('file_path')
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
                    return True
            
            return False
        except Exception as e:
            print(f"❌ File deletion error: {e}")
            return False
    
    def cleanup_old_files(self, days: int = None):
        """
        Delete files older than specified days
        """
        if days is None:
            days = Config.FILE_RETENTION_DAYS
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        try:
            if self.mode == 'database':
                # Find and delete old GridFS files
                old_files = self.fs.find({'upload_date': {'$lt': cutoff_date}})
                count = 0
                for file in old_files:
                    self.fs.delete(file._id)
                    count += 1
                print(f"🗑️ Cleaned {count} old files from database")
            else:
                # Delete old filesystem files
                count = 0
                for filename in os.listdir(self.uploads_folder):
                    file_path = os.path.join(self.uploads_folder, filename)
                    if os.path.isfile(file_path):
                        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                        if file_time < cutoff_date:
                            os.remove(file_path)
                            count += 1
                print(f"🗑️ Cleaned {count} old files from filesystem")
        except Exception as e:
            print(f"❌ Cleanup error: {e}")


# Singleton instance
_storage_service = None

def get_storage() -> FileStorageService:
    """Get file storage service instance"""
    global _storage_service
    if _storage_service is None:
        _storage_service = FileStorageService()
    return _storage_service


if __name__ == "__main__":
    # Test storage service
    storage = get_storage()
    print(f"Storage mode: {storage.mode}")