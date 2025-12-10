"""
Application Configuration - Docker Compatible
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # MongoDB
    MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb+srv://smart_form_db:smartfill20@smartfill.sqmqcfc.mongodb.net/?retryWrites=true&w=majority&appName=smartfill')
    MONGODB_DATABASE = os.getenv('MONGODB_DATABASE', 'ocr_database')
    
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))
    
    # Upload
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', './uploads')
    ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}
    
    # FILE STORAGE SETTINGS
    FILE_STORAGE_MODE = os.getenv('FILE_STORAGE_MODE', 'filesystem')
    FILE_RETENTION_DAYS = int(os.getenv('FILE_RETENTION_DAYS', 30))
    ENABLE_FILE_STORAGE = os.getenv('ENABLE_FILE_STORAGE', 'True').lower() == 'true'
    
    # OCR - Smart path detection
    # Check if running in Docker (common indicators)
    IN_DOCKER = os.path.exists('/.dockerenv') or os.getenv('DOCKER_CONTAINER') == 'true'
    
    if IN_DOCKER:
        # Always use Linux path in Docker
        TESSERACT_PATH = '/usr/bin/tesseract'
    else:
        # Local development - detect OS
        import platform
        if platform.system() == 'Windows':
            TESSERACT_PATH = os.getenv('TESSERACT_PATH', r'D:\tesseract\tesseract.exe')
        else:
            TESSERACT_PATH = os.getenv('TESSERACT_PATH', '/usr/bin/tesseract')
    
    OCR_DPI = int(os.getenv('OCR_DPI', 300))
    OCR_CONFIDENCE_THRESHOLD = float(os.getenv('OCR_CONFIDENCE_THRESHOLD', 0.2))
    
    # YOLO
    ENABLE_YOLO = os.getenv('ENABLE_YOLO', 'False').lower() == 'true'
    YOLO_WEIGHTS = os.getenv('YOLO_WEIGHTS', 'yolov8n.pt')
    
    @staticmethod
    def init_app(app):
        # Create upload folder if not exists
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        
        # Validate storage mode
        if Config.FILE_STORAGE_MODE not in ['database', 'filesystem']:
            print(f"⚠️  Invalid FILE_STORAGE_MODE: {Config.FILE_STORAGE_MODE}")
            print("   Defaulting to 'filesystem'")
            Config.FILE_STORAGE_MODE = 'filesystem'
        
        # Print config on startup
        print(f"ℹ️  Tesseract path: {Config.TESSERACT_PATH}")
        print(f"ℹ️  Running in Docker: {Config.IN_DOCKER}")