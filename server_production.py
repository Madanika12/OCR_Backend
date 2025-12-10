"""
Production Server - Koyeb Docker Compatible
"""
import os
from waitress import serve
from flask import Flask, jsonify
from flask_cors import CORS
from routes.ocr_routes import ocr_blueprint
from config import Config
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.config.from_object(Config)
Config.init_app(app)

# Enable CORS
CORS(app, resources={
    r"/api/*": {
        "origins": "https://smart-form-frontend-gold.vercel.app",
        "methods": ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Register Blueprints
app.register_blueprint(ocr_blueprint, url_prefix="/api")

# Root endpoints
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "name": "OCR API",
        "version": "2.0.0",
        "status": "running",
        "mode": "production",
        "documentation": "/api/docs"
    })

@app.route("/health", methods=["GET"])
def health():
    try:
        from services.database import get_db
        db = get_db()
        stats = db.get_statistics()
        return jsonify({
            "status": "healthy",
            "service": "OCR API",
            "database": "connected",
            "total_scans": stats.get('total_scans', 0)
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return jsonify({"success": False, "error": "Internal server error"}), 500

if __name__ == '__main__':
    # Get port from environment variable (Koyeb sets this)
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info("=" * 70)
    logger.info("🏭 OCR API - PRODUCTION SERVER")
    logger.info("=" * 70)
    logger.info(f"Port: {port}")
    logger.info(f"Host: {host}")
    logger.info(f"MongoDB: {Config.MONGODB_URI[:50]}...")
    logger.info("=" * 70)
    
    try:
        # Serve the app with Waitress
        serve(
            app,
            host=host,
            port=port,
            threads=4,
            channel_timeout=120,
            url_scheme='http',
            ident='OCR-API/2.0'
        )
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise