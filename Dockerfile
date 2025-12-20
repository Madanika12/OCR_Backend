# Use Python 3.9 base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OCR and document processing
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    # Image processing libraries
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    libglib2.0-dev \
    # Build tools (needed for some packages)
    build-essential \
    gcc \
    g++ \
    wget \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python packages - BASE DEPENDENCIES
RUN pip install --no-cache-dir \
    # Web Framework
    Flask==3.0.3 \
    flask-cors==4.0.1 \
    Werkzeug==3.0.3 \
    waitress==3.0.0 \
    # Database
    pymongo==4.8.0 \
    python-dotenv==1.0.1 \
    # Image Processing
    Pillow==10.4.0 \
    pytesseract==0.3.13 \
    numpy==1.26.4 \
    # PDF Processing
    pdf2image==1.17.0 \
    pdfplumber==0.11.4 \
    # Data Processing
    pandas==2.2.2 \
    # Utilities
    python-dateutil==2.9.0 \
    requests==2.32.3

# Install PyTorch (CPU version) - FULL VERSION
RUN pip install --no-cache-dir \
    torch==2.3.1 \
    torchvision==0.18.1 \
    torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install Computer Vision and ML packages
RUN pip install --no-cache-dir \
    # YOLO and object detection
    ultralytics==8.2.0 \
    # OpenCV (full version with GUI support)
    opencv-python==4.10.0.84 \
    # Document OCR
    python-doctr==0.8.1 \
    # Additional ML/CV tools
    scikit-learn==1.5.1 \
    scipy==1.13.1 \
    matplotlib==3.9.1 \
    seaborn==0.13.2

# Install additional useful packages
RUN pip install --no-cache-dir \
    # Image augmentation
    albumentations==1.4.11 \
    # Progress bars
    tqdm==4.66.4 \
    # YAML parsing
    PyYAML==6.0.1 \
    # HTTP library
    urllib3==2.2.2 \
    # Typing extensions
    typing-extensions==4.12.2

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p /app/uploads /app/logs && \
    chmod 755 /app/uploads /app/logs

# Expose port
EXPOSE 8000

# Set runtime environment variables
ENV TESSERACT_PATH=/usr/bin/tesseract \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata \
    FLASK_ENV=production \
    FLASK_DEBUG=False \
    ENABLE_YOLO=True

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Run application
CMD ["python", "server_production.py"]