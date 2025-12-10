FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install ONLY Tesseract and PDF tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py .
COPY server_production.py .
COPY routes/ ./routes/
COPY services/ ./services/

RUN mkdir -p /app/uploads /app/logs

ENV TESSERACT_PATH=/usr/bin/tesseract \
    FLASK_ENV=production \
    ENABLE_YOLO=False \
    PORT=8000 \
    APP_MODE=light

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["python", "server_production.py"]