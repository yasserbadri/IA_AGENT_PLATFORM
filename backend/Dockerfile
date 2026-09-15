FROM python:3.12-slim

WORKDIR /app

# tesseract-ocr : moteur OCR utilisé par l'outil read_file pour extraire le
# texte des photos/images uploadées (voir app/agent/tools/read_file.py).
# fra + eng : langues nécessaires pour l'OCR en français et en anglais.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-fra \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
