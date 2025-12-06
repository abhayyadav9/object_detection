# ---- Base Python Image ----
FROM python:3.10-slim

# ---- Install System Dependencies ----
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    && apt-get clean

# ---- Create App Directory ----
WORKDIR /app

# ---- Copy Project Files ----
COPY . .

# ---- Install Python Dependencies ----
RUN pip install --no-cache-dir -r requirements.txt

# ---- Expose Flask Port ----
EXPOSE 5001

# ---- Start Flask App ----
CMD ["python", "app.py"]
