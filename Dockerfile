FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Use Gunicorn to serve Flask app in production
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "backend.app.main:app"]
