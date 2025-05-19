FROM python:3.11-slim

# 1. System deps (EasyOCR butuh libgl)
RUN apt-get update && apt-get install -y libgl1 && rm -rf /var/lib/apt/lists/*

# 2. Workdir & requirements
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy source & model
COPY app/ /app/

# 4. Start
CMD ["python", "wsgi.py"]