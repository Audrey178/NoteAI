FROM python:3.12-slim

# Thư viện hệ thống cho OpenCV / Paddle
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PADDLE_PDX_CACHE_HOME=/models \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
