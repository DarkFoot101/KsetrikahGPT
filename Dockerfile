FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code FIRST
COPY src/ ./src/
COPY app.py ./app.py

# Copy models LAST (never overwritten)
COPY models/ /app/models/

# Hard fail if models missing
RUN test -f /app/models/best_model.joblib
RUN test -f /app/models/encoders.joblib

# Debug visibility
RUN echo "📂 Final models directory:" && ls -la /app/models/

EXPOSE 5000

CMD ["python", "-u", "src/api/app.py"]
