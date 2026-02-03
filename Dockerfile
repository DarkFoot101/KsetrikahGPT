FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 1. COPY THE MODELS EXPLICITLY
# This will fail the build if Docker can't find them (which is what we want!)
COPY models/ /app/models/

# 2. Copy the rest of the code
COPY . .

# 3. Debug: Print file structure during build to verify
RUN echo "📂 Checking for models..." && ls -la /app/models/

EXPOSE 5000

# Ensure we run with unbuffered output so logs show up instantly
CMD ["python", "-u", "src/api/app.py"]