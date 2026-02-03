FROM python:3.10-slim

WORKDIR /app

# 1. Install System Dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. COPY EVERYTHING (The Fix 🛠️)
# Instead of picking files one by one, we copy the WHOLE project.
# This ensures 'models/', 'src/', 'data/' all get copied exactly as they are.
COPY . .

# 4. Debug & Safety Check
# This will LIST the models folder during the build.
# If the file is missing, the build will CRASH here (so you know immediately).
RUN echo "🔍 Checking if models exist..." && ls -la /app/models/
RUN test -f /app/models/best_model.joblib || (echo "❌ ERROR: best_model.joblib is MISSING!" && exit 1)

EXPOSE 5000

# 5. Run the App
# We use "-u" to force logs to show up instantly
CMD ["python", "-u", "src/api/app.py"]