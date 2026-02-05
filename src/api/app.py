import os
import sys
import io
import joblib
import base64
import tempfile
import pandas as pd
import numpy as np
import requests
import whisper
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv

# --- CONFIGURATION ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
load_dotenv()

# API Keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://127.0.0.1:5000")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# This sets BASE_DIR to the root '/app' folder

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "encoders.joblib")
print("Success")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# --- LOAD MODELS ---
print("⏳ Loading KsetrikahGPT Brain...")
try:
    # 1. Load ML Price Model
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        print("📦 Loading ML models...")
        print("MODEL PATH:", MODEL_PATH, "exists:", os.path.exists(MODEL_PATH))
        print("ENCODER PATH:", ENCODER_PATH, "exists:", os.path.exists(ENCODER_PATH))

        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODER_PATH)

        print("✅ Models loaded successfully")
    else:
        print("⚠️ Warning: Price models not found. Prediction will fail.")
        model, encoders = None, None

    # 2. Load Whisper (Speech-to-Text)
    # Note: Requires 'ffmpeg' installed on the system
    print("⏳ Loading Whisper Model (this may take a moment)...")
    whisper_model = whisper.load_model("base")
    print("✅ Whisper Model Ready.")

except Exception as e:
    import traceback
    print(f"❌ Critical Error Loading Models: {e}")
    traceback.print_exc()
    model, encoders, whisper_model = None, None, None


# --- THE FRONTEND (Updated with AI Assistant UI) ---
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not encoders:
        return jsonify({"error": "Model not loaded"}), 500
    try:
        data = request.json
        input_data = {
            'MSP': [data['MSP']], 'Price_1DayAgo': [data['Price_1DayAgo']],
            'Price_2DaysAgo': [data['Price_2DaysAgo']], 'Arrival_Today': [data['Arrival_Today']],
            'Arrival_1DayAgo': [data['Arrival_1DayAgo']], 'Arrival_2DaysAgo': [data['Arrival_2DaysAgo']]
        }
        df = pd.DataFrame(input_data)
        
        # Features
        df['msp_premium'] = df['Price_1DayAgo'] - df['MSP']
        df['price_momentum'] = (df['Price_1DayAgo'] - df['Price_2DaysAgo']) / (df['Price_2DaysAgo'] + 1e-9)
        df['price_volatility'] = df[['Price_1DayAgo', 'Price_2DaysAgo']].std(axis=1).fillna(0)

        # Encoders
        cat_cols = ['Commodity_Group', 'Commodity', 'Variety']
        for col in cat_cols:
            raw_val = data.get(col, "Unknown")
            if col in encoders:
                le = encoders[col]
                df[f'{col}_Encoded'] = le.transform([raw_val]) if raw_val in le.classes_ else 0
            else:
                df[f'{col}_Encoded'] = 0

        pred = model.predict(df)[0]
        trend = "UP" if pred > data['Price_1DayAgo'] else "DOWN"
        return jsonify({"predicted_price_tomorrow": round(float(pred), 2), "trend": trend})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/assistant/analyze', methods=['POST'])
def assistant_analyze():
    # 1. Handle Voice Input (Whisper)
    prompt = request.form.get('prompt', '').strip()
    language = request.form.get('language', 'en')
    
    if 'audio' in request.files and whisper_model:
        audio_file = request.files['audio']
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            audio_path = tmp.name
            audio_file.save(audio_path)
        try:
            result = whisper_model.transcribe(audio_path, language=language if language != 'en' else None)
            prompt = result['text']
        except Exception as e:
            print(f"Whisper Error: {e}")
        finally:
            if os.path.exists(audio_path): os.remove(audio_path)

    # 2. Handle Image
    if 'image' not in request.files:
        return jsonify({"error": "Please upload an image for the Agronomist to analyze."}), 400
    
    image = request.files['image']
    image_bytes = image.read()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    # 3. Call Vision Model
    if not prompt: prompt = "Analyze this crop image. Diagnose any diseases and suggest treatments."
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": FRONTEND_URL,
    }
    
    # Map codes to full language names for the System Prompt
    lang_map = {"hi": "Hindi", "ta": "Tamil", "en": "English"}
    target_lang = lang_map.get(language, "English")

    payload = {
        "model": "qwen/qwen2.5-vl-32b-instruct",
        "messages": [
            {"role": "system", "content": f"You are an expert agricultural AI. Analyze the crop image and answer the user's question. Reply ONLY in {target_lang}. Keep it helpful and concise for a farmer."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{image.mimetype};base64,{base64_image}"}}
            ]}
        ]
    }

    try:
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        r.raise_for_status()
        reply = r.json()["choices"][0]["message"]["content"]
        return jsonify({"response": reply, "transcribed_prompt": prompt})
    except Exception as e:
        return jsonify({"error": f"AI Error: {str(e)}"}), 500

@app.route('/assistant/text-to-speech', methods=['POST'])
def assistant_tts():
    data = request.get_json()
    text = data.get("text")
    language = data.get("language", "en")
    
    # Voice IDs (English, Hindi, Tamil placeholders - replace with your preferred IDs)
    voice_ids = {
        "en": "21m00Tcm4TlvDq8ikWAM", # Rachel
        "hi": "FiIgWdzVKAalJyAgg8Pg", # Example
        "ta": "Z0ocGS7BSRxFSMhV00nB"  # Example
    }
    voice_id = voice_ids.get(language, voice_ids["en"])

    try:
        r = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            json={"text": text, "model_id": "eleven_multilingual_v2"},
            headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
        )
        r.raise_for_status()
        return send_file(io.BytesIO(r.content), mimetype="audio/mpeg")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)