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
from flask import Flask, render_template_string, request, jsonify, send_file
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
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>KsetrikahGPT | Smart Farming</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />

  <!-- Fonts -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">

  <!-- Icons -->
  <script src="https://code.iconify.design/iconify-icon/1.0.7/iconify-icon.min.js"></script>

  <style>
    :root {
      --bg:
        radial-gradient(1200px 600px at 20% 10%, #0f766e33, transparent),
        radial-gradient(800px 500px at 80% 90%, #2563eb22, transparent),
        #020617;

      --glass: rgba(255,255,255,0.05);
      --border: rgba(255,255,255,0.1);
      --text: #e5e7eb;
      --muted: #9ca3af;
      --accent: #22d3ee;
      --success: #22c55e;
      --danger: #ef4444;

      --radius: 18px;
      --blur: blur(18px);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      font-family: 'Inter', system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
    }

    /* SIDEBAR */
    .sidebar {
      width: 260px;
      padding: 32px 24px;
      background: rgba(0,0,0,0.35);
      border-right: 1px solid var(--border);
      backdrop-filter: var(--blur);
      flex-shrink: 0;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      font-size: 1.4rem;
      font-weight: 800;
      margin-bottom: 40px;
    }

    .brand-icon {
      width: 38px;
      height: 38px;
      background: linear-gradient(135deg, #22d3ee, #38bdf8);
      color: #020617;
      border-radius: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .nav-item {
      padding: 14px 18px;
      border-radius: 14px;
      margin-bottom: 10px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 12px;
      font-weight: 600;
      color: var(--muted);
      transition: all .2s ease;
    }

    .nav-item.active,
    .nav-item:hover {
      background: rgba(255,255,255,0.08);
      color: var(--text);
    }

    /* MAIN */
    .main {
      flex: 1;
      padding: 48px;
      overflow-y: auto;
    }

    h1 {
      font-size: 2.2rem;
      font-weight: 800;
      margin: 0 0 8px 0;
    }

    .subtitle {
      color: var(--muted);
      margin-bottom: 36px;
      font-size: 1.05rem;
    }

    /* CARDS */
    .card {
      background: var(--glass);
      backdrop-filter: var(--blur);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 28px;
      box-shadow: 0 40px 80px rgba(0,0,0,0.45);
    }

    .grid {
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 32px;
    }
    
    .grid-single {
      display: grid;
      grid-template-columns: 1fr;
      gap: 32px;
      max-width: 800px;
    }

    label {
      font-size: .85rem;
      color: var(--muted);
      font-weight: 600;
      margin-bottom: 6px;
      display: block;
    }

    .input, input[type="file"], textarea {
      width: 100%;
      padding: 14px 18px;
      border-radius: 14px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.05);
      color: var(--text);
      font-size: .95rem;
      font-family: inherit;
    }
    
    textarea {
      resize: vertical;
      min-height: 100px;
    }

    .input:focus, input[type="file"]:focus, textarea:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 4px rgba(34,211,238,0.2);
    }
    
    input[type="file"]::file-selector-button {
      background: rgba(255,255,255,0.1);
      border: none;
      color: var(--text);
      padding: 8px 12px;
      border-radius: 8px;
      margin-right: 12px;
      cursor: pointer;
      font-weight: 600;
      transition: background .2s;
    }
    
    input[type="file"]::file-selector-button:hover {
      background: rgba(255,255,255,0.2);
    }

    .row {
      display: grid;
      grid-template-columns: repeat(2,1fr);
      gap: 16px;
      margin-bottom: 18px;
    }

    .row-3 {
      display: grid;
      grid-template-columns: repeat(3,1fr);
      gap: 16px;
    }

    .btn {
      margin-top: 28px;
      width: 100%;
      height: 56px;
      border-radius: 18px;
      border: none;
      cursor: pointer;
      font-weight: 800;
      font-size: 1rem;
      background: linear-gradient(135deg, #22d3ee, #38bdf8);
      color: #020617;
      transition: all .15s ease;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
    }

    .btn:hover {
      transform: translateY(-2px);
      box-shadow: 0 18px 45px rgba(34,211,238,0.4);
    }
    
    .btn:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      transform: none;
    }

    /* RESULT */
    .result {
      text-align: center;
      padding: 50px 30px;
      background: linear-gradient(
        145deg,
        rgba(34,211,238,0.15),
        rgba(34,197,94,0.05)
      );
      border-radius: 26px;
      border: 1px solid var(--border);
    }
    
    .ai-response-box {
        margin-top: 32px;
        background: rgba(0,0,0,0.2);
        border-radius: var(--radius);
        padding: 24px;
        border: 1px solid var(--border);
    }
    
    .ai-text {
        line-height: 1.6;
        white-space: pre-wrap;
        color: var(--text);
        text-align: left;
    }
    
    audio {
        width: 100%;
        margin-top: 20px;
        height: 40px;
    }

    .result p {
      color: var(--muted);
      margin: 0;
      font-weight: 600;
    }

    .price {
      font-size: 4.2rem;
      font-weight: 900;
      letter-spacing: -0.04em;
      margin: 16px 0;
      color: var(--accent);
    }

    .trend {
      display: inline-block;
      padding: 10px 22px;
      border-radius: 999px;
      font-weight: 700;
      font-size: .95rem;
    }

    .trend.up {
      background: rgba(34,197,94,0.2);
      color: var(--success);
    }

    .trend.down {
      background: rgba(239,68,68,0.2);
      color: var(--danger);
    }

    .hidden {
      display: none !important;
    }
    
    .error-msg {
        color: var(--danger);
        background: rgba(239,68,68,0.1);
        padding: 12px;
        border-radius: 12px;
        margin-top: 16px;
        display: none;
    }
  </style>
</head>

<body>

  <!-- SIDEBAR -->
  <div class="sidebar">
    <div class="brand">
      <div class="brand-icon">
        <iconify-icon icon="lucide:wheat"></iconify-icon>
      </div>
      KsetrikahGPT
    </div>

    <div id="nav-market" class="nav-item active" onclick="showMarket()">
        <iconify-icon icon="lucide:trending-up"></iconify-icon>
        Isi Mandi (Market)
    </div>
    <div id="nav-agro" class="nav-item" onclick="showAgronomist()">
        <iconify-icon icon="lucide:sprout"></iconify-icon>
        AI Agronomist
    </div>
  </div>

  <!-- MAIN -->
  <div class="main">

    <!-- MARKET SECTION -->
    <div id="market-section">
        <h1>Market Price Forecast</h1>
        <div class="subtitle">
        Predict tomorrow’s commodity prices using machine learning.
        </div>

        <div class="grid">

        <!-- INPUT CARD -->
        <div class="card">

            <div class="row">
            <div>
                <label>Commodity Group</label>
                <select id="grp" class="input">
                <option>Cereals</option>
                <option>Pulses</option>
                <option>Vegetables</option>
                </select>
            </div>
            <div>
                <label>Commodity</label>
                <input id="comm" class="input" value="Bajra (Pearl Millet)">
            </div>
            </div>

            <label>MSP (₹ / Quintal)</label>
            <input id="msp" class="input" type="number" value="2775">

            <div class="row" style="margin-top:20px">
            <div>
                <label>Price Yesterday</label>
                <input id="p1" class="input" type="number" value="2054">
            </div>
            <div>
                <label>Price 2 Days Ago</label>
                <input id="p2" class="input" type="number" value="1987">
            </div>
            </div>

            <div class="row-3" style="margin-top:20px">
            <div>
                <label>Arrival Today</label>
                <input id="a0" class="input" type="number" value="100">
            </div>
            <div>
                <label>Arrival Yesterday</label>
                <input id="a1" class="input" type="number" value="167">
            </div>
            <div>
                <label>Arrival 2 Days Ago</label>
                <input id="a2" class="input" type="number" value="54">
            </div>
            </div>

            <button class="btn" onclick="predict()">
                <iconify-icon icon="lucide:rocket"></iconify-icon>
                Predict Price
            </button>
        </div>

        <!-- RESULT -->
        <div id="resultBox" class="result hidden">
            <p>Tomorrow’s Estimated Market Price</p>
            <div id="price" class="price">₹ 0</div>
            <div id="trend" class="trend down">Bearish Trend</div>
        </div>

        </div>
    </div>
    
    <!-- AGRONOMIST SECTION -->
    <div id="agronomist-section" class="hidden">
        <h1>AI Agronomist</h1>
        <div class="subtitle">
        Get expert crop advice from our multi-modal AI assistant.
        </div>
        
        <div class="grid-single">
            <div class="card">
                <label>Upload Crop Image (Required)</label>
                <input type="file" id="cropImage" accept="image/*" class="mb-4" style="margin-bottom: 20px;">
                
                <label>Voice Note (Optional)</label>
                <div class="row">
                   <button class="btn" style="margin-top:0; background: #22c55e; color: #fff" onclick="startRecording()" id="startRecBtn">
                       <iconify-icon icon="lucide:mic"></iconify-icon> Record
                   </button>
                   <button class="btn" style="margin-top:0; background: #ef4444; color: #fff" onclick="stopRecording()" id="stopRecBtn" disabled>
                       <iconify-icon icon="lucide:square"></iconify-icon> Stop
                   </button>
                </div>
                <div id="recStatus" style="color:var(--accent); margin-top:8px; font-size:0.9rem; font-weight:600; min-height:20px"></div>
                
                <input type="file" id="voiceNote" accept="audio/*" style="display:none">
                <div style="margin-bottom: 20px; color: var(--muted); font-size: 0.85rem; margin-top: 8px;">
                    Or <a href="#" onclick="document.getElementById('voiceNote').click(); return false;" style="color:var(--accent)">upload an audio file</a>
                </div>
                
                <label>Your Question</label>
                <textarea id="promptText" class="input" placeholder="e.g., What disease is affecting this leaf? How should I treat it?"></textarea>
                
                <div class="row" style="margin-top: 20px;">
                    <div>
                         <label>Language</label>
                         <select id="responseLang" class="input">
                             <option value="en">English</option>
                             <option value="hi">Hindi</option>
                             <option value="ta">Tamil</option>
                         </select>
                    </div>
                </div>
                
                <button class="btn" id="analyzeBtn" onclick="analyzeCrop()">
                    <iconify-icon icon="lucide:scan-face"></iconify-icon>
                    Analyze Crop
                </button>
                
                <div id="agroError" class="error-msg"></div>

                <div id="agroResult" class="ai-response-box hidden">
                    <label>AI Diagnosis:</label>
                    <div id="agroText" class="ai-text"></div>
                    <audio id="agroAudio" controls class="hidden"></audio>
                </div>
            </div>
        </div>
    </div>

  </div>

<script>
// --- TABS ---
function showMarket() {
    document.getElementById('market-section').classList.remove('hidden');
    document.getElementById('agronomist-section').classList.add('hidden');
    document.getElementById('nav-market').classList.add('active');
    document.getElementById('nav-agro').classList.remove('active');
}

function showAgronomist() {
    document.getElementById('market-section').classList.add('hidden');
    document.getElementById('agronomist-section').classList.remove('hidden');
    document.getElementById('nav-market').classList.remove('active');
    document.getElementById('nav-agro').classList.add('active');
}

// --- MARKET PREDICTION ---
async function predict() {
  const payload = {
    Commodity_Group: grp.value,
    Commodity: comm.value,
    MSP: +msp.value,
    Price_1DayAgo: +p1.value,
    Price_2DaysAgo: +p2.value,
    Arrival_Today: +a0.value,
    Arrival_1DayAgo: +a1.value,
    Arrival_2DaysAgo: +a2.value
  };

  const res = await fetch('/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });

  const data = await res.json();

    if (data.error) {
        alert("Server Error: " + data.error); // This will tell you if the model is missing!
        return;
    }

  resultBox.classList.remove('hidden');
  price.innerText = `₹ ${data.predicted_price_tomorrow}`;
  trend.className = 'trend ' + (data.trend === 'UP' ? 'up' : 'down');
  trend.innerText = data.trend === 'UP' ? 'Bullish Trend' : 'Bearish Trend';
}

// --- AI AGRONOMIST ---
let mediaRecorder;
let audioChunks = [];
let recordedBlob = null;

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        
        mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
        };
        
        mediaRecorder.onstop = () => {
            recordedBlob = new Blob(audioChunks, { type: 'audio/webm' });
            document.getElementById('recStatus').innerText = "✅ Recording saved!";
            document.getElementById('voiceNote').value = ""; // Clear file input
        };
        
        mediaRecorder.start();
        document.getElementById('startRecBtn').disabled = true;
        document.getElementById('stopRecBtn').disabled = false;
        document.getElementById('recStatus').innerText = "🔴 Recording... Speak now.";
        
    } catch (err) {
        alert("Microphone access denied or not available: " + err);
    }
}

function stopRecording() {
    if (mediaRecorder) {
        mediaRecorder.stop();
        document.getElementById('startRecBtn').disabled = false;
        document.getElementById('stopRecBtn').disabled = true;
    }
}

async function analyzeCrop() {
    const imgInput = document.getElementById('cropImage');
    const audInput = document.getElementById('voiceNote');
    const promptRef = document.getElementById('promptText');
    const langRef = document.getElementById('responseLang');
    const btn = document.getElementById('analyzeBtn');
    const errBox = document.getElementById('agroError');
    const resBox = document.getElementById('agroResult');
    const txtBox = document.getElementById('agroText');
    const audPlayer = document.getElementById('agroAudio');
    
    // Reset inputs
    errBox.style.display = 'none';
    resBox.classList.add('hidden');
    audPlayer.classList.add('hidden');
    
    if (!imgInput.files[0]) {
        errBox.innerText = "Please upload an image of the crop.";
        errBox.style.display = 'block';
        return;
    }
    
    btn.disabled = true;
    btn.innerHTML = '<iconify-icon icon="lucide:loader-2" class="spin"></iconify-icon> Analyzing...';
    
    try {
        const formData = new FormData();
        formData.append('image', imgInput.files[0]);
        
        // Prioritize recorded audio, else check file upload
        if (recordedBlob) {
            formData.append('audio', recordedBlob, 'recording.webm');
        } else if (audInput.files[0]) {
            formData.append('audio', audInput.files[0]);
        }
        
        formData.append('prompt', promptRef.value);
        formData.append('language', langRef.value);
        
        // 1. Analyze
        const res = await fetch('/assistant/analyze', {
            method: 'POST',
            body: formData
        });
        
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Analysis failed');
        
        // Show Text
        resBox.classList.remove('hidden');
        txtBox.innerText = data.response;
        
        // 2. TTS
        const ttsRes = await fetch('/assistant/text-to-speech', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                text: data.response,
                language: langRef.value
            })
        });
        
        if (ttsRes.ok) {
            const blob = await ttsRes.blob();
            const url = URL.createObjectURL(blob);
            audPlayer.src = url;
            audPlayer.classList.remove('hidden');
            audPlayer.play();
        } else {
            console.warn("TTS failed or not configured cleanly");
        }
        
    } catch (e) {
        errBox.innerText = e.message;
        errBox.style.display = 'block';
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<iconify-icon icon="lucide:scan-face"></iconify-icon> Analyze Crop';
    }
}
</script>

</body>
</html>

"""

# --- ROUTES ---

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

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