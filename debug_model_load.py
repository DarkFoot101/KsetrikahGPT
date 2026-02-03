
import os
import joblib
import sys

MODEL_PATH = "/home/akhi/Desktop/KsetrikahGPT/KsetrikahGPT/models/best_model.joblib"

try:
    print(f"Attempting to load model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
    print("Model type:", type(model))
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()
