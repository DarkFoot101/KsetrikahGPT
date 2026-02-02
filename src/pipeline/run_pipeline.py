import sys
import os

# Add project root to python path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.fetch_data import fetch_daily_data
from src.data.preprocess import clean_data
from src.features.build_features import build_features
from src.models.train import train

def run_pipeline():
    print("="*50)
    print("🚜 KsetrikahGPT: STARTING PIPELINE")
    print("="*50)
    
    # 1. Fetch
    print("\n[STEP 1] Fetching Data...")
    fetch_daily_data()
    
    # 2. Clean
    print("\n[STEP 2] Cleaning Data...")
    clean_data()
    
    # 3. Features
    print("\n[STEP 3] Engineering Features...")
    build_features()
    
    # 4. Train
    print("\n[STEP 4] Training Model...")
    train()
    
    print("\n" + "="*50)
    print("✅ PIPELINE FINISHED")
    print("="*50)

if __name__ == "__main__":
    run_pipeline()