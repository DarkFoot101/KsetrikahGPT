import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# CONFIG
INPUT_PATH = "data/processed/clean_data.csv"
OUTPUT_PATH = "data/features/training_data.csv"
ENCODER_PATH = "models/encoders.joblib"

def build_features():
    print("🏗️  Building Features...")
    
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Error: {INPUT_PATH} not found. Run preprocess first.")
        return

    try:
        df = pd.read_csv(INPUT_PATH)
        
        # --- 1. Feature Engineering ---
        # Economic Signal: Is price above MSP?
        if 'MSP' in df.columns and 'Price_Today' in df.columns:
            df['msp_premium'] = df['Price_Today'] - df['MSP']
        
        # Momentum: Price change (Last Week vs Last Month proxies)
        # Note: Depending on your raw data columns. 
        # Using Price_1DayAgo and Price_2DaysAgo from your standardized schema:
        if 'Price_1DayAgo' in df.columns and 'Price_2DaysAgo' in df.columns:
            df['price_momentum'] = (df['Price_1DayAgo'] - df['Price_2DaysAgo']) / (df['Price_2DaysAgo'] + 1e-9)
            
            # Volatility (Std Dev of recent prices)
            price_cols = ['Price_Today', 'Price_1DayAgo', 'Price_2DaysAgo']
            df['price_volatility'] = df[price_cols].std(axis=1)

        # --- 2. Encoding & Saving Encoders ---
        # This is the part we cannot skip. We must remember that "Gujarat" = 5.
        encoders = {}
        cat_cols = ['Commodity_Group', 'Commodity', 'Variety']
        
        for col in cat_cols:
            if col in df.columns:
                le = LabelEncoder()
                # Convert to string to be safe, then encode
                df[f'{col}_Encoded'] = le.fit_transform(df[col].astype(str))
                # Store the encoder for later use in API
                encoders[col] = le
        
        # Save Encoders
        os.makedirs(os.path.dirname(ENCODER_PATH), exist_ok=True)
        joblib.dump(encoders, ENCODER_PATH)
        print(f"💾 Encoders saved to {ENCODER_PATH}")

        # --- 3. Final Prep ---
        # Drop rows with NaNs (created by lag features)
        df_final = df.dropna()
        
        # Save to features folder
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        df_final.to_csv(OUTPUT_PATH, index=False)
        print(f"✅ Features ready: {df_final.shape} rows saved to {OUTPUT_PATH}")

    except Exception as e:
        print(f"❌ Failed to build features: {e}")

if __name__ == "__main__":
    build_features()