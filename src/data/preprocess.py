import pandas as pd
import numpy as np
import os
import glob

# CONFIG
RAW_DIR = "data/raw"
PROCESSED_PATH = "data/processed/clean_data.csv"

def clean_data():
    print("🧹 Starting Data Cleaning...")
    
    # 1. Find the latest file in data/raw (regardless of name)
    list_of_files = glob.glob(f'{RAW_DIR}/*.csv') 
    if not list_of_files:
        print("❌ No raw data found!")
        return
    
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"📄 Processing latest file: {latest_file}")
    
    try:
        # header=2 skips the first two title rows
        df = pd.read_csv(latest_file, header=2)
        
        # Standardize Columns
        new_columns = [
            'Commodity_Group', 'Commodity', 'Variety', 'MSP',
            'Price_Today', 'Price_1DayAgo', 'Price_2DaysAgo',
            'Arrival_Today', 'Arrival_1DayAgo', 'Arrival_2DaysAgo'
        ]
        
        # Ensure we grab the first 10 columns
        df = df.iloc[:, :10] 
        
        # Rename columns if counts match, else force assignment
        if len(df.columns) == len(new_columns):
            df.columns = new_columns
        else:
            # Fallback for slight schema mismatches
            df.columns = new_columns[:len(df.columns)]

        # Handle Hyphens & Types
        numeric_cols = ['MSP', 'Price_Today', 'Price_1DayAgo', 'Price_2DaysAgo', 
                        'Arrival_Today', 'Arrival_1DayAgo', 'Arrival_2DaysAgo']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].replace('-', np.nan)
                df[col] = df[col].astype(str).str.replace(',', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop empty rows where we have no price
        df = df.dropna(subset=['Price_Today'])
        
        # Save
        df.to_csv(PROCESSED_PATH, index=False)
        print(f"✅ Clean data saved to {PROCESSED_PATH}")
        
    except Exception as e:
        print(f"❌ Error in cleaning: {e}")

if __name__ == "__main__":
    clean_data()