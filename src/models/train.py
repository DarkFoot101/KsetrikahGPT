import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import mlflow
import os
import sys
import warnings

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sklearn.model_selection import train_test_split
from src.utils.metrics import calculate_smape

warnings.filterwarnings('ignore')

# CONFIG
DATA_PATH = 'data/features/training_data.csv'  # <--- UPDATED PATH
MODEL_PATH = 'models/best_model.joblib'
MLFLOW_EXPERIMENT_NAME = "Agri_Price_Prediction"

def train():
    # Setup MLflow
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    print("🚀 Loading data for training...")
    if not os.path.exists(DATA_PATH):
        print(f" Data file not found: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    
    # Select only numeric features for LightGBM
    # We drop the Target (Price_Today) and non-numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    X = numeric_df.drop(columns=['Price_Today'])
    y = numeric_df['Price_Today']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with mlflow.start_run():
        print(" Training LightGBM...")
        
        params = {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 10,
            "num_leaves": 31,
            "random_state": 42
        }
        
        mlflow.log_params(params)
        
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="mae",
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        # Evaluate
        preds = model.predict(X_test)
        smape = calculate_smape(y_test, preds)
        
        print(f"✅ SMAPE Score: {smape:.4f}%")
        mlflow.log_metric("smape", smape)
        
        # Save
        joblib.dump(model, MODEL_PATH)
        print(f" Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()