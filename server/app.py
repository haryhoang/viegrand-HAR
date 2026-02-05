import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import numpy as np
import os

# Call extract_peak from utils.py
from utils import extract_peak

app = FastAPI(title="viegrand_HAR")

# --- 1. LOAD MODEL & CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '../models')

try:
    model = joblib.load(os.path.join(MODEL_DIR, 'fall_detection_rf.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# --- 2. Dữ liệu ESP ---
class SensorData(BaseModel):
    # ESP32 gửi lên một danh sách (mảng) các giá trị
    ax: List[float]
    ay: List[float]
    az: List[float]
    gx: List[float]
    gy: List[float]
    gz: List[float]
    # fs: int = 50 (we use 50 signs for 1 second (extract features for 1 second (Hz)))

# --- 3. API ENDPOINT ---
@app.post("/predict/fall")
def predict_fall(data: SensorData):
    
    try:
        df_raw = pd.DataFrame({
            'ax': data.ax, 'ay': data.ay, 'az': data.az,
            'gx': data.gx, 'gy': data.gy, 'gz': data.gz
        })

        if len(df_raw) < 50:
            return {"error": "Not enough data. Need at least 1 second (50 samples)."}

        # Trích xuất đặc trưng (Feature Extraction)
        features_df = extract_peak(df_raw, fs=50)

        for col in feature_names:
            if col not in features_df.columns:
                features_df[col] = 0
        features_df = features_df[feature_names]

        # Scale dữ liệu
        features_scaled = scaler.transform(features_df)

        # Dự đoán (Predict)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1] # Xác suất ngã

        # Logic Stroke Risk (Dựa trên Post-fall Features đã extract)
        is_stroke_risk = False
        post_std = features_df['acc_post_std'].values[0]
        
        if prediction == 1: 
            if post_std < 0.5: # Ngưỡng bất động 
                is_stroke_risk = True

        return {
            "status": "success",
            "is_fall": int(prediction),
            "fall_probability": round(float(probability), 4),
            "stroke_risk": is_stroke_risk,
            "details": {
                "impact_max": float(features_df['impact_max'].values[0]),
                "post_stability": float(post_std)
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
