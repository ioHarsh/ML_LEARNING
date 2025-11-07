# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import numpy as np

MODEL_PATH = os.path.join("models", "model.pkl")

class PredictRequest(BaseModel):
    data: list  # list of rows; each row is a list of features

app = FastAPI(title="Local ML API")

_model_artifact = None

def load_model():
    global _model_artifact
    if os.path.exists(MODEL_PATH):
        try:
            _model_artifact = joblib.load(MODEL_PATH)
            print("Loaded model artifact.")
        except Exception as e:
            print("Error loading model artifact:", e)
            _model_artifact = None
    else:
        print("Model file not found at models/model.pkl. Run train.py to create it first.")
        _model_artifact = None

# load model at startup (if present)
load_model()

@app.get("/")
def root():
    return {"status": "ok", "model_loaded": _model_artifact is not None}

@app.post("/predict")
def predict(request: PredictRequest):
    if _model_artifact is None:
        # helpful message to user instead of crashing
        raise HTTPException(status_code=503, detail="Model not loaded. Run python train.py to train and save a model.")

    model = _model_artifact.get("model")
    scaler = _model_artifact.get("scaler")
    feature_names = _model_artifact.get("feature_names", None)

    arr = np.array(request.data)
    if arr.ndim == 1:
        # single row passed as flat list -> reshape to (1, n_features)
        arr = arr.reshape(1, -1)

    # optional: verify shape
    if scaler is not None:
        try:
            arr = scaler.transform(arr)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error transforming input: {e}")

    try:
        preds = model.predict(arr)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {e}")

    return {"predictions": preds.tolist()}
