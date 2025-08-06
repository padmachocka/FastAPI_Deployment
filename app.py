from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

class Features(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

app = FastAPI()

# Load the model with joblib
model = joblib.load("./lr_model.pkl")

@app.post("/predict")
def predict(features: Features):
    data = features.dict()
    # Calculate bp_anomaly inside the API
    data['bp_anomaly'] = 1 if data['trestbps'] >= 140 or data['trestbps'] < 90 else 0

    input_df = pd.DataFrame([data])
    try:
        prediction = model.predict(input_df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"prediction": prediction.tolist()}