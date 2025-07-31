from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd

# Define your feature schema expected for prediction
# Adjust these fields to match your model's training features
class Features(BaseModel):
    # Example features; replace these with your actual feature names and types
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

model_name = "Logistic Regression"
model_stage = "challenger"  # Change stage if required
model_uri = f"models:/{model_name}@{model_stage}"

# Load model once when app starts
model = mlflow.sklearn.load_model(model_uri)

@app.post("/predict")
def predict(features: Features):
    # Convert input features to DataFrame (shape: 1 row, columns as features)
    input_df = pd.DataFrame([features.dict()])

    # Optionally ensure columns are in correct order as expected by the model
    # input_df = input_df[training_feature_list]

    try:
        prediction = model.predict(input_df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"prediction": prediction.tolist()}

