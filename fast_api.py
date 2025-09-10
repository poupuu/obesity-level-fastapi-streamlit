from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np 
import obesity_base
from age_cleaner import AgeCleaner 

class ObesityPredictRequest(BaseModel):
    Gender: str = Field(..., example="Male", description="Gender of the individual")
    Age: float = Field(..., example=31.0, description="Age in years (will be cleaned and converted to float)")
    Height: float = Field(..., example=1.87, description="Height in meters")
    Weight: float = Field(..., example=128.87, description="Weight in kilograms")
    family_history_with_overweight: str = Field(..., example="yes", description="Does the family have a history of overweight? (yes/no)")
    FAVC: str = Field(..., example="yes", description="Frequent consumption of high caloric food (yes/no)")
    FCVC: float = Field(..., example=2.96, description="Frequency of consumption of vegetables (1-3)")
    NCP: float = Field(..., example=3.00, description="Number of main meals (1-4)")
    CAEC: str = Field(..., example="Sometimes", description="Consumption of food between meals (no, Sometimes, Frequently, Always)")
    SMOKE: str = Field(..., example="yes", description="Smokes (yes/no)")
    CH2O: float = Field(..., example=1.28, description="Consumption of water daily (1-3)")
    SCC: str = Field(..., example="no", description="Calories consumption monitoring (yes/no)")
    FAF: float = Field(..., example=0.90, description="Physical activity frequency (0-3)")
    TUE: float = Field(..., example=1.875, description="Time using technology devices (0-2)")
    CALC: str = Field(..., example="Sometimes", description="Consumption of alcohol (no, Sometimes, Frequently, Always)")
    MTRANS: str = Field(..., example="Automobile", description="Transportation used (Automobile, Motorbike, Bike, Public_Transportation, Walking)")

app = FastAPI(
    title="Obesity Prediction API",
    description="API for predicting obesity levels based on various health and lifestyle factors. "
                "The model uses an XGBoost Classifier and includes data preprocessing steps."
)

model_pipeline = None
target_encoder = None

@app.on_event("startup")
async def load_model():
    global model_pipeline, target_encoder
    model_pipeline, target_encoder = obesity_base.load_model_and_encoder()
    if model_pipeline is None or target_encoder is None:
        raise RuntimeError("FastAPI startup failed: Model and/or encoder could not be loaded.")
    print("FastAPI application started and models loaded.")

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Obesity Prediction API!",
        "instructions": "Send a POST request to the /predict endpoint with your data in JSON format.",
        "documentation": "Visit /docs for interactive API documentation."
    }

@app.post("/predict")
async def predict_obesity(request: ObesityPredictRequest):
    if model_pipeline is None or target_encoder is None:
        raise HTTPException(status_code=500, detail="Prediction model is not loaded. Please contact support.")
    try:
        input_data_dict = request.model_dump()
        input_df = pd.DataFrame([input_data_dict], columns=obesity_base.feature_columns)
        prediction_encoded = model_pipeline.predict(input_df)
        predicted_label = target_encoder.inverse_transform(prediction_encoded.reshape(-1, 1))[0][0]

        return {"predicted_obesity_level": predicted_label}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed due to invalid input or internal error: {e}")