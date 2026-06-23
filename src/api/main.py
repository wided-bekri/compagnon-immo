from fastapi import FastAPI
from src.api.schemas import PredictionRequest, PredictionResponse
import pickle
import os

app = FastAPI(title="Compagnon Immobilier")

MODEL_PATH = "models/model.pkl"

@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API Compagnon Immobilier"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    prediction = model.predict([[
        request.surface,
        request.nb_pieces,
        request.code_departement
    ]])
    return PredictionResponse(prix_estime=round(prediction[0], 2))