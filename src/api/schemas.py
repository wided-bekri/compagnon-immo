from pydantic import BaseModel

class PredictionRequest(BaseModel):
    surface: float
    nb_pieces: int
    code_departement: int

class PredictionResponse(BaseModel):
    prix_estime: float