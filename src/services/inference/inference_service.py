import os
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import mlflow
import mlflow.xgboost
import pandas as pd
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field, field_validator
from prometheus_client import Counter, Histogram, make_asgi_app

# === Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Configuration ===
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = "compagnon-immobilier"
IMMO_API_KEY = os.environ.get("IMMO_API_KEY", "")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# === Metriques Prometheus ===
PREDICTION_COUNT = Counter("immo_predictions_total", "Nombre total de predictions")
PREDICTION_LATENCY = Histogram("immo_prediction_latency_seconds", "Latence des predictions")
PREDICTION_ERRORS = Counter("immo_prediction_errors_total", "Nombre d erreurs de prediction")

# === Etat global ===
state = {
    "model": None,
    "model_version": None,
    "model_loaded": False,
    "features": None,
}

FEATURES = [
    "surface_reelle_bati",
    "nombre_pieces_principales",
    "surface_terrain",
    "annee",
    "mois",
    "code_departement",
    "longitude",
    "latitude",
    "is_maison",
    "is_neuf",
    "anciennete_mois",
    "surface_par_piece",
    "revenu_median",
    "taux_pauvrete",
    "nb_equipements_total",
    "population_2023",
    "evolution_pop_5_ans",
    "evolution_pop_10_ans",
    "taux_cambriolages",
    "taux_vols_total",
    "taux_violences_total",
    "commune_prix_m2",
    "commune_volume",
    "dept_prix_m2",
    "dept_prix_m2_maison",
    "commune_prix_m2_maison",
    "commune_volume_maison",
    "dept_prix_m2_appart",
    "commune_prix_m2_appart",
    "commune_volume_appart",
    "prix_estime_commune",
]


# === Schemas ===
class PredictionRequest(BaseModel):
    surface_reelle_bati: float = Field(..., gt=0, le=1000)
    nombre_pieces_principales: int = Field(3, ge=1, le=30)
    type_bien: str = Field("appart")
    surface_terrain: float = Field(0.0, ge=0)
    annee: int = Field(2025, ge=2020, le=2035)
    mois: int = Field(6, ge=1, le=12)
    code_departement: Optional[str] = Field(None, max_length=3)
    longitude: Optional[float] = None
    latitude: Optional[float] = None
    commune_prix_m2: float = Field(3000.0, ge=0)
    dept_prix_m2: float = Field(3000.0, ge=0)
    prix_estime_commune: float = Field(3000.0, ge=0)

    @field_validator("type_bien")
    @classmethod
    def normalize_type(cls, v: str) -> str:
        v = v.lower().strip()
        if v in ("appartement", "appart"):
            return "appart"
        if v in ("maison",):
            return "maison"
        raise ValueError("type_bien doit etre 'appart' ou 'maison'")


class PredictionResponse(BaseModel):
    prediction_eur_m2: float
    prediction_total_eur: float
    interval_low_eur: float
    interval_high_eur: float
    model_version: Optional[str]
    features_used: int


# === Chargement du modele ===
def load_model_from_mlflow() -> bool:
    client = mlflow.tracking.MlflowClient()
    try:
        model_version = client.get_model_version_by_alias(MODEL_NAME, "production")
        run_id = model_version.run_id
        uri = f"runs:/{run_id}/model"
        state["model"] = mlflow.xgboost.load_model(uri)
        state["model_version"] = f"v{model_version.version}"
        state["model_loaded"] = True
        if hasattr(state["model"], "feature_names_in_"):
            state["features"] = list(state["model"].feature_names_in_)
        else:
            state["features"] = FEATURES
        logger.info(f"Modele charge depuis MLflow registry (v{model_version.version})")
        return True
    except mlflow.exceptions.MlflowException as e:
        logger.error(f"Impossible de charger le modele : {e}")
        return False


# === Construction des features ===
def build_features(req: PredictionRequest) -> pd.DataFrame:
    is_maison = 1 if req.type_bien == "maison" else 0
    surface_par_piece = req.surface_reelle_bati / max(req.nombre_pieces_principales, 1)

    try:
        code_dep = int(req.code_departement.replace("A", "0").replace("B", "0")) if req.code_departement else 75
    except Exception:
        code_dep = 75

    row = {
        "surface_reelle_bati": req.surface_reelle_bati,
        "nombre_pieces_principales": req.nombre_pieces_principales,
        "surface_terrain": req.surface_terrain,
        "annee": req.annee,
        "mois": req.mois,
        "code_departement": code_dep,
        "longitude": req.longitude or 2.3,
        "latitude": req.latitude or 48.8,
        "is_maison": is_maison,
        "is_neuf": 0,
        "anciennete_mois": 360,
        "surface_par_piece": surface_par_piece,
        "revenu_median": 22000.0,
        "taux_pauvrete": 14.0,
        "nb_equipements_total": 50,
        "population_2023": 50000.0,
        "evolution_pop_5_ans": 0.0,
        "evolution_pop_10_ans": 0.0,
        "taux_cambriolages": 5.0,
        "taux_vols_total": 10.0,
        "taux_violences_total": 8.0,
        "commune_prix_m2": req.commune_prix_m2,
        "commune_volume": 100.0,
        "dept_prix_m2": req.dept_prix_m2,
        "dept_prix_m2_maison": req.dept_prix_m2 * 0.9,
        "commune_prix_m2_maison": req.commune_prix_m2 * 0.9,
        "commune_volume_maison": 50.0,
        "dept_prix_m2_appart": req.dept_prix_m2 * 1.1,
        "commune_prix_m2_appart": req.commune_prix_m2 * 1.1,
        "commune_volume_appart": 50.0,
        "prix_estime_commune": req.prix_estime_commune,
    }

    features = state["features"] or FEATURES
    X = pd.DataFrame([{f: row.get(f, 0.0) for f in features}])
    return X


# === Lifespan ===
@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Demarrage de l API Compagnon Immobilier...")
    load_model_from_mlflow()
    yield
    logger.info("Arret de l API.")


# === Application FastAPI ===
app = FastAPI(
    title="Compagnon Immobilier - API de Prediction",
    description="Predit le prix au m2 d un bien immobilier via XGBoost.",
    version="3.0.0",
    lifespan=lifespan,
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if IMMO_API_KEY and x_api_key != IMMO_API_KEY:
        raise HTTPException(status_code=401, detail="Cle API manquante ou invalide.")


# === Routes ===
@app.get("/health")
def health():
    return {
        "status": "ok" if state["model_loaded"] else "degraded",
        "model_loaded": state["model_loaded"],
        "model_version": state["model_version"],
        "model_name": MODEL_NAME,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest, x_api_key: Optional[str] = Header(default=None)):
    require_api_key(x_api_key)

    if not state["model_loaded"]:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=503, detail="Modele non disponible.")

    start = time.time()
    try:
        X = build_features(req)
        residuel = float(state["model"].predict(X)[0])
        prix_m2 = residuel + req.commune_prix_m2
        prix_m2 = max(300.0, min(15000.0, prix_m2))
        prix_total = prix_m2 * req.surface_reelle_bati

        PREDICTION_COUNT.inc()
        PREDICTION_LATENCY.observe(time.time() - start)

        return PredictionResponse(
            prediction_eur_m2=round(prix_m2, 2),
            prediction_total_eur=round(prix_total, 2),
            interval_low_eur=round(prix_total * 0.85, 2),
            interval_high_eur=round(prix_total * 1.15, 2),
            model_version=state["model_version"],
            features_used=len(state["features"] or FEATURES),
        )

    except Exception as e:
        PREDICTION_ERRORS.inc()
        logger.error(f"Erreur prediction : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur de prediction : {str(e)}")


@app.post("/reload_model")
def reload_model(x_api_key: Optional[str] = Header(default=None)):
    require_api_key(x_api_key)
    success = load_model_from_mlflow()
    if not success:
        raise HTTPException(status_code=503, detail="Impossible de recharger le modele depuis MLflow.")
    return {
        "status": "ok",
        "message": "Modele rechargé depuis MLflow registry (alias production).",
        "model_version": state["model_version"],
    }


@app.get("/model/info")
def model_info():
    client = mlflow.tracking.MlflowClient()
    try:
        model_version = client.get_model_version_by_alias(MODEL_NAME, "production")
        run = client.get_run(model_version.run_id)
        return {
            "model_name": MODEL_NAME,
            "production_version": model_version.version,
            "run_id": model_version.run_id,
            "metrics": {
                "mae": run.data.metrics.get("mae"),
                "r2": run.data.metrics.get("r2"),
                "rmse": run.data.metrics.get("rmse"),
            },
            "params": {
                "n_estimators": run.data.params.get("n_estimators"),
                "max_depth": run.data.params.get("max_depth"),
                "learning_rate": run.data.params.get("learning_rate"),
                "n_features": run.data.params.get("n_features"),
            },
        }
    except mlflow.exceptions.MlflowException:
        return {
            "model_name": MODEL_NAME,
            "production_version": None,
            "info": "Aucun modele en production.",
        }
