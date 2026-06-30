"""Tests unitaires et d'intégration pour l'API FastAPI Compagnon Immobilier."""
import sys
import os
import pytest
from unittest.mock import MagicMock, patch

# Ajout du path pour accéder aux modules src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "services", "inference"))

# Mock MLflow avant l'import du module principal
import mlflow
mlflow.set_tracking_uri = MagicMock()

from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Crée un client de test avec un modèle mocké."""
    with patch("inference_service.load_model_from_mlflow") as mock_load:
        mock_load.return_value = True
        from inference_service import app, state
        # Simuler un modèle chargé
        mock_model = MagicMock()
        mock_model.predict.return_value = [500.0]
        mock_model.feature_names_in_ = None
        state["model"] = mock_model
        state["model_loaded"] = True
        state["model_version"] = "v1"
        state["features"] = None
        yield TestClient(app)


def test_health_ok(client):
    """L'endpoint /health retourne status ok quand le modèle est chargé."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_basic(client):
    """Une prédiction simple retourne les champs attendus."""
    payload = {
        "surface_reelle_bati": 80.0,
        "nombre_pieces_principales": 3,
        "type_bien": "appart",
        "commune_prix_m2": 3000.0,
        "dept_prix_m2": 3000.0,
        "prix_estime_commune": 3000.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction_eur_m2" in data
    assert "prediction_total_eur" in data
    assert "interval_low_eur" in data
    assert "interval_high_eur" in data
    assert data["interval_low_eur"] < data["prediction_total_eur"]
    assert data["interval_high_eur"] > data["prediction_total_eur"]


def test_predict_surface_negative(client):
    """Surface négative ou nulle → erreur 422."""
    payload = {
        "surface_reelle_bati": -10.0,
        "commune_prix_m2": 3000.0,
        "dept_prix_m2": 3000.0,
        "prix_estime_commune": 3000.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_type_bien_invalide(client):
    """Type de bien inconnu → erreur 422."""
    payload = {
        "surface_reelle_bati": 60.0,
        "type_bien": "studio",
        "commune_prix_m2": 3000.0,
        "dept_prix_m2": 3000.0,
        "prix_estime_commune": 3000.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_maison(client):
    """Type maison accepté sans erreur."""
    payload = {
        "surface_reelle_bati": 120.0,
        "nombre_pieces_principales": 5,
        "type_bien": "maison",
        "surface_terrain": 400.0,
        "commune_prix_m2": 2500.0,
        "dept_prix_m2": 2500.0,
        "prix_estime_commune": 2500.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction_eur_m2"] > 0


def test_health_model_version(client):
    """L'endpoint /health expose la version du modèle."""
    response = client.get("/health")
    data = response.json()
    assert "model_version" in data
    assert data["model_version"] == "v1"
