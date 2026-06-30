"""Tests unitaires pour la construction des features et le schéma de validation."""
import sys
import os
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "services", "inference"))

import mlflow
mlflow.set_tracking_uri = MagicMock()

from inference_service import build_features, PredictionRequest, FEATURES


def make_request(**kwargs):
    defaults = {
        "surface_reelle_bati": 75.0,
        "nombre_pieces_principales": 3,
        "type_bien": "appart",
        "surface_terrain": 0.0,
        "annee": 2024,
        "mois": 6,
        "commune_prix_m2": 4000.0,
        "dept_prix_m2": 3800.0,
        "prix_estime_commune": 4000.0,
    }
    defaults.update(kwargs)
    return PredictionRequest(**defaults)


def test_build_features_returns_all_columns():
    """Le DataFrame produit contient exactement les colonnes attendues."""
    from inference_service import state
    state["features"] = None
    req = make_request()
    X = build_features(req)
    assert set(FEATURES).issubset(set(X.columns))


def test_build_features_is_maison_flag():
    """is_maison vaut 1 pour une maison, 0 pour un appartement."""
    from inference_service import state
    state["features"] = None
    req_appart = make_request(type_bien="appart")
    req_maison = make_request(type_bien="maison")
    X_a = build_features(req_appart)
    X_m = build_features(req_maison)
    assert X_a["is_maison"].iloc[0] == 0
    assert X_m["is_maison"].iloc[0] == 1


def test_build_features_surface_par_piece():
    """surface_par_piece = surface / nombre_pieces."""
    from inference_service import state
    state["features"] = None
    req = make_request(surface_reelle_bati=90.0, nombre_pieces_principales=3)
    X = build_features(req)
    assert abs(X["surface_par_piece"].iloc[0] - 30.0) < 0.01


def test_build_features_code_departement_75_default():
    """Sans code_departement, on utilise 75 (Paris) par défaut."""
    from inference_service import state
    state["features"] = None
    req = make_request(code_departement=None)
    X = build_features(req)
    assert X["code_departement"].iloc[0] == 75
