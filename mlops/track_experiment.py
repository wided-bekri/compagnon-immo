"""
Enregistre les métriques des 6 modèles XGBoost dans MLflow à partir du CSV de résultats.

Usage:
    python mlops/track_experiment.py

Variables d'environnement:
    MLFLOW_TRACKING_URI  (défaut: http://localhost:5000)
    RESULTS_CSV          (défaut: notebooks/Models/result_xgboost_6modeles.csv)
"""
import os
import pandas as pd
import mlflow

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
RESULTS_CSV = os.environ.get(
    "RESULTS_CSV",
    os.path.join(os.path.dirname(__file__), "..", "notebooks", "Models", "result_xgboost_6modeles.csv"),
)
EXPERIMENT_NAME = "compagnon-immobilier-xgboost"


def log_segment_run(row: pd.Series) -> str:
    """Crée un run MLflow pour un segment et retourne son run_id."""
    with mlflow.start_run(run_name=f"xgboost_{row['segment']}") as run:
        mlflow.set_tag("segment", row["segment"])
        mlflow.set_tag("model_type", "XGBoost")

        metrics_map = {
            "mae": "mae",
            "rmse": "rmse",
            "r2": "r2",
            "mape": "mape",
        }
        for col, metric_name in metrics_map.items():
            if col in row and pd.notna(row[col]):
                mlflow.log_metric(metric_name, float(row[col]))

        params_map = {
            "n_estimators": "n_estimators",
            "max_depth": "max_depth",
            "learning_rate": "learning_rate",
            "n_features": "n_features",
        }
        for col, param_name in params_map.items():
            if col in row and pd.notna(row[col]):
                mlflow.log_param(param_name, row[col])

        return run.info.run_id


def main():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    if not os.path.exists(RESULTS_CSV):
        raise FileNotFoundError(f"Fichier de résultats introuvable : {RESULTS_CSV}")

    df = pd.read_csv(RESULTS_CSV)
    print(f"Chargement de {len(df)} segment(s) depuis {RESULTS_CSV}")

    for _, row in df.iterrows():
        run_id = log_segment_run(row)
        segment = row.get("segment", "inconnu")
        mae = row.get("mae", "N/A")
        print(f"  [OK] segment={segment}  MAE={mae}  run_id={run_id[:8]}...")

    print(f"\nExpérience '{EXPERIMENT_NAME}' enregistrée dans MLflow ({MLFLOW_URI}).")


if __name__ == "__main__":
    main()
