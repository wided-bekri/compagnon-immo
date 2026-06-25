import os
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

# === Configuration ===
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = "compagnon-immobilier"
EXPERIMENT_NAME = "xgboost_single_model"

MODELS_DIR = os.environ.get(
    "MODELS_DIR",
    os.path.join(os.path.dirname(__file__), "../../notebooks/Models"),
)

XGB_PARAMS = {
    "n_estimators": 800,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "objective": "reg:squarederror",
    "n_jobs": -1,
    "random_state": 42,
}


def get_production_model_mae(client: MlflowClient) -> float:
    try:
        model_version = client.get_model_version_by_alias(MODEL_NAME, "production")
        run = client.get_run(model_version.run_id)
        mae = run.data.metrics.get("mae", float("inf"))
        print(f"[train] Modele production actuel v{model_version.version} - MAE : {mae:.2f}")
        return mae
    except mlflow.exceptions.MlflowException:
        print("[train] Aucun modele en production - premier entrainement.")
        return float("inf")


def register_and_promote(client: MlflowClient, run_id: str, new_mae: float, prod_mae: float):
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, MODEL_NAME)
    version = result.version
    print(f"[train] Modele enregistre - version {version}")

    client.update_model_version(
        name=MODEL_NAME,
        version=version,
        description=f"MAE : {new_mae:.2f} euros/m2",
    )

    if new_mae < prod_mae:
        client.set_registered_model_alias(MODEL_NAME, "production", version)
        print(f"[train] [OK] Version {version} promue en PRODUCTION (MAE {new_mae:.2f} < {prod_mae:.2f})")
    else:
        client.set_registered_model_alias(MODEL_NAME, "challenger", version)
        print(f"[train] [--] Version {version} en CHALLENGER (MAE {new_mae:.2f} >= {prod_mae:.2f})")


def train():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Chargement des données déjà préparées
    print(f"[train] Chargement des données depuis : {MODELS_DIR}")
    X_train = pd.read_csv(os.path.join(MODELS_DIR, "X_train_optimized.csv"))
    X_test  = pd.read_csv(os.path.join(MODELS_DIR, "X_test_optimized.csv"))
    y_train = pd.read_csv(os.path.join(MODELS_DIR, "y_train_optimized.csv")).values.ravel()
    y_test  = pd.read_csv(os.path.join(MODELS_DIR, "y_test_optimized.csv")).values.ravel()

    print(f"[train] X_train : {X_train.shape}")
    print(f"[train] X_test  : {X_test.shape}")

    client = MlflowClient()
    prod_mae = get_production_model_mae(client)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"[train] Run MLflow demarre : {run_id}")

        for k, v in XGB_PARAMS.items():
            mlflow.log_param(k, v)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train", len(X_train))

        # Entrainement
        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # Evaluation — reconstruction du prix_m2 reel
        preds = model.predict(X_test)
        commune_px = X_test["commune_prix_m2"].values
        preds_prix_m2 = preds + commune_px
        true_prix_m2  = y_test + commune_px

        mae  = mean_absolute_error(true_prix_m2, preds_prix_m2)
        r2   = r2_score(true_prix_m2, preds_prix_m2)
        rmse = np.sqrt(mean_squared_error(true_prix_m2, preds_prix_m2))

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("rmse", rmse)

        print(f"[train] MAE  : {mae:.2f} euros/m2")
        print(f"[train] R2   : {r2:.4f}")
        print(f"[train] RMSE : {rmse:.2f}")

        # Sauvegarde pkl
        pkl_path = os.path.join(MODELS_DIR, "model_xgboost_single.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(model, f)
        print(f"[train] Modele sauvegarde : {pkl_path}")

        # Log dans MLflow
        mlflow.xgboost.log_model(model, name="model")

        # Enregistrement et promotion
        register_and_promote(client, run_id, mae, prod_mae)

    return run_id, mae, r2


if __name__ == "__main__":
    run_id, mae, r2 = train()
    print(f"\n=== Resultat ===")
    print(f"Run ID : {run_id}")
    print(f"MAE    : {mae:.2f} euros/m2")
    print(f"R2     : {r2:.4f}")
