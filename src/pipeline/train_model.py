import os
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

# === Configuration ===
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = "compagnon-immobilier"
EXPERIMENT_NAME = "xgboost_6_segments"

SEGMENTS = [
    "bas_appart",
    "bas_maison",
    "milieu_appart",
    "milieu_maison",
    "haut_appart",
    "haut_maison",
]

META_PATH = os.environ.get(
    "META_PATH",
    os.path.join(os.path.dirname(__file__), "../../notebooks/Models/meta_6modeles.pkl"),
)
MODELS_DIR = os.environ.get(
    "MODELS_DIR",
    os.path.join(os.path.dirname(__file__), "../../notebooks/Models"),
)
DATA_PATH = os.environ.get(
    "DATA_PATH",
    os.path.join(os.path.dirname(__file__), "../../dvf_clean_model_ready_enriched.csv"),
)


def load_meta() -> dict:
    with open(META_PATH, "rb") as f:
        return pickle.load(f)


def get_tranche(prix_m2: float, q33: float, q66: float) -> str:
    if prix_m2 < q33:
        return "bas"
    if prix_m2 <= q66:
        return "milieu"
    return "haut"


def get_production_model_mae(client: MlflowClient) -> float:
    """Récupère le MAE global du modèle actuellement en production."""
    try:
        model_version = client.get_model_version_by_alias(MODEL_NAME, "production")
        run = client.get_run(model_version.run_id)
        mae = run.data.metrics.get("mae_global", float("inf"))
        print(f"[train] Modèle production actuel v{model_version.version} — MAE global : {mae:.2f}")
        return mae
    except mlflow.exceptions.MlflowException:
        print("[train] Aucun modèle en production trouvé dans le registry.")
        return float("inf")


def register_and_promote(client: MlflowClient, run_id: str, new_mae: float, prod_mae: float) -> bool:
    """Enregistre le modèle et le promeut en production s'il est meilleur."""
    model_uri = f"runs:/{run_id}/model_bas_appart"
    result = mlflow.register_model(model_uri, MODEL_NAME)
    version = result.version
    print(f"[train] Modèle enregistré — version {version}")

    client.update_model_version(
        name=MODEL_NAME,
        version=version,
        description=f"MAE global : {new_mae:.2f} €/m²",
    )

    if new_mae < prod_mae:
        client.set_registered_model_alias(MODEL_NAME, "production", version)
        print(f"[train] [OK] Version {version} promue en PRODUCTION (MAE {new_mae:.2f} < {prod_mae:.2f})")
        return True
    else:
        client.set_registered_model_alias(MODEL_NAME, "challenger", version)
        print(f"[train] [--] Version {version} NOT promue - challenger (MAE {new_mae:.2f} >= {prod_mae:.2f})")
        return False


def train():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    meta = load_meta()
    q33 = meta["q33"]
    q66 = meta["q66"]
    features = meta["features"]

    print(f"[train] Chargement des données : {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"[train] {len(df):,} lignes chargées")

    # Calcul du prix au m² depuis valeur_fonciere / surface
    if "prix_m2" not in df.columns:
        if "valeur_fonciere" in df.columns:
            df["prix_m2"] = df["valeur_fonciere"] / df["surface_reelle_bati"].replace(0, float("nan"))
        else:
            raise ValueError("Impossible de calculer prix_m2 : colonne valeur_fonciere manquante.")

    TARGET = "prix_m2"
    meta_q33, meta_q66 = meta["prix_m2_min"] * 3, meta["prix_m2_max"] * 0.5
    prix_min = meta.get("prix_m2_min", 300)
    prix_max = meta.get("prix_m2_max", 15000)
    df = df[df[TARGET].between(prix_min, prix_max)].dropna(subset=[TARGET])

    df["tranche"] = df[TARGET].apply(lambda x: get_tranche(x, q33, q66))
    df["type_seg"] = df["is_maison"].apply(lambda x: "maison" if x == 1 else "appart") if "is_maison" in df.columns else "appart"
    df["segment"] = df["tranche"] + "_" + df["type_seg"]

    client = MlflowClient()
    prod_mae = get_production_model_mae(client)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"[train] Run MLflow démarré : {run_id}")

        mlflow.log_param("q33", q33)
        mlflow.log_param("q66", q66)
        mlflow.log_param("n_features", len(features))
        mlflow.log_param("n_segments", len(SEGMENTS))
        mlflow.log_param("total_samples", len(df))

        all_maes = []
        all_r2s = []
        models = {}

        for segment in SEGMENTS:
            df_seg = df[df["segment"] == segment].copy()
            if len(df_seg) < 100:
                print(f"[train] Segment {segment} — pas assez de données ({len(df_seg)}), ignoré.")
                continue

            # Features disponibles pour ce segment
            seg_features = [f for f in features if f in df_seg.columns]
            X = df_seg[seg_features].fillna(df_seg[seg_features].median())
            y = df_seg[TARGET]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                tree_method="hist",
            )
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))

            mlflow.log_metric(f"mae_{segment}", mae)
            mlflow.log_metric(f"r2_{segment}", r2)
            mlflow.log_metric(f"rmse_{segment}", rmse)
            mlflow.log_metric(f"n_train_{segment}", len(X_train))

            all_maes.append(mae)
            all_r2s.append(r2)
            models[segment] = model
            print(f"[train] {segment:20s} — MAE: {mae:.0f} €/m²  R²: {r2:.3f}  n={len(X_train):,}")

        mae_global = float(np.mean(all_maes))
        r2_global = float(np.mean(all_r2s))
        mlflow.log_metric("mae_global", mae_global)
        mlflow.log_metric("r2_global", r2_global)
        mlflow.log_metric("n_segments_trained", len(models))
        print(f"[train] MAE global : {mae_global:.2f} €/m²  —  R² global : {r2_global:.3f}")

        # Log des modèles comme artefacts MLflow
        for segment, model in models.items():
            mlflow.xgboost.log_model(model, artifact_path=f"model_{segment}")

        # Log du meta fichier
        mlflow.log_artifact(META_PATH, artifact_path="meta")

        promoted = register_and_promote(client, run_id, mae_global, prod_mae)

    return run_id, mae_global, promoted


if __name__ == "__main__":
    run_id, mae, promoted = train()
    print(f"\n=== Résultat ===")
    print(f"Run ID   : {run_id}")
    print(f"MAE      : {mae:.2f} €/m²")
    print(f"Promu    : {'[OK] OUI' if promoted else '[--] NON'}")
