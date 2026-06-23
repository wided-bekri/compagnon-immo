import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import pickle
import os

DATA_PATH = "data/immobilier.csv"
MODEL_PATH = "models/model.pkl"

def train():
    print("Chargement des données...")
    df = pd.read_csv(DATA_PATH)
    X = df[["surface", "nb_pieces", "code_departement"]]
    y = df["prix"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        n_estimators = 100
        random_state = 42

        print("Entraînement du modèle...")
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"MAE : {mae:.2f} €")
        print(f"R2  : {r2:.4f}")

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="compagnon-immo"
        )

        os.makedirs("models", exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        print(f"Modèle sauvegardé dans {MODEL_PATH}")

if __name__ == "__main__":
    train()