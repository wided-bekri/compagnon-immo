import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os

# Chemins des fichiers
DATA_PATH = "data/immobilier.csv"
MODEL_PATH = "models/model.pkl"

def train():
    # 1. Charger les données
    print("Chargement des données...")
    df = pd.read_csv(DATA_PATH)

    # 2. Préparer les features et la cible
    X = df[["surface", "nb_pieces", "code_departement"]]
    y = df["prix"]

    # 3. Séparer en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Entraîner le modèle
    print("Entraînement du modèle...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5. Évaluer le modèle
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MAE : {mae:.2f} €")
    print(f"R2  : {r2:.4f}")

    # 6. Sauvegarder le modèle
    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Modèle sauvegardé dans {MODEL_PATH}")

if __name__ == "__main__":
    train()