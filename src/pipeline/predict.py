import mlflow.pyfunc

MODEL_NAME = "compagnon-immo"
MODEL_ALIAS = "production"

def load_model():
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model

def predict(surface, nb_pieces, code_departement):
    model = load_model()
    import pandas as pd
    data = pd.DataFrame([{
        "surface": surface,
        "nb_pieces": nb_pieces,
        "code_departement": code_departement
    }])
    prediction = model.predict(data)
    return round(float(prediction[0]), 2)

if __name__ == "__main__":
    prix = predict(65, 3, 75)
    print(f"Prix estimé : {prix} €")