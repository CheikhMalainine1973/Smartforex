import pandas as pd
import joblib
import os

#  Chemins
DATA_FILE = os.getenv("CLEAN_DATA_FILE", "data/eurusd_clean.csv")
MODEL_FILE = os.getenv("MODEL_FILE", "models/model.pkl")

#  Chargement du modèle
model = joblib.load(MODEL_FILE)

#  Chargement des données
df = pd.read_csv(DATA_FILE)

#  Préparation des features pour la prédiction
X = df[["Open", "High", "Low", "Volume"]]

#  Prédiction des prix de clôture
predictions = model.predict(X)

#  Affichage des résultats
df["Predicted_Close"] = predictions

print(df[["Date", "Close", "Predicted_Close"]].tail(10))  # Affiche les 10 dernières lignes
