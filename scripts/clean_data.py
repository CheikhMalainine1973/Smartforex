import pandas as pd
import os

#  Chemin vers le fichier brut
INPUT_FILE = "/app/data/eurusd_1y.csv"
OUTPUT_FILE = "/app/data/eurusd_clean.csv"

#  Lecture des données
df = pd.read_csv(INPUT_FILE, header=None, names=["Date", "Open", "High", "Low", "Close", "Volume"])

#  Nettoyage
df["Date"] = pd.to_datetime(df["Date"], errors='coerce')  # Conversion des dates
df.dropna(inplace=True)  # Suppression des lignes avec valeurs manquantes
df = df[df["Volume"] != 0]  # Suppression des lignes où volume = 0
df = df.sort_values("Date")  # Tri chronologique

#  Sauvegarde du fichier propre
df.to_csv(OUTPUT_FILE, index=False)

print(f"[✔] Données nettoyées sauvegardées dans: {OUTPUT_FILE}")
