# scripts/clean_new_data.py

import pandas as pd
import os

# Chemins
INPUT_FILE = "data/eurusd_2024.csv"
OUTPUT_FILE = "data/eurusd_2024_clean.csv"

print("🧹 Nettoyage des données 2024...")

# Chargement des données
df = pd.read_csv(INPUT_FILE, header=None, names=["Date", "Open", "High", "Low", "Close", "Volume"])


# Conversion de la colonne "Date"
df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

# Suppression des lignes avec valeurs manquantes
df = df.dropna()

# Réinitialisation de l'index
df.reset_index(drop=True, inplace=True)

# Sauvegarde du fichier nettoyé
df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Données nettoyées sauvegardées dans: {OUTPUT_FILE}")
