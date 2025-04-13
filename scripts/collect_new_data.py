# scripts/collect_new_data.py

import yfinance as yf
import os

# Crée le dossier s'il n'existe pas
os.makedirs("data", exist_ok=True)

# Spécifie la paire et la période
symbol = "EURUSD=X"
start_date = "2024-01-01"
end_date = "2024-12-31"

print("📥 Téléchargement des données 2024...")

# Téléchargement des données avec ajustements automatiques
df = yf.download(symbol, start=start_date, end=end_date, interval="1d", auto_adjust=True)

# Sauvegarde du fichier
output_file = "data/eurusd_2024.csv"
df.to_csv(output_file)

print(f"✅ Données sauvegardées dans {output_file}")
