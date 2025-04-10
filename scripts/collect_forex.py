import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import os

# Créer dossier s'il n'existe pas
os.makedirs("data", exist_ok=True)

# Télécharger les données EUR/USD
print("📥 Téléchargement des données...")
df = yf.download("EURUSD=X", start="2023-01-01", end="2024-01-01", interval="1d")

# Sauvegarder dans un CSV
df.to_csv("data/eurusd_1y.csv")
print("✅ Données sauvegardées dans data/eurusd_1y.csv")

# Visualisation du prix de clôture
plt.figure(figsize=(10, 4))
plt.plot(df["Close"], label="Clôture EUR/USD")
plt.title("Évolution du cours EUR/USD")
plt.xlabel("Date")
plt.ylabel("Prix")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

