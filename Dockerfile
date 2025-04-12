# Étape 1 : Choisir une image Python légère
FROM python:3.10-slim

# Étape 2 : Définir le répertoire de travail dans le container
WORKDIR /app

# Étape 3 : Copier les dépendances et installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Étape 4 : Copier le code source
COPY . .

# Étape 5 : Commande par défaut exécutée quand on lance le container
CMD ["bash", "-c", "python scripts/collect_forex.py && python scripts/clean_data.py"]


