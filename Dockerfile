# Utilise une image officielle Python légère
FROM python:3.10-slim

# Dossier de travail à l'intérieur du conteneur
WORKDIR /app

# Copier les dépendances
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code source dans le conteneur
COPY . .

# Lancer le script de collecte de données
CMD ["python", "scripts/collect_forex.py"]

