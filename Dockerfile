# ⚠️ CHANGEMENT 1 : On passe à Python 3.12 pour correspondre à ton ordi
# C'est INDISPENSABLE pour éviter l'erreur de "pickle" / "code()"
FROM python:3.12.8-slim

# Dossier de travail
WORKDIR /app

# Installation des dépendances système (LightGBM et Numpy en ont besoin)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ⚠️ CHANGEMENT 2 : On copie le fichier "API" (le complet), pas le fichier Streamlit
COPY requirements_api.txt .

# Installation des dépendances Python
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements_api.txt

# On copie tout le reste du projet (dont le dossier model_prod)
COPY . .

# On expose le port standard de Render
EXPOSE 10000

# ⚠️ CHANGEMENT 3 : La commande de démarrage
# On ne lance plus uvicorn manuellement.
# On demande à MLflow de servir le dossier "model_prod" créé par le script.
CMD mlflow models serve -m model_prod -h 0.0.0.0 -p 10000 --no-conda