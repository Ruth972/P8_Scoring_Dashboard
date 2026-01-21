# 1. Image de base : On part d'une version légère de Python
# (Utilise 3.9 ou 3.10 selon ta version locale, 3.9 est très stable pour le ML)
FROM python:3.9-slim

# 2. Dossier de travail dans le conteneur
WORKDIR /app

# 3. Installation des dépendances système
# 'build-essential' est nécessaire pour compiler certaines librairies Python
# 'libgomp1' est souvent requis par Scikit-Learn/XGBoost pour les calculs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copie et installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 5. --- ÉTAPE CRUCIALE (Le cœur du système) ---
# On copie TOUT ton dossier projet (main.py, le dossier mlruns, etc.)
# C'est grâce à ça que le modèle se retrouve DANS le conteneur.
COPY . .

# 6. Configuration de MLflow pour qu'il regarde au bon endroit
# On lui dit : "Tes données sont ici, en local dans le conteneur"
ENV MLFLOW_TRACKING_URI=file:///app/mlruns

# 7. Commande de démarrage
# On utilise la variable $PORT fournie automatiquement par Render
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]