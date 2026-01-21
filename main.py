import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn
import shap  # --- P8 ADDITION : Import de SHAP
import numpy as np
import os

# Initialisation de l'application FastAPI
app = FastAPI(
    title="API Scoring Crédit & Explainability",
    description="Microservice de prédiction du risque de crédit avec explicabilité SHAP.",
    version="1.1.0"
)

# --- CONFIGURATION MLOPS ---
MODEL_PATH = "./mlruns/9/models/m-0a84d69a2e314f0e82736c01fbcdd540/artifacts"

# --- BLOC DE DÉBOGAGE (A supprimer plus tard) ---
print(f"Chemin actuel (CWD) : {os.getcwd()}")
if os.path.exists(MODEL_PATH):
    print(f"✅ Le dossier existe. Contenu : {os.listdir(MODEL_PATH)}")
else:
    print(f"❌ Le chemin {MODEL_PATH} est introuvable !")
    print("Contenu de la racine :", os.listdir("."))
    if os.path.exists("mlruns"):
        print("Contenu de mlruns :", os.listdir("mlruns"))
# ------------------------------------------------

# --- CHARGEMENT DU MODÈLE ET EXPLAINER ---
print(f"Initialisation : Chargement du modèle depuis {MODEL_PATH}...")
model = None
explainer = None  # --- P8 ADDITION : Variable pour l'explainer

try:
    model = mlflow.sklearn.load_model(MODEL_PATH)
    print("Succès : Le modèle de scoring est chargé.")
    
    # --- P8 ADDITION : Initialisation de l'explainer SHAP ---
    # On tente de créer un TreeExplainer (optimisé pour XGBoost/LGBM/RandomForest)
    try:
        print("Initialisation de l'explainer SHAP...")
        # Note : Si ton modèle est dans un Pipeline, il faudra peut-être accéder à model.named_steps['classifier']
        explainer = shap.TreeExplainer(model)
        print("Succès : Explainer SHAP prêt.")
    except Exception as e_shap:
        print(f"Attention : Impossible d'initier TreeExplainer ({e_shap}). L'explicabilité ne sera pas disponible.")
        
except Exception as e:
    print(f"Erreur Critique : Échec du chargement du modèle. Exception : {e}")

class ClientData(BaseModel):
    features: dict

@app.get("/")
def health_check():
    return {
        "status": "API en ligne",
        "model_loaded": model is not None,
        "explainer_ready": explainer is not None
    }

@app.post("/predict")
def predict_credit_score(data: ClientData):
    if not model:
        raise HTTPException(status_code=503, detail="Service indisponible : Modèle non chargé.")
    
    try:
        # 1. Transformation en DataFrame
        df = pd.DataFrame([data.features])
        
        # 2. Nettoyage technique
        cols_techniques = ['SK_ID_CURR', 'TARGET', 'index', 'Unnamed: 0']
        df_clean = df.drop(columns=[c for c in cols_techniques if c in df.columns], errors='ignore')

        # 3. Alignement des colonnes (Sécurité)
        if hasattr(model, "feature_names_in_"):
            expected_cols = model.feature_names_in_
            missing_cols = set(expected_cols) - set(df_clean.columns)
            if missing_cols:
                for c in missing_cols:
                    df_clean[c] = 0
            df_clean = df_clean[expected_cols]

        # 4. Prédiction
        proba_defaut = model.predict_proba(df_clean)[:, 1][0]
        
        # 5. Seuil (Logique Métier)
        seuil_risque = 0.067 # Arrondi pour la lisibilité
        decision_finale = "REFUSÉ" if proba_defaut > seuil_risque else "ACCORDÉ"
        
        # --- P8 ADDITION : Calcul des SHAP Values ---
        shap_data = {}
        base_value = 0
        
        if explainer:
            # Calcul des shap values
            shap_values = explainer.shap_values(df_clean)
            
            # Gestion du format de retour de SHAP (dépend de la version et du modèle)
            # Cas 1: SHAP renvoie une liste [valeurs_classe_0, valeurs_classe_1] -> On prend l'indice 1
            if isinstance(shap_values, list):
                vals = shap_values[1][0] # [1] pour la classe positive, [0] pour le 1er échantillon
            # Cas 2: SHAP renvoie un array directement (rare pour classifier binaire mais possible)
            else:
                vals = shap_values[0]
            
            # On convertit en liste simple pour le JSON
            shap_data = dict(zip(df_clean.columns, vals.tolist()))
            
            # Récupération de l'expected_value (la moyenne globale)
            if isinstance(explainer.expected_value, list) or isinstance(explainer.expected_value, np.ndarray):
                 base_value = float(explainer.expected_value[1])
            else:
                 base_value = float(explainer.expected_value)

        return {
            "score": float(proba_defaut),
            "decision": decision_finale,
            "threshold": seuil_risque,
            # Nouvelles clés pour le dashboard
            "shap_values": shap_data, 
            "base_value": base_value
        }

    except Exception as e:
        import traceback
        traceback.print_exc() # Utile pour débugger dans la console
        raise HTTPException(status_code=400, detail=f"Erreur de traitement : {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)