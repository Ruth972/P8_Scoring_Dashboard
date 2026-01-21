import mlflow
import mlflow.pyfunc
import joblib
import shap
import pandas as pd
import os

# ==============================================================================
# 1. DÉFINITION DU WRAPPER (Le "Cerveau" qui gère Score + SHAP)
# ==============================================================================
class CreditScoringWrapper(mlflow.pyfunc.PythonModel):
    
    def load_context(self, context):
        """
        Charge le modèle en mémoire au démarrage du serveur.
        'context.artifacts' permet de récupérer le chemin du fichier empaqueté.
        """
        print("Initialisation du modèle wrapper...")
        # On charge le modèle physique
        self.model = joblib.load(context.artifacts["model_file"])
        
        # On initialise l'explainer SHAP (TreeExplainer est optimisé pour XGBoost/LGBM/RF)
        print("Initialisation de SHAP TreeExplainer...")
        self.explainer = shap.TreeExplainer(self.model)

    def predict(self, context, model_input):
        """
        Fonction appelée à chaque requête API.
        Renvoie un dictionnaire complet : Score, Décision, SHAP.
        """
        # 1. Calcul du Score (Probabilité de la classe 1)
        proba = self.model.predict_proba(model_input)[:, 1]
        
        # 2. Calcul des valeurs SHAP
        shap_values = self.explainer.shap_values(model_input)
        
        # Gestion du format SHAP (selon la version, renvoie liste ou array)
        if isinstance(shap_values, list):
            vals = shap_values[1]  # Pour la classification binaire
        else:
            vals = shap_values

        # 3. Logique métier (Seuil)
        threshold = 0.5
        decision = ["REFUSÉ" if p > threshold else "ACCORDÉ" for p in proba]

        # 4. Retour formaté
        return {
            "score": proba.tolist(),
            "decision": decision,
            "threshold": threshold,
            "shap_values": vals.tolist()
        }

# ==============================================================================
# 2. CRÉATION DU MODÈLE DE PRODUCTION
# ==============================================================================

# Ton chemin spécifique vers le modèle actuel
CURRENT_MODEL_PATH = "./mlruns/9/models/m-0a84d69a2e314f0e82736c01fbcdd540/artifacts/model.pkl"

# Vérification de sécurité
if not os.path.exists(CURRENT_MODEL_PATH):
    raise FileNotFoundError(f"❌ Le fichier modèle est introuvable ici : {CURRENT_MODEL_PATH}")

# Dictionnaire des artefacts à empaqueter
artifacts = {
    "model_file": CURRENT_MODEL_PATH
}

print(f"📦 Emballage du modèle depuis : {CURRENT_MODEL_PATH}")

# Lancement de la construction MLflow
with mlflow.start_run(run_name="Production_Scoring_SHAP") as run:
    
    mlflow.pyfunc.log_model(
        artifact_path="scoring_model_final",         # Nom du dossier de sortie
        python_model=CreditScoringWrapper(),         # Notre classe wrapper
        artifacts=artifacts,                         # Le fichier .pkl
        pip_requirements=["joblib", "scikit-learn", "shap", "pandas", "numpy"] # Dépendances pour Render
    )

print("\n" + "="*50)
print(f"✅ SUCCÈS ! Le modèle est prêt.")
print(f"👉 ID du Run : {run.info.run_id}")
print(f"👉 Chemin du nouveau modèle : ./mlruns/0/{run.info.run_id}/artifacts/scoring_model_final")
print("="*50)