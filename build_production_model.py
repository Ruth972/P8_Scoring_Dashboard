import mlflow
import mlflow.pyfunc
import joblib
import shap
import pandas as pd
import numpy as np
import os

# ==============================================================================
# ⚙️ CONFIGURATION DU MODÈLE
# ==============================================================================

# 1. TON SEUIL OPTIMAL (Celui calculé dans ton Notebook P7)
# D'après tes tests précédents, c'était environ 0.067.
# ⚠️ Vérifie cette valeur dans ton notebook de modélisation !
OPTIMAL_THRESHOLD = 0.067 

# 2. Chemin vers ton fichier modèle actuel
CURRENT_MODEL_PATH = "./mlruns/9/models/m-0a84d69a2e314f0e82736c01fbcdd540/artifacts/model.pkl"

# ==============================================================================
# 🧠 DÉFINITION DU WRAPPER (Pipeline + SHAP + Seuil Custom)
# ==============================================================================
class CreditScoringWrapper(mlflow.pyfunc.PythonModel):
    
    def load_context(self, context):
        """
        Chargement intelligent : On sépare le Pipeline en deux morceaux.
        1. Le Preprocessor (pour transformer les données avant SHAP)
        2. Le Classifieur (pour calculer SHAP)
        """
        print("Initialisation du Wrapper de Production...")
        
        # Chargement du Pipeline complet (ImbPipeline)
        self.pipeline = joblib.load(context.artifacts["model_file"])
        
        # --- EXTRACTION POUR SHAP ---
        # SHAP ne digère pas les Pipelines entiers, il veut juste le modèle final.
        if hasattr(self.pipeline, 'steps'):
            # Le modèle est la dernière étape (index -1)
            self.model_classifier = self.pipeline.steps[-1][1]
            # Le préprocesseur est tout ce qu'il y a avant (slicing [:-1])
            self.preprocessor = self.pipeline[:-1]
        else:
            # Cas où ce n'est pas un pipeline mais juste un modèle
            self.model_classifier = self.pipeline
            self.preprocessor = None
        
        print(f"Modèle extrait : {type(self.model_classifier)}")
        print("Initialisation de SHAP TreeExplainer...")
        
        # On initialise SHAP sur le classifieur uniquement
        self.explainer = shap.TreeExplainer(self.model_classifier)

    def predict(self, context, model_input):
        """
        Prédiction avec seuil personnalisé et explication SHAP
        """
        # 1. Calcul du Score (Probabilité)
        # On utilise le pipeline complet, il gère lui-même les transformations
        proba = self.pipeline.predict_proba(model_input)[:, 1]
        
        # 2. Calcul des SHAP Values
        # ATTENTION : Il faut donner à SHAP des données transformées (mises à l'échelle)
        if self.preprocessor:
            try:
                data_for_shap = self.preprocessor.transform(model_input)
            except Exception as e:
                print(f"Erreur transformation SHAP : {e}")
                data_for_shap = model_input # Fallback
        else:
            data_for_shap = model_input
            
        shap_values = self.explainer.shap_values(data_for_shap)
        
        # Gestion du format de retour SHAP (liste vs array)
        if isinstance(shap_values, list):
            vals = shap_values[1]
        else:
            vals = shap_values

        # 3. Décision métier avec TON SEUIL
        # C'est ici qu'on utilise OPTIMAL_THRESHOLD au lieu de 0.5
        decision = ["REFUSÉ" if p > OPTIMAL_THRESHOLD else "ACCORDÉ" for p in proba]

        # 4. Retour formaté pour l'API
        return {
            "score": proba.tolist(),
            "decision": decision,
            "threshold": OPTIMAL_THRESHOLD,
            "shap_values": vals.tolist()
        }

# ==============================================================================
# 📦 CONSTRUCTION ET SAUVEGARDE
# ==============================================================================

if not os.path.exists(CURRENT_MODEL_PATH):
    raise FileNotFoundError(f"Fichier introuvable : {CURRENT_MODEL_PATH}")

artifacts = {
    "model_file": CURRENT_MODEL_PATH
}

print(f"📦 Emballage du modèle avec Seuil={OPTIMAL_THRESHOLD}...")

with mlflow.start_run(run_name="Production_Scoring_Final") as run:
    
    mlflow.pyfunc.log_model(
        artifact_path="scoring_model_final",
        python_model=CreditScoringWrapper(),
        artifacts=artifacts,
        # On force imbalanced-learn et scikit-learn compatible
        pip_requirements=["joblib", "scikit-learn", "shap", "pandas", "numpy", "imbalanced-learn"]
    )

print("\n" + "="*60)
print(f"✅ MODÈLE DE PRODUCTION PRÊT !")
print(f"👉 ID du Run : {run.info.run_id}")
print(f"👉 Chemin : ./mlruns/0/{run.info.run_id}/artifacts/scoring_model_final")
print("="*60)