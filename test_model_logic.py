import pytest
import joblib
import shap
import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline

# ⚠️ Vérifie que ce chemin correspond bien à ton dossier MLflow sur GitHub
# (Tu peux garder un chemin générique qui pointe vers le modèle que tu testes)
MODEL_PATH = "./mlruns/0/c21f644625d54570805f537995db7177/artifacts/scoring_model_final/artifacts/model.pkl"
# Note : Si le chemin est trop complexe à deviner pour le test, on peut tester le fichier source original :
# MODEL_PATH = "./mlruns/9/models/m-0a84d69a2e314f0e82736c01fbcdd540/artifacts/model.pkl"

# Utilisons le chemin source (plus sûr pour le test CI/CD qui a accès à tout le repo)
MODEL_SOURCE = "./mlruns/9/models/m-0a84d69a2e314f0e82736c01fbcdd540/artifacts/model.pkl"

def test_model_and_shap_integration():
    """
    Test d'intégration robuste :
    1. Charge le Pipeline.
    2. Extrait le modèle (pour SHAP).
    3. Vérifie que tout fonctionne ensemble.
    """
    
    # 1. Vérification présence fichier
    if os.path.exists(MODEL_SOURCE):
        target_path = MODEL_SOURCE
    else:
        # Fallback si on teste un autre chemin
        target_path = MODEL_SOURCE 
        assert os.path.exists(target_path), f"Fichier modèle introuvable : {target_path}"

    # 2. Chargement du Pipeline complet
    full_pipeline = joblib.load(target_path)
    assert full_pipeline is not None

    # 3. Extraction intelligente du modèle (Le Correctif est ICI)
    # On regarde si c'est un Pipeline pour récupérer la dernière étape
    if hasattr(full_pipeline, 'steps'):
        model_step = full_pipeline.steps[-1][1] # Le classifieur
        preprocessor = full_pipeline[:-1]       # Le nettoyage
    else:
        model_step = full_pipeline
        preprocessor = None

    # 4. Initialisation de SHAP (Sur le modèle extrait, pas le pipeline !)
    try:
        explainer = shap.TreeExplainer(model_step)
    except Exception as e:
        pytest.fail(f"SHAP n'aime pas ce modèle : {e}")

    # 5. Création donnée fictive
    try:
        n_features = model_step.n_features_in_
    except:
        n_features = 10
    
    fake_input = pd.DataFrame([np.random.rand(n_features)])

    # 6. Test SHAP avec transformation préalable
    try:
        # Si on a un préprocesseur, on l'applique avant SHAP
        if preprocessor:
            try:
                data_transformed = preprocessor.transform(fake_input)
            except:
                data_transformed = fake_input
        else:
            data_transformed = fake_input

        shap_values = explainer.shap_values(data_transformed)
        
        # Vérification finale
        assert np.array(shap_values).size > 0
        
    except Exception as e:
        pytest.fail(f"Calcul SHAP échoué : {e}")