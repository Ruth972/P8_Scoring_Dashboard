import pytest
import joblib
import shap
import pandas as pd
import numpy as np
import os

# Adapte ce chemin vers ton fichier modèle (celui utilisé dans build_production_model.py)
MODEL_PATH = "./mlruns/9/models/m-0a84d69a2e314f0e82736c01fbcdd540/artifacts/model.pkl"

def test_model_and_shap_integration():
    """
    Ce test simule ce que fait le serveur MLflow :
    1. Il charge le modèle.
    2. Il charge SHAP.
    3. Il fait une prédiction.
    4. Il vérifie que tout renvoie le bon format.
    """
    
    # 1. Vérifier que le fichier existe
    assert os.path.exists(MODEL_PATH), "Le fichier modèle n'est pas trouvé !"

    # 2. Chargement du modèle
    model = joblib.load(MODEL_PATH)
    assert model is not None

    # 3. Initialisation de SHAP (Point critique : ça ne doit pas planter)
    try:
        explainer = shap.TreeExplainer(model)
    except Exception as e:
        pytest.fail(f"L'initialisation de SHAP a échoué : {e}")

    # 4. Création d'une donnée fictive (Fake Client)
    # On crée un DataFrame avec quelques colonnes standards (les noms importent peu pour ce test technique)
    # L'important est d'avoir le bon format (DataFrame)
    # Note : XGBoost/LGBM acceptent souvent des inputs même si les colonnes ne matchent pas à 100% lors d'un test technique,
    # mais idéalement, il faudrait les colonnes exactes. Pour un smoke test, ça suffit souvent.
    
    # On essaie de récupérer le nombre de features attendu par le modèle
    try:
        n_features = model.n_features_in_
    except:
        n_features = 10 # Valeur par défaut si non trouvée
        
    fake_input = pd.DataFrame([np.random.rand(n_features)])

    # 5. Test de la Prédiction (Score)
    try:
        proba = model.predict_proba(fake_input)[:, 1]
        assert isinstance(proba[0], float)
        assert 0 <= proba[0] <= 1
    except Exception as e:
        pytest.fail(f"La prédiction du modèle a échoué : {e}")

    # 6. Test de SHAP (Transparence)
    try:
        shap_values = explainer.shap_values(fake_input)
        # Vérification que shap_values n'est pas vide
        assert np.array(shap_values).size > 0
    except Exception as e:
        pytest.fail(f"Le calcul des valeurs SHAP a échoué : {e}")