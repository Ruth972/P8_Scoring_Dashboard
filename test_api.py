from fastapi.testclient import TestClient
from main import app

# Initialisation du client de test
client = TestClient(app)

def test_health_check():
    """
    Teste si l'endpoint racine '/' répond correctement
    et confirme le chargement du moteur (Modèle + Explainer).
    """
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "API en ligne"
    # Vérification que l'API suit l'état du modèle
    assert "model_loaded" in data
    assert "explainer_ready" in data

def test_predict_complete_response():
    """
    Teste une prédiction complète (P8).
    Vérifie le Score, la Décision, mais aussi les SHAP Values (Transparence).
    """
    # Données fictives minimales
    fake_client_data = {
        "features": {
            "SK_ID_CURR": 100002,
            "AMT_INCOME_TOTAL": 200000.0,
            "AMT_CREDIT": 100000.0,
            "EXT_SOURCE_2": 0.5, # Souvent une feature importante
            "EXT_SOURCE_3": 0.5
        }
    }

    response = client.post("/predict", json=fake_client_data)
    
    # 1. Vérification technique
    assert response.status_code == 200
    
    data = response.json()
    
    # 2. Vérification des éléments de base (P7)
    assert "score" in data
    assert "decision" in data
    assert "threshold" in data
    assert isinstance(data["score"], float)
    
    # 3. Vérification des éléments de transparence (P8 - NOUVEAU)
    assert "shap_values" in data
    assert "base_value" in data
    
    # On vérifie que shap_values est bien un dictionnaire (Feature -> Impact)
    assert isinstance(data["shap_values"], dict)

def test_predict_error_handling():
    """
    Teste si l'API gère bien les données invalides.
    """
    bad_payload = ["Ceci n'est pas un dictionnaire"]
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422