from fastapi.testclient import TestClient
from main import app

# On crée un "faux" client qui va interroger l'API sans lancer le serveur
client = TestClient(app)

def test_health_check():
    """
    Teste si l'endpoint racine '/' répond correctement.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "API en ligne"

def test_predict_accorded():
    """
    Teste une prédiction avec des données fictives.
    On vérifie que l'API renvoie bien les clés attendues.
    """
    # Données fictives (ce n'est pas grave si les valeurs sont fausses, 
    # tant que le format dictionnaire est respecté)
    fake_client_data = {
        "features": {
            "SK_ID_CURR": 100002,
            "AMT_INCOME_TOTAL": 200000.0,
            "AMT_CREDIT": 100000.0,
            # Ajoute d'autres colonnes si nécessaire, 
            # mais l'API est censée gérer les manquants grâce à ton nettoyage
        }
    }

    response = client.post("/predict", json=fake_client_data)
    
    # 1. Vérification technique (Code 200 = OK)
    assert response.status_code == 200
    
    # 2. Vérification métier (Présence des infos)
    data = response.json()
    assert "score" in data
    assert "decision" in data
    assert "threshold" in data
    
    # 3. Vérification du seuil (Ton fameux 0.059)
    # On vérifie que l'API utilise bien le seuil optimisé
    assert data["threshold"] == 0.06699999999999995

def test_predict_error_handling():
    """
    Teste si l'API gère bien les mauvaises données (ex: format invalide)
    """
    # On envoie une liste au lieu d'un dictionnaire -> Doit planter proprement (422)
    bad_payload = ["Ceci n'est pas un dictionnaire"]
    
    response = client.post("/predict", json=bad_payload)
    
    # 422 est le code standard FastAPI pour "Unprocessable Entity" (Erreur de validation)
    assert response.status_code == 422