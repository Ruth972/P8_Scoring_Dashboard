from streamlit.testing.v1 import AppTest

# ==========================================================
# TEST 1 : SMOKE TEST (LE DASHBOARD DÉMARRE-T-IL ?)
# ==========================================================
def test_dashboard_startup():
    """
    Vérifie que l'application démarre sans lever d'exception critique.
    """
    at = AppTest.from_file("dashboard.py").run()
    assert not at.exception

# ==========================================================
# TEST 2 : VÉRIFICATION DU TITRE ET DE LA STRUCTURE
# ==========================================================
def test_title_and_header():
    """
    Vérifie que le titre principal est correct et correspond à la maquette.
    """
    at = AppTest.from_file("dashboard.py").run()
    
    # On cible le premier titre de la page principale
    # Note : at.title est une liste de tous les titres trouvés
    assert len(at.title) > 0
    assert "Dashboard de Scoring Crédit" in at.title[0].value

# ==========================================================
# TEST 3 : VÉRIFICATION DE LA SIDEBAR (BARRE LATÉRALE)
# ==========================================================
def test_sidebar_elements():
    """
    Vérifie que la barre latérale contient bien les contrôles attendus.
    """
    at = AppTest.from_file("dashboard.py").run()
    
    # 1. Vérifier la présence de la Selectbox (Liste déroulante) dans la sidebar
    # Dans ton dashboard.py, c'est 'at.sidebar.selectbox'
    assert len(at.sidebar.selectbox) == 1
    
    # 2. Vérifier le label de la Selectbox (optionnel mais recommandé)
    # On s'attend à trouver "Identifiant Client" ou "ID Client"
    assert "ID" in at.sidebar.selectbox[0].label or "Client" in at.sidebar.selectbox[0].label

    # 3. Vérifier la présence du Bouton d'analyse dans la sidebar
    assert len(at.sidebar.button) == 1
    assert "analyse" in at.sidebar.button[0].label

# ==========================================================
# TEST 4 : CHARGEMENT DES DONNÉES (TEST D'INTÉGRATION LOCAL)
# ==========================================================
def test_data_loading():
    """
    Vérifie indirectement que le fichier CSV est bien chargé.
    Si le CSV est chargé, la selectbox ne doit pas être vide.
    """
    at = AppTest.from_file("dashboard.py").run()
    
    # Si le fichier CSV est lu correctement, la selectbox doit avoir des options
    # Note : Cela nécessite que 'donnees_sample.csv' soit présent lors du test
    try:
        options = at.sidebar.selectbox[0].options
        assert len(options) > 0
    except IndexError:
        # Si la selectbox n'existe pas, le test échoue
        assert False, "La selectbox des clients n'a pas été trouvée."