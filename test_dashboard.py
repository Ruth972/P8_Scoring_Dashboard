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
    Vérifie que le titre principal est correct.
    """
    at = AppTest.from_file("dashboard.py").run()
    
    # On cible le premier titre de la page principale
    assert len(at.title) > 0
    assert "Dashboard" in at.title[0].value

# ==========================================================
# TEST 3 : VÉRIFICATION DE LA SIDEBAR (LOGIQUE DYNAMIQUE)
# ==========================================================
def test_sidebar_elements():
    """
    Vérifie la logique d'apparition des éléments dans la sidebar.
    Le bouton 'Calculer' ne doit apparaître qu'après une sélection.
    """
    at = AppTest.from_file("dashboard.py").run()
    
    # 1. Vérifier la présence de la Selectbox (Liste déroulante)
    assert len(at.sidebar.selectbox) == 1
    
    # 2. Au démarrage, aucune sélection n'est faite : LE BOUTON DOIT ÊTRE ABSENT
    assert len(at.sidebar.button) == 0

    # 3. ACTION : On simule la sélection de "🆕 Nouveau Dossier (Vierge)"
    # .set_value(...) change la valeur et .run() relance le script comme un utilisateur
    at.sidebar.selectbox[0].set_value("🆕 Nouveau Dossier (Vierge)").run()

    # 4. VÉRIFICATION : Maintenant, le bouton doit être présent
    assert len(at.sidebar.button) == 1
    assert "Calculer" in at.sidebar.button[0].label

# ==========================================================
# TEST 4 : CHARGEMENT DES DONNÉES
# ==========================================================
def test_data_loading():
    """
    Vérifie que le fichier CSV est bien chargé (la liste n'est pas vide).
    """
    at = AppTest.from_file("dashboard.py").run()
    
    try:
        # On vérifie qu'il y a plus d'1 option (l'option par défaut + les IDs)
        options = at.sidebar.selectbox[0].options
        assert len(options) > 1 
    except IndexError:
        assert False, "La selectbox des clients est introuvable."