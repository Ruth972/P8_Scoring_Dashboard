from streamlit.testing.v1 import AppTest

# ==========================================================
# TEST 1 : LE DASHBOARD DÉMARRE-T-IL ?
# ==========================================================
def test_dashboard_loading():
    """
    Simule le lancement de l'application Streamlit.
    """
    at = AppTest.from_file("dashboard.py").run()
    assert not at.exception

# ==========================================================
# TEST 2 : LES ÉLÉMENTS CLÉS SONT-ILS LÀ ? (CORRIGÉ)
# ==========================================================
def test_dashboard_elements():
    """
    Vérifie la présence du titre et de la sélection de l'ID.
    """
    at = AppTest.from_file("dashboard.py").run()
    
    # 1. Vérifier le Titre
    assert len(at.title) > 0
    
    # 2. Vérifier la sélection de l'ID (Adapté pour Selectbox)
    # On compte tout ce qui ressemble à un input (Liste, Texte, Nombre)
    # Dans la page principale
    main_inputs = len(at.selectbox) + len(at.number_input) + len(at.text_input)
    
    # Dans la barre latérale (Sidebar) - Au cas où tu l'aurais mis là
    sidebar_inputs = len(at.sidebar.selectbox) + len(at.sidebar.number_input) + len(at.sidebar.text_input)
    
    total_inputs = main_inputs + sidebar_inputs
    
    # On vérifie qu'il y a bien au moins un élément pour choisir l'ID
    assert total_inputs > 0

# ==========================================================
# TEST 3 : LE BOUTON EST-IL PRÉSENT ?
# ==========================================================
def test_button_exists():
    at = AppTest.from_file("dashboard.py").run()
    
    # On cherche un bouton (soit page principale, soit sidebar)
    total_buttons = len(at.button) + len(at.sidebar.button)
    assert total_buttons > 0