import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ==============================================================================
# CONFIGURATION & CONSTANTES
# ==============================================================================

# URL de l'endpoint de prédiction (API MLflow).
# Note : Avec MLflow Serving, l'endpoint standard est "/invocations"
API_URL = "https://p8-scoring-dashboard.onrender.com/invocations"

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Dashboard Scoring Crédit",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# GESTION DES DONNÉES (DATA LOADING)
# ==============================================================================

@st.cache_data
def load_data():
    """
    Charge le jeu de données client (échantillon) pour la navigation.
    Utilise le cache Streamlit pour optimiser les performances.
    Returns:
        pd.DataFrame: DataFrame contenant les données clients.
    """
    try:
        # Chargement du fichier CSV local
        df = pd.read_csv("donnees_sample.csv")
        return df
    except FileNotFoundError:
        st.error("Erreur Critique : Le fichier source 'donnees_sample.csv' est introuvable.")
        return pd.DataFrame()

# Initialisation du DataFrame
df = load_data()

# ==============================================================================
# INTERFACE UTILISATEUR : BARRE LATÉRALE (SIDEBAR)
# ==============================================================================

st.sidebar.header("🔍 Sélection du Dossier Client")

if not df.empty:
    # Création de la liste de sélection par ID Client (SK_ID_CURR)
    id_list = df['SK_ID_CURR'].unique()
    selected_id = st.sidebar.selectbox("Identifiant Client (ID)", id_list)
    
    # Bouton de déclenchement de l'analyse
    btn_predict = st.sidebar.button("📊 Lancer l'analyse de risque")
else:
    st.sidebar.warning("Base de données clients indisponible.")
    selected_id = None
    btn_predict = False

st.sidebar.markdown("---")
st.sidebar.info(
    "**Note :** Ce dashboard est une interface d'aide à la décision. "
    "Les scores sont générés par un modèle de Machine Learning via API."
)

# ==============================================================================
# CORPS PRINCIPAL DE L'APPLICATION
# ==============================================================================

st.title("🏦 Dashboard de Scoring Crédit")
st.markdown("Analyse du risque de crédit et explicabilité de la décision.")

if btn_predict and selected_id:
    
    # --------------------------------------------------------------------------
    # 1. PRÉPARATION DES DONNÉES (DATA PREPROCESSING)
    # --------------------------------------------------------------------------
    # Extraction de la ligne correspondant au client sélectionné
    client_row = df[df['SK_ID_CURR'] == selected_id]
    
    if not client_row.empty:
        # Exclusion des colonnes techniques non requises par le modèle
        cols_excluded = ['TARGET', 'SK_ID_CURR', 'index', 'Unnamed: 0']
        
        # Conversion en dictionnaire pour sérialisation JSON
        client_data_dict = client_row.drop(columns=cols_excluded, errors='ignore').iloc[0].to_dict()
        
        # Gestion des valeurs manquantes (NaN) :
        # JSON ne supporte pas NaN. On remplace par 0 ou None selon la logique métier.
        clean_features = {k: (0 if pd.isna(v) else v) for k, v in client_data_dict.items()}

        # ----------------------------------------------------------------------
        # 2. APPEL API (INFERENCE REQUEST)
        # ----------------------------------------------------------------------
        with st.spinner('Interrogation du modèle MLflow...'):
            try:
                # MLflow attend un format "dataframe_records"
                payload = {
                    "dataframe_records": [clean_features]
                }
                
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    api_result = response.json()
                    
                    # --- ✅ DÉBALLAGE ROBUSTE DU JSON ---
                    
                    if isinstance(api_result, dict) and "predictions" in api_result:
                        preds = api_result["predictions"]
                        # CAS A : predictions est une LISTE (Format standard)
                        if isinstance(preds, list):
                            data = preds[0]
                        # CAS B : predictions est un DICTIONNAIRE (Format 'columnar')
                        else:
                            data = preds 
                    elif isinstance(api_result, list):
                        data = api_result[0]
                    else:
                        data = api_result

                    # --- Extraction des Valeurs ---
                    
                    # 1. SCORE
                    score_raw = data.get('score', [0])
                    score = score_raw[0] if isinstance(score_raw, list) else score_raw
                    
                    # 2. DÉCISION
                    decision_raw = data.get('decision', ["Inconnu"])
                    decision = decision_raw[0] if isinstance(decision_raw, list) else decision_raw
                    
                    # 3. SEUIL
                    threshold_raw = data.get('threshold', 0.5)
                    threshold = threshold_raw[0] if isinstance(threshold_raw, list) else threshold_raw
                    
                    # 4. SHAP VALUES
                    shap_values_raw = data.get('shap_values', [])
                    
                    if shap_values_raw:
                        # Si c'est une liste de listes, on prend la première sous-liste
                        if isinstance(shap_values_raw[0], list):
                             raw_list = shap_values_raw[0]
                        else:
                             raw_list = shap_values_raw
                        
                        # Mapping : Nom de colonne -> Impact SHAP
                        shap_values = dict(zip(clean_features.keys(), raw_list))
                    else:
                        shap_values = {}
                    
                    # ----------------------------------------------------------
                    # 3. VISUALISATION : SCORE & DÉCISION
                    # ----------------------------------------------------------
                    st.subheader("1️⃣ Synthèse de la décision")
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # Indicateur visuel (Badge de décision)
                        color = "green" if decision == "ACCORDÉ" else "red"
                        st.markdown(f"""
                            <div style="text-align: center; border: 2px solid {color}; padding: 15px; border-radius: 10px; background-color: rgba(0,0,0,0.05);">
                                <h2 style="color: {color}; margin:0;">{decision}</h2>
                                <hr style="margin: 10px 0;">
                                <p style="margin:0; font-size: 1.1em;">Probabilité de défaut : <strong>{score:.1%}</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                    with col2:
                        # Graphique Jauge (Gauge Chart) via Plotly
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=score,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Score de Risque"},
                            # Delta par rapport au seuil critique
                            delta={'reference': threshold, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                            gauge={
                                'axis': {'range': [0, 1]},
                                'bar': {'color': "black"}, # Indicateur actuel
                                'steps': [
                                    {'range': [0, threshold], 'color': "#2ecc71"}, # Zone Verte (Sûre)
                                    {'range': [threshold, 1], 'color': "#e74c3c"}  # Zone Rouge (Risque)
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': threshold
                                }
                            }
                        ))
                        # Ajustement des marges pour un affichage compact
                        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
                        st.plotly_chart(fig_gauge, use_container_width=True)

                    # ----------------------------------------------------------
                    # 4. EXPLICABILITÉ DU MODÈLE (SHAP VALUES)
                    # ----------------------------------------------------------
                    st.markdown("---")
                    st.subheader("2️⃣ Interprétabilité : Facteurs d'influence")
                    st.caption("Analyse des variables ayant le plus impacté le score (Feature Importance Locale).")
                    
                    if shap_values:
                        # Conversion dict -> DataFrame pour manipulation
                        shap_df = pd.DataFrame(list(shap_values.items()), columns=['Feature', 'Impact'])
                        
                        # Tri par impact absolu pour identifier les drivers principaux
                        shap_df['Abs_Impact'] = shap_df['Impact'].abs()
                        top_shap = shap_df.sort_values(by='Abs_Impact', ascending=False).head(15)
                        
                        # Graphique Bar Plot Interactif
                        fig_shap = px.bar(
                            top_shap.sort_values(by='Impact', ascending=True), # Tri pour l'ordre d'affichage visuel
                            x='Impact', 
                            y='Feature', 
                            orientation='h',
                            color='Impact',
                            # Échelle de couleur divergente : Vert (Baisse risque) <-> Rouge (Hausse risque)
                            color_continuous_scale=['#2ecc71', '#e74c3c'], 
                            title="Top 15 des variables contributrices"
                        )
                        st.plotly_chart(fig_shap, use_container_width=True)
                    else:
                        st.warning("⚠️ Les données d'explicabilité (SHAP) ne sont pas disponibles pour ce dossier.")

                    # ----------------------------------------------------------
                    # 5. ANALYSE COMPARATIVE (BI-VARIÉE)
                    # ----------------------------------------------------------
                    st.markdown("---")
                    st.subheader("3️⃣ Positionnement du client")
                    
                    # Sélecteur de variable pour l'analyse comparative
                    compare_var = st.selectbox(
                        "Sélectionner une variable pour comparer le client à la population :", 
                        ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'],
                        index=0
                    )
                    
                    if compare_var in df.columns:
                        client_val = client_row[compare_var].values[0]
                        
                        # Histogramme de distribution globale
                        fig_dist = px.histogram(
                            df, 
                            x=compare_var, 
                            nbins=50, 
                            title=f"Distribution de la variable : {compare_var}",
                            color_discrete_sequence=['#95a5a6'], # Gris neutre pour le fond
                            opacity=0.6,
                            labels={compare_var: "Valeur", "count": "Nombre de clients"}
                        )
                        
                        # Ajout d'une ligne verticale marquant la position du client actuel
                        fig_dist.add_vline(
                            x=client_val, 
                            line_width=3, 
                            line_dash="dash", 
                            line_color="#e74c3c", # Rouge pour visibilité
                            annotation_text="Client Sélectionné", 
                            annotation_position="top right"
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)
                        
                        # Affichage de la valeur exacte
                        st.metric(f"Valeur du client ({compare_var})", f"{client_val:,.2f}")
                    
                    # ----------------------------------------------------------
                    # 6. AUDIT & DONNÉES BRUTES
                    # ----------------------------------------------------------
                    with st.expander("🔎 Audit : Voir les données brutes transmises"):
                        st.write("Données JSON envoyées à l'API :")
                        st.json(clean_features)

                else:
                    # Gestion des erreurs HTTP (404, 500, etc.)
                    st.error(f"Échec de la communication API (Code: {response.status_code})")
                    st.code(response.text)

            except requests.exceptions.ConnectionError:
                # Gestion des erreurs de connexion (API éteinte ou URL invalide)
                st.error("🚨 Connexion impossible au serveur de prédiction.")
                st.warning(f"Vérifiez que l'URL de l'API est correcte et que le service est actif : {API_URL}")
            except Exception as e:
                # Gestion générique des exceptions
                st.error(f"Une erreur technique est survenue : {e}")

    else:
        st.error("Identifiant client introuvable dans la base de données locale.")