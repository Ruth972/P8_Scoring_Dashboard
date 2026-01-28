import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ==============================================================================
# CONFIGURATION & CONSTANTES
# ==============================================================================

# URL de l'endpoint de prédiction (API MLflow).
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
    """
    try:
        # Chargement du fichier CSV local
        df = pd.read_csv("donnees_sample.csv")
        return df
    except FileNotFoundError:
        st.error("Erreur Critique : Le fichier source 'donnees_sample.csv' est introuvable.")
        return pd.DataFrame()

df = load_data()

# --- Simulation de données personnelles (car le dataset est anonymisé) ---
def get_client_info(client_id):
    # On utilise l'ID comme "seed" pour que le même ID donne toujours le même nom
    np.random.seed(int(client_id)) 
    noms = ["Martin", "Bernard", "Thomas", "Petit", "Robert", "Richard", "Durand", "Dubois"]
    prenoms = ["Jean", "Marie", "Michel", "Pierre", "Paul", "Jacques", "Sophie", "Julie"]
    villes = ["Paris", "Lyon", "Marseille", "Bordeaux", "Lille", "Toulouse", "Nantes"]
    
    return {
        "Nom": np.random.choice(noms),
        "Prénom": np.random.choice(prenoms),
        "Ville": np.random.choice(villes),
        "Adresse": f"{np.random.randint(1, 150)} rue de la République",
        "Email": f"client.{client_id}@email.com"
    }

# ==============================================================================
# INTERFACE UTILISATEUR : BARRE LATÉRALE (SIDEBAR)
# ==============================================================================

st.sidebar.header("🔍 Sélection du Dossier")

if not df.empty:
    id_list = df['SK_ID_CURR'].unique()
    selected_id = st.sidebar.selectbox("Identifiant Client (ID)", id_list)
    btn_predict = st.sidebar.button("📊 Lancer l'analyse de risque")
else:
    st.sidebar.warning("Données indisponibles.")
    selected_id = None
    btn_predict = False

st.sidebar.markdown("---")

# --- AJOUT : Feature Importance Globale (Réelle) ---
st.sidebar.subheader("🌍 Importance Globale")
st.sidebar.caption("Les variables qui pèsent le plus lourd pour le modèle (calculé sur l'entraînement) :")

@st.cache_data
def load_global_importance():
    try:
        # On charge le fichier CSV qu'on vient de créer
        return pd.read_csv("global_importance.csv")
    except FileNotFoundError:
        return pd.DataFrame()

global_feat_importance = load_global_importance()

if not global_feat_importance.empty:
    fig_global = px.bar(
        global_feat_importance.sort_values(by="Importance", ascending=True),
        x="Importance", 
        y="Feature", 
        orientation='h',
        color_discrete_sequence=['#3498db']
    )
    fig_global.update_layout(
        height=300, 
        margin=dict(l=0, r=0, t=0, b=0), 
        xaxis_title=None, 
        yaxis_title=None
    )
    st.sidebar.plotly_chart(fig_global, use_container_width=True)
else:
    st.sidebar.info("Données d'importance globale non disponibles.")


# ==============================================================================
# CORPS PRINCIPAL
# ==============================================================================

st.title("🏦 Dashboard de Scoring Crédit")

if btn_predict and selected_id:
    
    # --------------------------------------------------------------------------
    # 1. INFORMATIONS CLIENT (Haut de page)
    # --------------------------------------------------------------------------
    client_row = df[df['SK_ID_CURR'] == selected_id]
    
    if not client_row.empty:
        infos = get_client_info(selected_id)
        
        # Affichage style "Carte d'identité"
        with st.container():
            st.markdown("### 👤 Fiche Client")
            col_info1, col_info2, col_info3, col_info4 = st.columns(4)
            col_info1.metric("Nom", f"{infos['Nom']} {infos['Prénom']}")
            col_info2.metric("ID Client", str(selected_id))
            col_info3.metric("Ville", infos['Ville'])
            col_info4.metric("Revenu Annuel", f"{client_row['AMT_INCOME_TOTAL'].values[0]:,.0f} $")
            st.markdown("---")

        # Préparation données API
        cols_excluded = ['TARGET', 'SK_ID_CURR', 'index', 'Unnamed: 0']
        client_data_dict = client_row.drop(columns=cols_excluded, errors='ignore').iloc[0].to_dict()
        clean_features = {k: (0 if pd.isna(v) else v) for k, v in client_data_dict.items()}

        # ----------------------------------------------------------------------
        # 2. APPEL API
        # ----------------------------------------------------------------------
        with st.spinner('Analyse en cours...'):
            try:
                payload = {"dataframe_records": [clean_features]}
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    api_result = response.json()
                    
                    # --- DÉBALLAGE ROBUSTE DU JSON ---
                    # Cas 1: Dictionnaire avec clé 'predictions'
                    if isinstance(api_result, dict) and "predictions" in api_result:
                        preds = api_result["predictions"]
                        # Si predictions est une liste, on prend le premier élément
                        if isinstance(preds, list):
                            data = preds[0]
                        # Si predictions est un dictionnaire (columnar), on prend tel quel
                        else:
                            data = preds
                    # Cas 2: Liste directe
                    elif isinstance(api_result, list):
                        data = api_result[0]
                    # Cas 3: Dictionnaire direct
                    else:
                        data = api_result

                    # --- Extraction variables ---
                    score_raw = data.get('score', [0])
                    score = score_raw[0] if isinstance(score_raw, list) else score_raw
                    
                    decision_raw = data.get('decision', ["Inconnu"])
                    decision = decision_raw[0] if isinstance(decision_raw, list) else decision_raw
                    
                    threshold_raw = data.get('threshold', 0.5)
                    threshold = threshold_raw[0] if isinstance(threshold_raw, list) else threshold_raw
                    
                    shap_values_raw = data.get('shap_values', [])
                    if shap_values_raw:
                        # Gestion liste de listes
                        raw_list = shap_values_raw[0] if isinstance(shap_values_raw[0], list) else shap_values_raw
                        shap_values = dict(zip(clean_features.keys(), raw_list))
                    else:
                        shap_values = {}
                    
                    # ----------------------------------------------------------
                    # 3. VISUALISATION SCORE
                    # ----------------------------------------------------------
                    st.subheader("1️⃣ Synthèse de la décision")
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        color = "green" if decision == "ACCORDÉ" else "red"
                        st.markdown(f"""
                            <div style="text-align: center; border: 2px solid {color}; padding: 15px; border-radius: 10px; background-color: rgba(0,0,0,0.05);">
                                <h2 style="color: {color}; margin:0;">{decision}</h2>
                                <hr style="margin: 10px 0;">
                                <p style="margin:0; font-size: 1.1em;">Probabilité de défaut : <strong>{score:.1%}</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                    with col2:
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=score,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Score de Risque", 'font': {'size': 24}},
                            delta={'reference': threshold, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                            gauge={
                                'axis': {'range': [0, 1], 'tickwidth': 1},
                                'bar': {'color': "black"},
                                'bgcolor': "white",
                                'steps': [
                                    {'range': [0, threshold], 'color': "#2ecc71"},
                                    {'range': [threshold, 1], 'color': "#e74c3c"}
                                ],
                                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': threshold}
                            }
                        ))
                        # Marges ajustées (t=60) pour éviter de couper le titre
                        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20), font={'family': "Arial"})
                        st.plotly_chart(fig_gauge, use_container_width=True)

                    # ----------------------------------------------------------
                    # 4. FEATURE IMPORTANCE LOCALE (SHAP)
                    # ----------------------------------------------------------
                    st.markdown("---")
                    st.subheader("2️⃣ Interprétabilité : Facteurs d'influence (Local)")
                    st.caption(f"Pourquoi le client {selected_id} a eu ce score précis ?")
                    
                    if shap_values:
                        shap_df = pd.DataFrame(list(shap_values.items()), columns=['Feature', 'Impact'])
                        shap_df['Abs_Impact'] = shap_df['Impact'].abs()
                        top_shap = shap_df.sort_values(by='Abs_Impact', ascending=False).head(15)
                        
                        fig_shap = px.bar(
                            top_shap.sort_values(by='Impact', ascending=True),
                            x='Impact', y='Feature', orientation='h', color='Impact',
                            color_continuous_scale=['#2ecc71', '#e74c3c'],
                            title="Top 15 des variables contributrices pour ce dossier"
                        )
                        st.plotly_chart(fig_shap, use_container_width=True)

                    # ----------------------------------------------------------
                    # 5. ANALYSE UNI-VARIÉE (Positionnement)
                    # ----------------------------------------------------------
                    st.markdown("---")
                    st.subheader("3️⃣ Comparaison Uni-variée (1 variable)")
                    
                    compare_var = st.selectbox(
                        "Variable à comparer :", 
                        ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'],
                        index=0
                    )
                    
                    if compare_var in df.columns:
                        client_val = client_row[compare_var].values[0]
                        fig_dist = px.histogram(df, x=compare_var, nbins=50, title=f"Distribution : {compare_var}", color_discrete_sequence=['#95a5a6'], opacity=0.6)
                        fig_dist.add_vline(x=client_val, line_width=3, line_dash="dash", line_color="#e74c3c", annotation_text="Client", annotation_position="top right")
                        st.plotly_chart(fig_dist, use_container_width=True)
                    
                    # ----------------------------------------------------------
                    # 6. ANALYSE BI-VARIÉE (Scatter Plot)
                    # ----------------------------------------------------------
                    st.markdown("---")
                    st.subheader("4️⃣ Comparaison Bi-variée (Croisement)")
                    st.caption("Positionner le client selon deux critères simultanés.")

                    col_bi1, col_bi2 = st.columns(2)
                    with col_bi1:
                        var_x = st.selectbox("Axe X :", ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_BIRTH'], index=0)
                    with col_bi2:
                        var_y = st.selectbox("Axe Y :", ['AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_BIRTH', 'EXT_SOURCE_2'], index=0)

                    if var_x in df.columns and var_y in df.columns:
                        # Création du Scatter Plot
                        fig_bi = px.scatter(
                            df, x=var_x, y=var_y, 
                            color_discrete_sequence=['#bdc3c7'], # Gris clair pour les autres
                            title=f"Croisement : {var_x} vs {var_y}",
                            opacity=0.5
                        )
                        
                        # Ajout du point spécifique au client (En gros et rouge)
                        client_pt = client_row[[var_x, var_y]]
                        fig_bi.add_trace(
                            go.Scatter(
                                x=client_pt[var_x], 
                                y=client_pt[var_y], 
                                mode='markers',
                                marker=dict(color='red', size=15, symbol='star'),
                                name='Client Sélectionné'
                            )
                        )
                        st.plotly_chart(fig_bi, use_container_width=True)

                    # ----------------------------------------------------------
                    # AUDIT
                    # ----------------------------------------------------------
                    with st.expander("🔎 Audit des données"):
                        st.json(clean_features)

                else:
                    st.error(f"Erreur API : {response.status_code}")

            except Exception as e:
                st.error(f"Erreur technique : {e}")