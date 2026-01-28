import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import math  # Indispensable pour l'aiguille

# ==============================================================================
# CONFIGURATION & CONSTANTES
# ==============================================================================

API_URL = "https://p8-scoring-dashboard.onrender.com/invocations"

st.set_page_config(
    page_title="Dashboard Scoring Crédit",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# GESTION DE LA SESSION
# ==============================================================================
if 'api_data' not in st.session_state:
    st.session_state.api_data = None
if 'current_client_id' not in st.session_state:
    st.session_state.current_client_id = None

# ==============================================================================
# GESTION DES DONNÉES
# ==============================================================================

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("donnees_sample.csv")
        return df
    except FileNotFoundError:
        st.error("Erreur : 'donnees_sample.csv' introuvable.")
        return pd.DataFrame()

df = load_data()

@st.cache_data
def load_global_importance():
    try:
        return pd.read_csv("global_importance.csv")
    except FileNotFoundError:
        return pd.DataFrame()

def get_client_info(client_id):
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
# SIDEBAR
# ==============================================================================

st.sidebar.header("🔍 Sélection du Dossier")

if not df.empty:
    id_list = df['SK_ID_CURR'].unique()
    selected_id = st.sidebar.selectbox("Identifiant Client (ID)", id_list)
    
    if st.sidebar.button("📊 Lancer l'analyse de risque"):
        st.session_state.current_client_id = selected_id
        
        client_row = df[df['SK_ID_CURR'] == selected_id]
        if not client_row.empty:
            cols_excluded = ['TARGET', 'SK_ID_CURR', 'index', 'Unnamed: 0']
            client_data_dict = client_row.drop(columns=cols_excluded, errors='ignore').iloc[0].to_dict()
            clean_features = {k: (0 if pd.isna(v) else v) for k, v in client_data_dict.items()}
            
            with st.spinner('Analyse en cours...'):
                try:
                    payload = {"dataframe_records": [clean_features]}
                    response = requests.post(API_URL, json=payload)
                    if response.status_code == 200:
                        st.session_state.api_data = response.json()
                        st.session_state.api_data['clean_features'] = clean_features 
                    else:
                        st.error(f"Erreur API : {response.status_code}")
                        st.session_state.api_data = None
                except Exception as e:
                    st.error(f"Erreur technique : {e}")
                    st.session_state.api_data = None
else:
    st.sidebar.warning("Données indisponibles.")

st.sidebar.markdown("---")
st.sidebar.info(
    "**Note :** Ce dashboard est une interface d'aide à la décision. "
    "Les scores sont générés par un modèle de Machine Learning via API."
)
st.sidebar.markdown("---")

st.sidebar.subheader("🌍 Importance Globale")
global_feat_importance = load_global_importance()
if not global_feat_importance.empty:
    fig_global = px.bar(
        global_feat_importance.sort_values(by="Importance", ascending=True),
        x="Importance", y="Feature", orientation='h',
        color_discrete_sequence=['#3498db']
    )
    fig_global.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0), xaxis_title=None, yaxis_title=None)
    st.sidebar.plotly_chart(fig_global, use_container_width=True)

# ==============================================================================
# CORPS PRINCIPAL
# ==============================================================================

st.title("🏦 Dashboard de Scoring Crédit")

if st.session_state.api_data and st.session_state.current_client_id == selected_id:
    
    api_result = st.session_state.api_data
    clean_features = api_result.get('clean_features', {})
    
    # INFOS CLIENT
    client_row = df[df['SK_ID_CURR'] == selected_id]
    infos = get_client_info(selected_id)
    
    with st.container():
        st.markdown("### 👤 Fiche Client")
        col_info1, col_info2, col_info3, col_info4 = st.columns(4)
        col_info1.metric("Nom", f"{infos['Nom']} {infos['Prénom']}")
        col_info2.metric("ID Client", str(selected_id))
        col_info3.metric("Ville", infos['Ville'])
        col_info4.metric("Revenu Annuel", f"{client_row['AMT_INCOME_TOTAL'].values[0]:,.0f} $")
        st.markdown("---")

    # DÉBALLAGE JSON
    if isinstance(api_result, dict) and "predictions" in api_result:
        preds = api_result["predictions"]
        data = preds[0] if isinstance(preds, list) else preds
    elif isinstance(api_result, list):
        data = api_result[0]
    else:
        data = api_result

    score_raw = data.get('score', [0])
    score = score_raw[0] if isinstance(score_raw, list) else score_raw
    decision_raw = data.get('decision', ["Inconnu"])
    decision = decision_raw[0] if isinstance(decision_raw, list) else decision_raw
    threshold_raw = data.get('threshold', 0.5)
    threshold = threshold_raw[0] if isinstance(threshold_raw, list) else threshold_raw
    shap_values_raw = data.get('shap_values', [])
    if shap_values_raw:
        raw_list = shap_values_raw[0] if isinstance(shap_values_raw[0], list) else shap_values_raw
        shap_values = dict(zip(clean_features.keys(), raw_list))
    else:
        shap_values = {}
    
    # --- JAUGE AIGUILLE DROITE ---
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
        # --- CONFIGURATION DE LA JAUGE ---
        gauge_max = threshold * 2 
        val_visuel = min(score, gauge_max)
        
        # --- CALCUL DE L'AIGUILLE (Trigonométrie corrigée) ---
        # 0 (Gauche) = 180 degrés | Max (Droite) = 0 degrés
        angle_deg = 180 - (val_visuel / gauge_max) * 180
        angle_rad = math.radians(angle_deg)
        
        # Longueur de l'aiguille (0.5 = rayon complet, on met 0.4 pour qu'elle reste dedans)
        needle_length = 0.40
        
        # Coordonnées de la pointe (x1, y1)
        # Centre du graphique = 0.5, 0
        x1 = 0.5 + needle_length * math.cos(angle_rad)
        y1 = 0 + needle_length * math.sin(angle_rad) # y0 est à 0

        # --- CRÉATION DU GRAPHIQUE ---
        fig_gauge = go.Figure()

        # 1. Le fond coloré (Arc de cercle)
        fig_gauge.add_trace(go.Indicator(
            mode = "gauge+number",
            value = val_visuel,
            number = {'suffix': "", 'valueformat': ".1%", 'font': {'size': 40, 'weight': 'bold'}},
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Probabilité de Défaut", 'font': {'size': 20, 'color': "gray"}},
            gauge = {
                'axis': {'range': [0, gauge_max], 'visible': False}, # On cache les ticks moches
                'bar': {'color': "rgba(0,0,0,0)"}, # On cache la barre de progression par défaut
                'bgcolor': "white",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, threshold], 'color': "#2ecc71"},   # Vert (Accordé)
                    {'range': [threshold, gauge_max], 'color': "#e74c3c"}  # Rouge (Refusé)
                ]
            }
        ))

        # 2. LA LIGNE DE SEUIL (Noire verticale au milieu)
        fig_gauge.add_shape(
            type="line",
            x0=0.5, y0=0, x1=0.5, y1=0.45, # Ligne verticale à midi
            line=dict(color="black", width=2, dash="dot"),
            xref="paper", yref="paper"
        )
        # Annotation "Seuil"
        fig_gauge.add_annotation(
            x=0.5, y=0.55, text="SEUIL", showarrow=False,
            font=dict(size=12, color="black"), xref="paper", yref="paper"
        )

        # 3. L'AIGUILLE (Shape Line) - CORRECTION DU BUG D'AFFICHAGE
        fig_gauge.add_shape(
            type="line",
            x0=0.5, y0=0,       # Départ : Centre Bas exact
            x1=x1, y1=y1,       # Arrivée : Pointe calculée
            line=dict(color="#2c3e50", width=5), # Aiguille gris foncé élégante
            xref="paper", yref="paper"
        )
        
        # 4. LE PIVOT (Cercle au centre)
        fig_gauge.add_shape(
            type="circle",
            x0=0.48, y0=-0.02, x1=0.52, y1=0.02,
            fillcolor="#2c3e50", line_color="#2c3e50",
            xref="paper", yref="paper"
        )

        fig_gauge.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            height=300,
            font={'family': "Arial"}
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Légende sous le graphique pour rassurer l'utilisateur
        st.caption(f"Le seuil de risque est fixé à **{threshold:.1%}**. Si l'aiguille est dans la zone verte, le crédit est accordé.")

    # FEATURE IMPORTANCE
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
            title="Top 15 des variables contributrices"
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    # UNI-VARIÉE
    st.markdown("---")
    st.subheader("3️⃣ Comparaison Uni-variée")
    st.caption(f"Où se situe le client {selected_id} par rapport à l'ensemble de la population ?")
    
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
    
    # BI-VARIÉE
    st.markdown("---")
    st.subheader("4️⃣ Comparaison Bi-variée (Croisement)")
    st.caption(f"Le profil du client {selected_id} est-il atypique selon ces deux critères combinés ?")
    
    col_bi1, col_bi2 = st.columns(2)
    with col_bi1:
        options_x = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_BIRTH']
        var_x = st.selectbox("Axe X :", options_x, index=1)
    with col_bi2:
        options_y = ['AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_BIRTH', 'EXT_SOURCE_2']
        var_y = st.selectbox("Axe Y :", options_y, index=2)

    if var_x in df.columns and var_y in df.columns:
        plot_df = df.copy()
        
        if var_x == 'DAYS_BIRTH':
            plot_df['AGE_YEARS'] = (plot_df['DAYS_BIRTH'] / -365).astype(int)
            plot_var_x = 'AGE_YEARS'
        else:
            plot_var_x = var_x
            
        if var_y == 'DAYS_BIRTH':
            plot_df['AGE_YEARS'] = (plot_df['DAYS_BIRTH'] / -365).astype(int)
            plot_var_y = 'AGE_YEARS'
        else:
            plot_var_y = var_y
        
        client_val_x = client_row[var_x].values[0]
        client_val_y = client_row[var_y].values[0]
        if var_x == 'DAYS_BIRTH': client_val_x = int(client_val_x / -365)
        if var_y == 'DAYS_BIRTH': client_val_y = int(client_val_y / -365)

        fig_bi = go.Figure()
        fig_bi.add_trace(go.Scatter(
            x=plot_df[plot_var_x], y=plot_df[plot_var_y],
            mode='markers',
            marker=dict(color='#bdc3c7', size=5, opacity=0.3),
            name='Population'
        ))
        fig_bi.add_trace(go.Scatter(
            x=[client_val_x], y=[client_val_y], 
            mode='markers',
            marker=dict(color='red', size=15, symbol='star', opacity=1.0),
            name='Client Sélectionné'
        ))
        fig_bi.update_layout(title=f"Croisement : {plot_var_x} vs {plot_var_y}", title_font_size=20, xaxis_title=plot_var_x, yaxis_title=plot_var_y)
        st.plotly_chart(fig_bi, use_container_width=True)

    with st.expander("🔎 Audit des données"):
        st.json(clean_features)

elif selected_id:
    st.info("👈 Veuillez cliquer sur 'Lancer l'analyse' dans la barre latérale pour démarrer.")