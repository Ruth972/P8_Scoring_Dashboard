import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import math

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

def get_client_info(client_id, modified_income=None):
    # Simulation d'infos civiles (non utilisées par le modèle)
    if client_id == "Nouveau Dossier":
        return {
            "Nom": "Nouveau", "Prénom": "Client", 
            "Ville": "Inconnue", "Email": "nouveau@client.com"
        }
    
    np.random.seed(int(client_id)) 
    noms = ["Martin", "Bernard", "Thomas", "Petit", "Robert", "Richard", "Durand", "Dubois"]
    prenoms = ["Jean", "Marie", "Michel", "Pierre", "Paul", "Jacques", "Sophie", "Julie"]
    villes = ["Paris", "Lyon", "Marseille", "Bordeaux", "Lille", "Toulouse", "Nantes"]
    return {
        "Nom": np.random.choice(noms),
        "Prénom": np.random.choice(prenoms),
        "Ville": np.random.choice(villes),
        "Email": f"client.{client_id}@email.com"
    }

# ==============================================================================
# SIDEBAR (SÉLECTION & SIMULATION)
# ==============================================================================

st.sidebar.header("🔍 Dossier & Simulation")

if not df.empty:
    # Liste des ID + Option "Nouveau Dossier"
    id_list = df['SK_ID_CURR'].unique().tolist()
    id_options = ["Sélectionner un ID..."] + id_list + ["🆕 Nouveau Dossier (Vierge)"]
    
    selected_option = st.sidebar.selectbox("Identifiant Client", id_options)
    
    # Dictionnaire pour stocker les valeurs modifiées
    input_data = {}
    
    if selected_option != "Sélectionner un ID...":
        
        # 1. RÉCUPÉRATION DES DONNÉES DE BASE
        if selected_option == "🆕 Nouveau Dossier (Vierge)":
            # On prend la moyenne des données comme base pour un nouveau dossier
            base_data = df.mean(numeric_only=True).to_dict()
            st.sidebar.warning("Mode Création : Les valeurs par défaut sont les moyennes du marché.")
            current_id = "Nouveau Dossier"
        else:
            # On prend les données réelles du client
            current_id = selected_option
            base_data = df[df['SK_ID_CURR'] == current_id].iloc[0].to_dict()

        # 2. FORMULAIRE DE MODIFICATION (Les Top Features)
        st.sidebar.markdown("---")
        st.sidebar.subheader("✏️ Modifier les informations")
        with st.sidebar.expander("Paramètres du dossier", expanded=True):
            # On définit les variables clés qu'on veut laisser modifier
            # (Nom technique, Label lisible, Step)
            key_features = [
                ('AMT_INCOME_TOTAL', 'Revenus Annuels ($)', 1000.0),
                ('AMT_CREDIT', 'Montant du Crédit ($)', 5000.0),
                ('AMT_ANNUITY', 'Annuités ($)', 500.0),
                ('DAYS_BIRTH', 'Âge (Jours négatifs)', 100.0),
                ('EXT_SOURCE_1', 'Score Extérieur 1 (0-1)', 0.01),
                ('EXT_SOURCE_2', 'Score Extérieur 2 (0-1)', 0.01),
                ('EXT_SOURCE_3', 'Score Extérieur 3 (0-1)', 0.01),
                ('DAYS_EMPLOYED', 'Ancienneté Emploi (Jours)', 100.0),
                ('AMT_GOODS_PRICE', 'Prix du bien ($)', 5000.0)
            ]
            
            # Génération des inputs
            for col, label, step_val in key_features:
                if col in base_data:
                    val = base_data[col]
                    # Gestion des NaN
                    if pd.isna(val): val = 0.0
                    
                    # Input interactif
                    input_data[col] = st.number_input(
                        label, 
                        value=float(val), 
                        step=step_val,
                        format="%.2f"
                    )
        
        # 3. BOUTON D'ACTION
        if st.sidebar.button("🚀 Calculer le Score (Rafraîchir)"):
            st.session_state.current_client_id = current_id
            
            # Fusion : On prend toutes les données de base, et on écrase avec les inputs modifiés
            # Cela permet d'envoyer un vecteur complet à l'API (avec les 100+ autres colonnes inchangées)
            final_features = base_data.copy()
            final_features.update(input_data)
            
            # Nettoyage technique (NaN -> 0, exclusion colonnes inutiles)
            cols_excluded = ['TARGET', 'SK_ID_CURR', 'index', 'Unnamed: 0']
            clean_features = {k: (0 if pd.isna(v) else v) for k, v in final_features.items() if k not in cols_excluded}
            
            with st.spinner('Analyse du profil modifié en cours...'):
                try:
                    payload = {"dataframe_records": [clean_features]}
                    response = requests.post(API_URL, json=payload)
                    if response.status_code == 200:
                        st.session_state.api_data = response.json()
                        st.session_state.api_data['clean_features'] = clean_features 
                        st.session_state.is_simulation = (input_data != {k: base_data[k] for k in input_data if k in base_data})
                    else:
                        st.error(f"Erreur API : {response.status_code}")
                except Exception as e:
                    st.error(f"Erreur technique : {e}")

else:
    st.sidebar.warning("Données indisponibles.")

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
    st.sidebar.caption("📊 **Lecture :** Variables ayant le plus de poids dans le modèle global.")

# ==============================================================================
# CORPS PRINCIPAL
# ==============================================================================

st.title("🏦 Dashboard de Scoring Crédit")

if st.session_state.api_data:
    
    api_result = st.session_state.api_data
    clean_features = api_result.get('clean_features', {})
    current_id = st.session_state.current_client_id
    
    # INFOS CLIENT (Dynamiques selon les inputs)
    infos = get_client_info(current_id)
    
    with st.container():
        # Petit badge si c'est une simulation
        if getattr(st.session_state, 'is_simulation', False) or current_id == "Nouveau Dossier":
            st.warning("⚠️ **Mode Simulation actif :** Les résultats affichés sont basés sur les données modifiées dans la barre latérale.")
        
        st.markdown("### 👤 Fiche Client (Données utilisées)")
        col_info1, col_info2, col_info3, col_info4 = st.columns(4)
        col_info1.metric("Nom", f"{infos['Nom']} {infos['Prénom']}")
        col_info2.metric("ID Dossier", str(current_id))
        col_info3.metric("Revenu Annuel (Simulé)", f"{clean_features.get('AMT_INCOME_TOTAL', 0):,.0f} $")
        col_info4.metric("Montant Crédit (Simulé)", f"{clean_features.get('AMT_CREDIT', 0):,.0f} $")
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
    
    # --- JAUGE GÉOMÉTRIQUE ROBUSTE ---
    st.subheader("1️⃣ Synthèse de la décision")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        color = "#2ecc71" if decision == "ACCORDÉ" else "#e74c3c"
        st.markdown(f"""
            <div style="
                text-align: center; 
                padding: 20px; 
                border: 2px solid {color}; 
                border-radius: 10px; 
                margin-top: 40px; 
                background-color: rgba(255,255,255,0.05);">
                <h2 style="color: {color}; margin-bottom: 10px;">{decision}</h2>
                <hr style="margin: 10px 0; border-top: 1px solid {color}; opacity: 0.3;">
                <p style="margin: 0; font-size: 1.1em;">
                    Probabilité de défaut : <strong style="font-size: 1.2em;">{score:.1%}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
    with col2:
        gauge_max = threshold * 2 
        visual_score = max(0, min(score, gauge_max))
        
        angle_deg = 180 - (visual_score / gauge_max) * 180
        angle_rad = math.radians(angle_deg)
        
        fig = go.Figure()

        def draw_arc(start_angle, end_angle, color, name):
            theta = np.linspace(math.radians(start_angle), math.radians(end_angle), 50)
            r_in, r_out = 0.6, 1.0
            x_out = r_out * np.cos(theta)
            y_out = r_out * np.sin(theta)
            x_in = r_in * np.cos(theta[::-1])
            y_in = r_in * np.sin(theta[::-1])
            return go.Scatter(x=np.concatenate([x_out, x_in, [x_out[0]]]), y=np.concatenate([y_out, y_in, [y_out[0]]]), fill='toself', mode='none', fillcolor=color, name=name, hoverinfo='skip')

        fig.add_trace(draw_arc(90, 180, "#2ecc71", "Zone Verte"))
        fig.add_trace(draw_arc(0, 90, "#e74c3c", "Zone Rouge"))

        needle_len = 0.90
        x_needle = [0, needle_len * math.cos(angle_rad)]
        y_needle = [0, needle_len * math.sin(angle_rad)]

        fig.add_trace(go.Scatter(x=x_needle, y=y_needle, mode='lines', line=dict(color='#2c3e50', width=5), hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker=dict(color='#2c3e50', size=15), hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=[0], y=[0.25], mode='text', text=[f"{visual_score:.1%}"], textfont=dict(size=40, color="white", family="Arial Black"), hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=[0], y=[1.15], mode='text', text=["Score de Risque"], textfont=dict(size=18, color="gray"), hoverinfo='skip'))

        fig.update_layout(
            xaxis=dict(range=[-1.2, 1.2], visible=False, scaleanchor='y', scaleratio=1),
            yaxis=dict(range=[0, 1.3], visible=False),
            margin=dict(l=20, r=20, t=20, b=20),
            height=300,
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Le seuil de risque est fixé à **{threshold:.1%}**. Si l'aiguille est dans la zone verte, le crédit est accordé.")

    # --- FEATURE IMPORTANCE (SHAP) ---
    st.markdown("---")
    st.subheader("2️⃣ Interprétabilité : Facteurs d'influence (Local)")
    st.caption(f"Pourquoi ce score précis a été attribué (basé sur les données simulées) ?")
    
    if shap_values:
        shap_df = pd.DataFrame(list(shap_values.items()), columns=['Feature', 'Impact'])
        shap_df['Abs_Impact'] = shap_df['Impact'].abs()
        top_shap = shap_df.sort_values(by='Abs_Impact', ascending=False).head(15)
        
        fig_shap = px.bar(
            top_shap.sort_values(by='Impact', ascending=True),
            x='Impact', y='Feature', orientation='h', color='Impact',
            color_continuous_scale=['#2ecc71', '#e74c3c'],
        )
        
        fig_shap.update_layout(
            title="<b>Top 15 des variables contributrices</b>",
            title_font_size=18,
            xaxis_title="Contribution au risque (Gauche = Baisse, Droite = Hausse)",
            yaxis_title=None,
            showlegend=False,
            coloraxis_showscale=False,
            height=500,
            font={'family': "Arial"}
        )
        
        fig_shap.add_vline(x=0, line_width=1, line_color="white", opacity=0.5)
        st.plotly_chart(fig_shap, use_container_width=True)
        st.info("💡 **Lecture :** Les barres **ROUGES** augmentent le risque. Les barres **VERTES** diminuent le risque.")

    # --- COMPARAISON (Uni & Bi Variée) ---
    # Note : Pour la comparaison, on compare le client SIMULÉ (valeur dans clean_features) 
    # par rapport à la population statique (df).
    
    st.markdown("---")
    st.subheader("3️⃣ Comparaison : Profil vs Population")
    st.caption("Positionnement du dossier simulé par rapport à l'ensemble des clients.")
    
    col_comp1, col_comp2 = st.columns(2)
    
    with col_comp1:
        # UNI-VARIÉE
        compare_var = st.selectbox(
            "Distribution (Uni-variée)", 
            ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'],
            index=0
        )
        if compare_var in df.columns:
            # Valeur du client simulé
            client_val = clean_features.get(compare_var, 0)
            
            fig_dist = px.histogram(df, x=compare_var, nbins=50, title=f"Distribution : {compare_var}", color_discrete_sequence=['#95a5a6'], opacity=0.6)
            fig_dist.add_vline(x=client_val, line_width=3, line_dash="dash", line_color="#e74c3c", annotation_text="Dossier Simulé")
            fig_dist.update_layout(showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_dist, use_container_width=True)

    with col_comp2:
        # BI-VARIÉE
        var_x = st.selectbox("Axe X (Bi-variée)", ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_BIRTH'], index=1)
        var_y = st.selectbox("Axe Y (Bi-variée)", ['AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_BIRTH', 'EXT_SOURCE_2'], index=2)

        if var_x in df.columns and var_y in df.columns:
            plot_df = df.copy()
            # Gestion des âges
            if var_x == 'DAYS_BIRTH': 
                plot_df['AGE_YEARS'] = (plot_df['DAYS_BIRTH'] / -365).astype(int)
                plot_var_x = 'AGE_YEARS'
                client_val_x = int(clean_features.get(var_x, 0) / -365)
            else:
                plot_var_x = var_x
                client_val_x = clean_features.get(var_x, 0)
                
            if var_y == 'DAYS_BIRTH':
                plot_df['AGE_YEARS'] = (plot_df['DAYS_BIRTH'] / -365).astype(int)
                plot_var_y = 'AGE_YEARS'
                client_val_y = int(clean_features.get(var_y, 0) / -365)
            else:
                plot_var_y = var_y
                client_val_y = clean_features.get(var_y, 0)

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
                name='Dossier Simulé'
            ))
            fig_bi.update_layout(
                title=f"{plot_var_x} vs {plot_var_y}", 
                xaxis_title=plot_var_x, yaxis_title=plot_var_y,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig_bi, use_container_width=True)

elif selected_option == "Sélectionner un ID...":
    st.info("👈 Veuillez sélectionner un dossier ou créer une simulation dans la barre latérale.")