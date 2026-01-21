# dashboard.py
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# Ton URL API Render (Vérifie qu'elle est correcte)
API_URL = "https://api-scoring-v246.onrender.com/predict" 

# Configuration de la page
st.set_page_config(page_title="Scoring Crédit Dashboard", layout="wide")

st.title("🏦 Dashboard d'Octroi de Crédit")
st.markdown("Outil d'aide à la décision pour les chargés de clientèle.")

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_data():
    # Assure-toi que ce fichier existe bien sur GitHub
    data = pd.read_csv("donnees_sample.csv")
    return data

with st.spinner("Chargement des données clients..."):
    df = load_data()

# --- BARRE LATÉRALE ---
st.sidebar.header("🔍 Sélection du dossier")
# On récupère la liste des IDs
client_ids = df['SK_ID_CURR'].tolist()
selected_id = st.sidebar.selectbox("ID Client", client_ids)

# --- ANALYSE DU CLIENT ---
if st.sidebar.button("Lancer l'analyse"):
    
    # 1. Récupération des données du client (la ligne complète)
    client_row = df[df['SK_ID_CURR'] == selected_id].iloc[0]
    
    # 2. Préparation des données pour l'API (CORRECTION CRUCIALE ICI)
    # On convertit en dictionnaire
    features_raw = client_row.to_dict()
    
    # On nettoie les données :
    # - On enlève les colonnes inutiles (ID, Target, index...)
    # - On remplace les NaN (valeurs vides) par 0 ou None, sinon le JSON plante !
    cols_a_exclure = ['TARGET', 'SK_ID_CURR', 'index', 'Unnamed: 0']
    
    features = {}
    for k, v in features_raw.items():
        if k not in cols_a_exclure:
            # Si la valeur est vide (NaN), on met 0 pour que l'API accepte
            if pd.isna(v):
                features[k] = 0
            else:
                features[k] = v
    
    # 3. Appel à l'API
    try:
        # On envoie le dictionnaire propre
        response = requests.post(API_URL, json={"features": features})
        
        if response.status_code == 200:
            result = response.json()
            score = result['score']
            decision = result['decision']
            seuil = result['threshold']
            
            # --- AFFICHAGE DES RÉSULTATS ---
            col1, col2 = st.columns(2)
            
            with col1:
                st.header(f"Décision : {decision}")
                if decision == "ACCORDÉ":
                    st.success("✅ Risque Faible (Crédit Accordé)")
                else:
                    st.error("❌ Risque Élevé (Crédit Refusé)")
            
            with col2:
                st.metric("Probabilité de Défaut", f"{score:.1%}")
                # Barre de progression (rouge si élevé, vert si faible)
                st.progress(int(score * 100))
                st.caption(f"Seuil limite : {seuil*100}%")
            
            # Affichage des données brutes (Pour vérifier ce qu'on envoie)
            with st.expander("Voir les détails techniques du dossier"):
                st.write("Données envoyées à l'IA :")
                st.json(features)
                
        else:
            st.error(f"Erreur API ({response.status_code})")
            st.write(response.text)
            
    except requests.exceptions.ConnectionError:
        st.error("🚨 Impossible de contacter l'API.")
        st.warning(f"Vérifiez l'URL : {API_URL}")
    except Exception as e:
        st.error(f"Une erreur inattendue est survenue : {e}")