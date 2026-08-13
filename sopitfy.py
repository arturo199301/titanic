import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# 1. Configuración de página e Imagen
st.set_page_config(page_title="Spotify Classifier", page_icon="🎵")
st.image("spotify.png", use_container_width=True)

# 2. Cargar datos
@st.cache_data
def load_data():
    return pd.read_csv("most_streamed_spotify_2025_cleaned_v2.csv")

df = load_data()

# 3. Entrenar modelo de Clasificación (¿Es colaboración?)
X = df[['billed_artist_count', 'daily_streams', 'daily_stream_share_pct']]
y = df['is_collaboration_int']

model = LogisticRegression().fit(X, y)

# 4. Interfaz
st.title("Clasificador: Solista vs Colaboración")

artistas = st.number_input("Cantidad de artistas:", min_value=1, value=2)
diarias = st.number_input("Reproducciones diarias:", value=300000)
cuota = st.number_input("Cuota diaria (%):", value=0.15)

# 5. Predicción
if st.button("Clasificar"):
    pred = model.predict([[artistas, diarias, cuota]])[0]
    prob = model.predict_proba([[artistas, diarias, cuota]])[0][1] * 100
    
    if pred == 1:
        st.success(f"🤝 **Es una Colaboración** ({prob:.1f}% de probabilidad)")
    else:
        st.info(f"👤 **Es de un Solista** ({100 - prob:.1f}% de probabilidad)")
