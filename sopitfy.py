import streamlit as st
import pandas as pd
from sklearn.naive_bayes import GaussianNB

# 1. Configuración de página e Imagen
st.set_page_config(page_title="Spotify Top 100 Classifier", page_icon="🎵")
st.image("spotify.png", use_container_width=True)

# 2. Cargar datos y preparar la variable objetivo
@st.cache_data
def load_data():
    df = pd.read_csv("most_streamed_spotify_2025_cleaned_v2.csv")
    # Es 1 si está dentro del Top 100 del ranking diario, 0 si no
    df['is_top100'] = (df['daily_streams_rank'] <= 100).astype(int)
    return df

df = load_data()

# 3. Entrenar el Modelo (Naive Bayes)
features = ['spotify_streams_total', 'daily_streams', 'daily_stream_share_pct', 'billed_artist_count']
X = df[features]
y = df['is_top100']

model = GaussianNB().fit(X, y)

# 4. Interfaz de Usuario
st.title("🎯 Clasificador: Predictor Top 100 (Naive Bayes)")
st.write("Evalúa las probabilidades de que una canción ingrese al **Top 100 Diario**.")

col1, col2 = st.columns(2)

with col1:
    totales = st.number_input("Reproducciones Totales:", value=200000000, step=10000000)
    diarias = st.number_input("Reproducciones Diarias:", value=400000, step=20000)

with col2:
    cuota = st.number_input("Cuota Diaria (%):", value=0.18, step=0.01)
    artistas = st.number_input("Cantidad de Artistas:", min_value=1, value=1)

# 5. Predicción
if st.button("🔮 Evaluar Top 100"):
    input_data = [[totales, diarias, cuota, artistas]]
    
    prediccion = model.predict(input_data)[0]
    probabilidad = model.predict_proba(input_data)[0][1] * 100
    
    st.markdown("---")
    if prediccion == 1:
        st.success(f"🏆 **¡Dentro del Top 100!** Probabilidad calculada: **{probabilidad:.1f}%**")
    else:
        st.warning(f"📉 **Fuera del Top 100.** Probabilidad de entrar: **{probabilidad:.1f}%**")
