import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Configuración de página e Imagen
st.set_page_config(page_title="Spotify Popularity Classifier", page_icon="🎵")
st.image("spotify.png", use_container_width=True)

# 2. Cargar y preparar datos
@st.cache_data
def load_data():
    df = pd.read_csv("most_streamed_spotify_2025_cleaned_v2.csv")
    
    # Crear la variable objetivo: 1 si está sobre la mediana de reproducciones, 0 si no
    mediana_reproducciones = df['spotify_streams_total'].median()
    df['alta_popularidad'] = (df['spotify_streams_total'] > mediana_reproducciones).astype(int)
    
    return df

df = load_data()

# 3. Entrenar el Modelo de Clasificación (Random Forest)
features = ['daily_streams', 'daily_streams_rank', 'daily_stream_share_pct', 'billed_artist_count', 'is_collaboration_int']
X = df[features]
y = df['alta_popularidad']

model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)

# 4. Interfaz de Usuario
st.title("🎯 Clasificador: ¿Tendrá Alta Popularidad?")
st.write("Predice si una canción superará los **167 millones de reproducciones totales**.")

col1, col2 = st.columns(2)

with col1:
    diarias = st.number_input("Reproducciones diarias:", value=250000, step=10000)
    rank_diario = st.number_input("Ranking diario promedio:", min_value=1, value=100)
    cuota = st.number_input("Cuota de mercado diaria (%):", value=0.12, step=0.01)

with col2:
    artistas = st.number_input("Cantidad de artistas:", min_value=1, value=1)
    colaboracion = st.checkbox("¿Es una colaboración?")

# 5. Predicción
if st.button("🔮 Evaluar Popularidad"):
    input_data = [[diarias, rank_diario, cuota, artistas, int(colaboracion)]]
    
    prediccion = model.predict(input_data)[0]
    probabilidad = model.predict_proba(input_data)[0][1] * 100
    
    st.markdown("---")
    if prediccion == 1:
        st.success(f"🔥 **¡Alta Popularidad!** La canción tiene un **{probabilidad:.1f}%** de probabilidad de superar las 167M de reproducciones.")
    else:
        st.warning(f"📉 **Popularidad Normal / Baja.** Solo tiene un **{probabilidad:.1f}%** de probabilidad de superar las 167M de reproducciones.")
