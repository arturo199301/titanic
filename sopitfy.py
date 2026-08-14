import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# 1. Configuración y Encabezado con Imagen
st.set_page_config(page_title="Spotify Classifier", page_icon="🎵", layout="centered")

# Muestra la imagen (ajusta la ruta si es necesario)
st.image("image_0.png", use_container_width=True)

# 2. Cargar datos y entrenar el Modelo (Clasificación)
# Nota: Asumimos que el archivo 'image_0.png' está en la misma carpeta que este script.
@st.cache_data
def load_data():
    df = pd.read_csv("most_streamed_spotify_2025_cleaned_v2.csv")
    # Features y Target
    features = ['billed_artist_count', 'daily_streams', 'daily_stream_share_pct']
    target = 'is_collaboration_int'
    return df[features], df[target]

X, y = load_data()

# Entrenar modelo (Regresión Logística para clasificación simple)
model = LogisticRegression(max_iter=1000).fit(X, y)

# 3. Interfaz básica y Título
st.title("🎯 Clasificador de Colaboraciones")
st.markdown("Ingresa los datos para predecir si es una canción solista o una colaboración.")

# Entradas del usuario
artistas = st.number_input("Cantidad de artistas acreditados:", min_value=1, value=2, step=1)
diarias = st.number_input("Reproducciones diarias estimadas:", value=300000)
cuota = st.number_input("Cuota de mercado diaria (%):", value=0.15)

# 4. Predicción
if st.button("🔮 Clasificar Canción"):
    # Preparar datos de entrada
    input_data = [[artistas, diarias, cuota]]
    pred = model.predict(input_data)[0]
    
    # Mostrar resultado de forma clara
    if pred == 1:
        st.success("🤝 **Predicción:** Es una Colaboración")
    else:
        st.info("👤 **Predicción:** Es una canción de Solista")

# Pie de página opcional
st.markdown("---")
st.caption("Nota: Este es un modelo de ejemplo con fines educativos.")
