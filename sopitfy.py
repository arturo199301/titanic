import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Necesario para abrir la imagen
from PIL import Image

# ---------------------------------------------------------
# Configuración inicial de la página
# ---------------------------------------------------------
st.set_page_config(
    page_title="Predicción de Éxito en Spotify",
    # Puedes usar un emoji o la misma imagen como icono (ajustada a tamaño icono)
    page_icon="🎵",
    layout="wide"
)

# ---------------------------------------------------------
# Cargar la Imagen de Spotify
# Asegúrate de que el archivo 'image_0.png' esté en la misma carpeta
# ---------------------------------------------------------
try:
    spotify_logo = Image.open('spotify.png')
except FileNotFoundError:
    spotify_logo = None
    st.error("⚠️ No se encontró la imagen 'spotify.png'. Asegúrate de que esté en la misma carpeta que este script.")

# ---------------------------------------------------------
# Título Principal y Markdown
# ---------------------------------------------------------
st.title("🎵 Clasificador de Canciones: ¿Será un Top 100 en Spotify?")
st.markdown("""
Esta aplicación utiliza un modelo de Machine Learning (**Random Forest Classifier**) 
para predecir si una canción logrará estar en el **Top 100** en función de sus métricas de reproducción diaria y colaboración.
""")

# ... (El resto de la función `load_data` y df = load_data() permanece igual) ...

# ---------------------------------------------------------
# Barra Lateral (Sidebar): Parámetros interactivos e Imagen
# ---------------------------------------------------------
# AGREGADO: Mostrar la imagen en la barra lateral
if spotify_logo:
    # Mostramos la imagen con el ancho de la barra lateral
    st.sidebar.image(spotify_logo, use_column_width=True)

st.sidebar.header("⚙️ Configuración del Modelo")

# Hiperparámetros del modelo
n_estimators = st.sidebar.slider("Número de Árboles (n_estimators)", min_value=10, max_value=200, value=100, step=10)
max_depth = st.sidebar.slider("Profundidad Máxima del Árbol", min_value=1, max_value=20, value=10)
test_size = st.sidebar.slider("Proporción de Datos de Prueba (Test Size)", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

# ... (El resto del código del modelo, pestañas y predicción permanece igual) ...
