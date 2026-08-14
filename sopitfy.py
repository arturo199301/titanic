import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------
# 1. Configuración de la página
# ---------------------------------------------------------
st.set_page_config(
    page_title="Predictor de Éxito de Spotify", 
    page_icon="🎵",
    layout="centered"
)

# --- Mostrar la Imagen (Logo de Spotify) ---
try:
    # Cargar la imagen proporcionada (guardada como 'image_1.png')
    spotify_logo = Image.open('image_1.png')
    st.image(spotify_logo, width=250)
except FileNotFoundError:
    st.warning("⚠️ No se encontró la imagen 'image_1.png'. Asegúrate de guardarla en la misma carpeta que app.py.")
except Exception as e:
    st.error(f"Error al cargar la imagen: {e}")

# Título y Descripción
st.title("🎵 Predictor Avanzado de Éxito en Spotify")
st.write("Ingresa todas las métricas de la canción para determinar si se posicionará dentro del **Top 100 global**.")

# ---------------------------------------------------------
# 2. Cargar datos del CSV y Entrenar el Modelo
# ---------------------------------------------------------
@st.cache_data
def cargar_y_entrenar():
    try:
        df = pd.read_csv('most_streamed_spotify_2025_cleaned_v2.csv')
    except FileNotFoundError:
        st.error("No se encontró el archivo 'most_streamed_spotify_2025_cleaned_v2.csv'. Verifique la ruta del archivo.")
        return None, None

    # Variable objetivo (Clasificación): 1 si está en el Top 100 (rank <= 100), 0 si no
    df['is_top_100'] = (df['rank'] <= 100).astype(int)
    
    # Se incluyen TODOS los campos numéricos predictivos del dataset
    features = [
        'spotify_streams_total',
        'daily_streams', 
        'daily_streams_rank',
        'billed_artist_count', 
        'is_collaboration_int', 
        'daily_stream_share_pct',
        'wrapped_global_top10_rank'
    ]
    
    X = df[features]
    y = df['is_top_100']
    
    # Entrenamiento del modelo Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, features

modelo, features_names = cargar_y_entrenar()

if modelo is None:
    st.stop()

# ---------------------------------------------------------
# 3. Formulario con MÁS Campos de Entrada
# ---------------------------------------------------------
st.subheader("📋 Parámetros de Entrada de la Canción")

col1, col2 = st.columns(2)

with col1:
    streams_totales = st.number_input(
        "Reproducciones Totales Acumuladas (spotify_streams_total)", 
        min_value=0, value=250000000, step=1000000,
        help="Total histórico de streams en Spotify."
    )
    
    daily_streams_input = st.number_input(
        "Reproducciones Diarias (daily_streams)", 
        min_value=0, value=400000, step=10000,
        help="Promedio estimado de reproducciones diarias."
    )
    
    daily_rank_input = st.number_input(
        "Ranking Diario de Reproducciones (daily_streams_rank)", 
        min_value=1, max_value=1000, value=50, step=1,
        help="Lugar que ocupa la canción en la lista diaria de reproducciones."
    )
    
    billed_artist_count_input = st.number_input(
        "Número de Artistas Acreditados (billed_artist_count)", 
        min_value=1, max_value=10, value=1,
        help="Cantidad de artistas que firman la canción."
    )

with col2:
    is_collab_input = st.selectbox(
        "¿Es una Colaboración? (is_collaboration)", 
        options=["No", "Sí"],
        help="Marca 'Sí' si intervienen múltiples artistas en colaboración."
    )
    
    daily_stream_share_pct_input = st.number_input(
        "Cuota de Reproducción Diaria (%) (daily_stream_share_pct)", 
        min_value=0.0, max_value=100.0, value=0.15, step=0.01,
        help="Porcentaje de la cuota global diaria de reproducciones."
    )
    
    wrapped_rank_input = st.number_input(
        "Posición en Spotify Wrapped Top 10 (0 si no califica)", 
        min_value=0, max_value=10, value=0, step=1,
        help="Lugar ocupado en la lista anual Spotify Wrapped (de 1 a 10). Pon 0 si no figura."
    )

# Convertir la selección de colaboración a formato numérico (0 o 1)
is_collaboration_int_final = 1 if is_collab_input == "Sí" else 0

# ---------------------------------------------------------
# 4. Predicción con el Modelo
# ---------------------------------------------------------
st.markdown("---")

if st.button("🚀 Predecir Probabilidad de Éxito", use_container_width=True):
    # Crear el conjunto de datos de entrada alineado con las variables del modelo
    nueva_cancion_df = pd.DataFrame([[
        streams_totales,
        daily_streams_input, 
        daily_rank_input,
        billed_artist_count_input, 
        is_collaboration_int_final, 
        daily_stream_share_pct_input,
        wrapped_rank_input
    ]], columns=features_names)
    
    # Clasificación y cálculo de probabilidad
    prediccion = modelo.predict(nueva_cancion_df)[0]
    probabilidad = modelo.predict_proba(nueva_cancion_df)[0][1]
    
    st.divider()
    
    # Mostrar resultados en pantalla
    if prediccion == 1:
        st.success(f"🎉 **¡ÉXITO GARANTIZADO! La canción entra al TOP 100.**\n\nProbabilidad estimada: **{probabilidad * 100:.2f}%**")
        st.balloons()
    else:
        st.warning(f"📉 **LA CANCIÓN QUEDA FUERA DEL TOP 100.**\n\nProbabilidad de éxito estimada: **{probabilidad * 100:.2f}%**")
