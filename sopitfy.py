import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------
# 1. Configuración de la página y Logo
# ---------------------------------------------------------
st.set_page_config(
    page_title="Predictor de Éxito - Spotify", 
    page_icon="🎵",
    layout="centered"
)

# Mostrar la Imagen (Logo de Spotify)
try:
    spotify_logo = Image.open('image_1.png')
    st.image(spotify_logo, width=250)
except FileNotFoundError:
    st.warning("⚠️ No se encontró la imagen 'image_1.png'. Revisa que esté en la misma carpeta.")
except Exception as e:
    st.error(f"Error al cargar la imagen: {e}")

st.title("🎵 Predictor de Éxito en Spotify")
st.write("Selecciona una canción existente para autocompletar sus datos o ingresa una nueva para predecir si entra al **Top 100 global**.")

# ---------------------------------------------------------
# 2. Cargar datos del CSV y Entrenar el Modelo
# ---------------------------------------------------------
@st.cache_data
def cargar_datos_y_entrenar():
    try:
        df = pd.read_csv('most_streamed_spotify_2025_cleaned_v2.csv')
    except FileNotFoundError:
        st.error("No se encontró el archivo 'most_streamed_spotify_2025_cleaned_v2.csv'.")
        return None, None, None

    # Target de Clasificación: 1 si está en el Top 100 (rank <= 100), 0 si no
    df['is_top_100'] = (df['rank'] <= 100).astype(int)
    
    # Variables de entrenamiento
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
    
    # Entrenar Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return df, model, features

df, modelo, features_names = cargar_datos_y_entrenar()

if df is None:
    st.stop()

# ---------------------------------------------------------
# 3. Selección y Autocompletado por Canción
# ---------------------------------------------------------
st.subheader("🎧 Selección de Canción")

# Crear lista con opción personalizada
opciones_canciones = ["-- Ingresar Canción Nueva --"] + list(df['track'].astype(str) + " - " + df['artist'].astype(str))
cancion_seleccionada = st.selectbox("Selecciona una canción del dataset:", opciones_canciones)

# Valores por defecto
nombre_cancion_def = ""
artista_def = ""
streams_totales_def = 250000000
daily_streams_def = 400000
daily_rank_def = 50
artist_count_def = 1
is_collab_def = "No"
share_pct_def = 0.15
wrapped_rank_def = 0

# Si elige una canción existente, cargamos sus datos exactos
if cancion_seleccionada != "-- Ingresar Canción Nueva --":
    idx = opciones_canciones.index(cancion_seleccionada) - 1
    fila = df.iloc[idx]
    
    nombre_cancion_def = str(fila['track'])
    artista_def = str(fila['artist'])
    streams_totales_def = int(fila['spotify_streams_total'])
    daily_streams_def = int(fila['daily_streams'])
    daily_rank_def = int(fila['daily_streams_rank'])
    artist_count_def = int(fila['billed_artist_count'])
    is_collab_def = "Sí" if fila['is_collaboration_int'] == 1 else "No"
    share_pct_def = float(fila['daily_stream_share_pct'])
    wrapped_rank_def = int(fila['wrapped_global_top10_rank'])

# ---------------------------------------------------------
# 4. Campos de Entrada (Información y Métricas)
# ---------------------------------------------------------
st.subheader("📋 Información y Métricas de la Canción")

col_info1, col_info2 = st.columns(2)
with col_info1:
    track_name = st.text_input("Nombre de la Canción", value=nombre_cancion_def, placeholder="Ej: Tití Me Preguntó")
with col_info2:
    artist_name = st.text_input("Artista(s)", value=artista_def, placeholder="Ej: Bad Bunny")

col1, col2 = st.columns(2)

with col1:
    streams_totales = st.number_input(
        "Reproducciones Totales (spotify_streams_total)", 
        min_value=0, value=streams_totales_def, step=1000000
    )
    
    daily_streams_input = st.number_input(
        "Reproducciones Diarias (daily_streams)", 
        min_value=0, value=daily_streams_def, step=10000
    )
    
    daily_rank_input = st.number_input(
        "Ranking Diario (daily_streams_rank)", 
        min_value=1, max_value=1000, value=daily_rank_def, step=1
    )
    
    billed_artist_count_input = st.number_input(
        "Número de Artistas (billed_artist_count)", 
        min_value=1, max_value=10, value=artist_count_def
    )

with col2:
    is_collab_input = st.selectbox(
        "¿Es Colaboración? (is_collaboration)", 
        options=["No", "Sí"],
        index=0 if is_collab_def == "No" else 1
    )
    
    daily_stream_share_pct_input = st.number_input(
        "Cuota de Reproducción Diaria (%)", 
        min_value=0.0, max_value=100.0, value=share_pct_def, step=0.01
    )
    
    wrapped_rank_input = st.number_input(
        "Posición Spotify Wrapped Top 10 (0 si no califica)", 
        min_value=0, max_value=10, value=wrapped_rank_def, step=1
    )

is_collaboration_int_final = 1 if is_collab_input == "Sí" else 0

# ---------------------------------------------------------
# 5. Ejecución de la Predicción
# ---------------------------------------------------------
st.markdown("---")

if st.button("🚀 Predecir Probabilidad de Éxito", use_container_width=True):
    nueva_cancion_df = pd.DataFrame([[
        streams_totales,
        daily_streams_input, 
        daily_rank_input,
        billed_artist_count_input, 
        is_collaboration_int_final, 
        daily_stream_share_pct_input,
        wrapped_rank_input
    ]], columns=features_names)
    
    prediccion = modelo.predict(nueva_cancion_df)[0]
    probabilidad = modelo.predict_proba(nueva_cancion_df)[0][1]
    
    st.divider()
    
    titulo_cancion = f"**{track_name}**" if track_name else "La canción"
    if artist_name:
        titulo_cancion += f" de **{artist_name}**"
        
    if prediccion == 1:
        st.success(f"🎉 {titulo_cancion} **¡ENTRA AL TOP 100!**\n\nProbabilidad estimada: **{probabilidad * 100:.2f}%**")
        st.balloons()
    else:
        st.warning(f"📉 {titulo_cancion} **QUEDA FUERA DEL TOP 100.**\n\nProbabilidad de éxito estimada: **{probabilidad * 100:.2f}%**")
