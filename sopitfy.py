import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

# 1. Título y Logo
st.set_page_config(page_title="Predicción Spotify", page_icon="🎵")

try:
    st.image(Image.open('image_1.png'), width=180)
except:
    pass

st.title("🎵 Predictor Top 100 de Spotify")

# 2. Cargar datos y entrenar el modelo Random Forest
@st.cache_data
def entrenar():
    df = pd.read_csv('most_streamed_spotify_2025_cleaned_v2.csv')
    df['is_top_100'] = (df['rank'] <= 100).astype(int)
    
    features = [
        'spotify_streams_total', 'daily_streams', 
        'billed_artist_count', 'is_collaboration_int', 
        'daily_stream_share_pct', 'wrapped_global_top10_rank'
    ]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(df[features], df['is_top_100'])
    return df, model, features

df, modelo, features = entrenar()

# 3. Autocompletado opcional de canciones
cancion = st.selectbox("Elegir canción existente (o dejar en blanco):", ["-- Otra --"] + list(df['track']))

if cancion != "-- Otra --":
    fila = df[df['track'] == cancion].iloc[0]
    v_total, v_daily = int(fila['spotify_streams_total']), int(fila['daily_streams'])
    v_art, v_collab = int(fila['billed_artist_count']), "Sí" if fila['is_collaboration_int'] == 1 else "No"
    v_share, v_wrap = float(fila['daily_stream_share_pct']), int(fila['wrapped_global_top10_rank'])
else:
    v_total, v_daily, v_art, v_collab, v_share, v_wrap = 200000000, 300000, 1, "No", 0.12, 0

# 4. Formulario de entradas
st.subheader("Ingresa o ajusta los datos:")
col1, col2 = st.columns(2)

with col1:
    tot_streams = st.number_input("Streams Totales", value=v_total, step=1000000)
    daily_streams = st.number_input("Streams Diarios", value=v_daily, step=10000)
    art_count = st.number_input("Cantidad de Artistas", value=v_art, min_value=1)

with col2:
    is_collab = st.selectbox("¿Es Colaboración?", ["No", "Sí"], index=0 if v_collab == "No" else 1)
    share_pct = st.number_input("Cuota Diaria (%)", value=v_share, step=0.01)
    wrapped = st.number_input("Spotify Wrapped Top 10 (0 si no aplica)", value=v_wrap)

# 5. Predicción
if st.button("🚀 Predecir Éxito", use_container_width=True):
    collab_int = 1 if is_collab == "Sí" else 0
    
    datos = pd.DataFrame([[tot_streams, daily_streams, art_count, collab_int, share_pct, wrapped]], columns=features)
    pred = modelo.predict(datos)[0]
    prob = modelo.predict_proba(datos)[0][1]
    
    st.divider()
    if pred == 1:
        st.success(f"🎉 **¡ENTRA AL TOP 100!** (Probabilidad: {prob * 100:.1f}%)")
        st.balloons()
    else:
        st.warning(f"📉 **QUEDA FUERA DEL TOP 100** (Probabilidad: {prob * 100:.1f}%)")
