import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Modelo Sencillo (5 Variables) - Spotify Top 100", page_icon="🎵")

# Logo opcional
try:
    st.image(Image.open('spotify.png'), width=500)
except:
    pass

st.title("Regresión Logística para saber si la cancion esta en el top 100 en Spotify")
st.write("Modelo de clasificación s ranking general.")

# 1. Cargar datos y entrenar el modelo
@st.cache_data
def entrenar_modelo_simple():
    df = pd.read_csv('most_streamed_spotify_2025_cleaned_v2.csv')
    df['is_top_100'] = (df['rank'] <= 100).astype(int)
    
    # 5 variables
    features = [
        'daily_streams', 
        'daily_stream_share_pct', 
        'billed_artist_count', 
        'is_collaboration_int',
        'wrapped_global_top10_rank'
    ]
    
    model = LogisticRegression(max_iter=1000)
    model.fit(df[features], df['is_top_100'])
    
    return df, model, features

df, modelo, features = entrenar_modelo_simple()

# 2. Entradas del usuario (5 campos)
st.subheader("📋 Ingresa las 5 variables de entrada:")

col1, col2 = st.columns(2)

with col1:
    daily_streams = st.number_input("1. Streams Diarios", value=1500000, step=50000)
    share_pct = st.number_input("2. Cuota Diaria (%)", value=0.10, step=0.01)
    wrapped = st.number_input("5. Posición Top 10 Wrapped (0 si no aplica)", value=0, min_value=0, max_value=10)

with col2:
    art_count = st.number_input("3. Número de Artistas", value=1, min_value=1, max_value=10)
    is_collab = st.selectbox("4. ¿Es Colaboración?", ["No", "Sí"], index=0)

collab_int = 1 if is_collab == "Sí" else 0

# 3. Predicción
if st.button("🚀 Evaluar Canción", use_container_width=True):
    datos = pd.DataFrame([[daily_streams, share_pct, art_count, collab_int, wrapped]], columns=features)
    
    prediccion = modelo.predict(datos)[0]
    probabilidad = modelo.predict_proba(datos)[0][1]
    
    st.divider()
    if prediccion == 1:
        st.success(f"🎉 **Es Top 100** (Probabilidad: **{probabilidad * 100:.1f}%**)")
    else:
        st.warning(f"📉 **Fuera del Top 100** (Probabilidad: **{probabilidad * 100:.1f}%**)")
