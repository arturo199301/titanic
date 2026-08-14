import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Spotify Top 100 - Lógica Real", page_icon="🎵")

# Logo
try:
    st.image(Image.open('image_1.png'), width=180)
except:
    pass

st.title("🎵 Predictor Top 100 (Modelo Real)")
st.write("Predicción basada en métricas de rendimiento diario y características del artista, sin trampas de datos.")

# 1. Cargar datos y preparar modelo
@st.cache_data
def cargar_y_entrenar():
    df = pd.read_csv('most_streamed_spotify_2025_cleaned_v2.csv')
    df['is_top_100'] = (df['rank'] <= 100).astype(int)
    
    # ⚠️ Seleccionamos SOLO variables lógicas que NO contienen el resultado directo
    features = [
        'daily_streams', 
        'daily_stream_share_pct', 
        'billed_artist_count', 
        'is_collaboration_int', 
        'wrapped_global_top10_rank'
    ]
    
    X = df[features]
    y = df['is_top_100']
    
    # Separar en entrenamiento y prueba para medir lógica real
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    
    return df, model, features, acc

df, modelo, features, exactitud = cargar_y_entrenar()

st.caption(f"🎯 **Exactitud real del modelo en prueba:** {exactitud * 100:.1f}%")

# 2. Selección de canción
cancion = st.selectbox("Selecciona una canción para autocompletar:", ["-- Crear Nueva --"] + list(df['track']))

if cancion != "-- Crear Nueva --":
    f = df[df['track'] == cancion].iloc[0]
    v_daily = int(f['daily_streams'])
    v_share = float(f['daily_stream_share_pct'])
    v_art = int(f['billed_artist_count'])
    v_collab = "Sí" if f['is_collaboration_int'] == 1 else "No"
    v_wrap = int(f['wrapped_global_top10_rank'])
else:
    v_daily, v_share, v_art, v_collab, v_wrap = 1500000, 0.10, 1, "No", 0

# 3. Entradas del usuario
col1, col2 = st.columns(2)

with col1:
    daily_streams = st.number_input("Streams Diarios Actuales", value=v_daily, step=50000)
    share_pct = st.number_input("Cuota de Mercado Diaria (%)", value=v_share, step=0.01)
    wrapped = st.number_input("Posición Top 10 Wrapped (0 si no aplica)", value=v_wrap, min_value=0, max_value=10)

with col2:
    art_count = st.number_input("Número de Artistas", value=v_art, min_value=1)
    is_collab = st.selectbox("¿Es Colaboración?", ["No", "Sí"], index=0 if v_collab == "No" else 1)

# 4. Predicción
if st.button("🚀 Evaluar con Random Forest", use_container_width=True):
    collab_int = 1 if is_collab == "Sí" else 0
    
    datos = pd.DataFrame([[daily_streams, share_pct, art_count, collab_int, wrapped]], columns=features)
    pred = modelo.predict(datos)[0]
    prob = modelo.predict_proba(datos)[0][1]
    
    st.divider()
    if pred == 1:
        st.success(f"🎉 **Probable Top 100** (Probabilidad: **{prob * 100:.1f}%**)")
        st.balloons()
    else:
        st.warning(f"📉 **Fuera del Top 100** (Probabilidad de entrar: **{prob * 100:.1f}%**)")
