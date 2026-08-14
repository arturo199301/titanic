import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Árbol de Decisión - Spotify Top 100", page_icon="🌳")

# Logo opcional
try:
    st.image(Image.open('image_1.png'), width=180)
except:
    pass

st.title("🌳 Árbol de Decisión Explicable")
st.write("A diferencia del Random Forest, este modelo muestra las **reglas exactas** que sigue para tomar su decisión.")

# 1. Cargar datos y entrenar el Árbol
@st.cache_data
def cargar_y_entrenar():
    df = pd.read_csv('most_streamed_spotify_2025_cleaned_v2.csv')
    df['is_top_100'] = (df['rank'] <= 100).astype(int)
    
    # Features lógicas sin data leakage
    features = [
        'daily_streams', 
        'daily_stream_share_pct', 
        'billed_artist_count', 
        'is_collaboration_int', 
        'wrapped_global_top10_rank'
    ]
    
    X = df[features]
    y = df['is_top_100']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Árbol con profundidad 3 para mantenerlo visual y legible
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    
    return df, model, features, acc

df, modelo_arbol, features, exactitud = cargar_y_entrenar()

st.caption(f"🎯 **Exactitud del árbol en datos de prueba:** {exactitud * 100:.1f}%")

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

# 3. Entradas
col1, col2 = st.columns(2)

with col1:
    daily_streams = st.number_input("Streams Diarios Actuales", value=v_daily, step=50000)
    share_pct = st.number_input("Cuota de Mercado Diaria (%)", value=v_share, step=0.01)
    wrapped = st.number_input("Posición Top 10 Wrapped (0 si no aplica)", value=v_wrap, min_value=0, max_value=10)

with col2:
    art_count = st.number_input("Número de Artistas", value=v_art, min_value=1)
    is_collab = st.selectbox("¿Es Colaboración?", ["No", "Sí"], index=0 if v_collab == "No" else 1)

# 4. Predicción
if st.button("🚀 Evaluar con Árbol de Decisión", use_container_width=True):
    collab_int = 1 if is_collab == "Sí" else 0
    
    datos = pd.DataFrame([[daily_streams, share_pct, art_count, collab_int, wrapped]], columns=features)
    pred = modelo_arbol.predict(datos)[0]
    prob = modelo_arbol.predict_proba(datos)[0][1]
    
    st.divider()
    if pred == 1:
        st.success(f"🎉 **Probable Top 100** (Probabilidad: **{prob * 100:.1f}%**)")
    else:
        st.warning(f"📉 **Fuera del Top 100** (Probabilidad de entrar: **{prob * 100:.1f}%**)")

# 5. Visualización del Árbol de Decisión
st.divider()
st.subheader("📊 Diagrama Visual de las Reglas del Árbol")

fig, ax = plt.subplots(figsize=(12, 6))
plot_tree(
    modelo_arbol, 
    feature_names=features, 
    class_names=['Fuera Top 100', 'Top 100'], 
    filled=True, 
    rounded=True, 
    fontsize=9,
    ax=ax
)
st.pyplot(fig)
