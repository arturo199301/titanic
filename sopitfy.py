import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Configuración de página e Imagen
st.set_page_config(page_title="Spotify Artist Impact Classifier", page_icon="🎵", layout="wide")
st.image("spotify.png", use_container_width=True)

# 2. Cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv("most_streamed_spotify_2025_cleaned_v2.csv")
    # Target Binario: 1 si logró entrar al Top 10 Global Wrapped, 0 si no
    df['is_wrapped_top10'] = (df['wrapped_global_top10_rank'] > 0).astype(int)
    return df

df = load_data()

# 3. Definir EXACTAMENTE 5 Variables de Entrada enfocadas en el perfil del Artista/Tema
features = [
    'billed_artist_count',       # 1. Cantidad de artistas principales
    'is_collaboration_int',      # 2. ¿Es un junte/colaboración?
    'spotify_streams_total',     # 3. Tráfico total acumulado
    'daily_streams',             # 4. Tráfico diario que mueve el artista/tema
    'daily_stream_share_pct'     # 5. Dominio sobre la cuota del mercado diario
]

X = df[features]
y = df['is_wrapped_top10']

# 4. Entrenar el Árbol de Decisión
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))

# 5. Interfaz de Usuario
st.title("🎤 Clasificador por Perfil de Artista: Presence en Wrapped Top 10")
st.write("Evalúa si el formato del artista y su volumen de reproducción lo posicionan en el **Top 10 Global Wrapped**.")

# Formulario organizado en 2 columnas
col1, col2 = st.columns(2)

with col1:
    artistas = st.number_input("1. Cantidad de Artistas (billed_artist_count):", min_value=1, max_value=5, value=2)
    es_colab = st.checkbox("2. ¿Es Colaboración entre Artistas? (is_collaboration_int)", value=True)
    cuota = st.number_input("3. Cuota Diaria de Mercado % (daily_stream_share_pct):", value=0.30, step=0.01)

with col2:
    totales = st.number_input("4. Total Reproducciones del Tema (spotify_streams_total):", value=350000000, step=10000000)
    diarias = st.number_input("5. Reproducciones Diarias (daily_streams):", value=750000, step=25000)

# 6. Predicción
if st.button("🔮 Clasificar Presencia en Wrapped"):
    input_data = [[
        artistas, 
        int(es_colab), 
        totales, 
        diarias, 
        cuota
    ]]
    
    prediccion = model.predict(input_data)[0]
    probabilidad = model.predict_proba(input_data)[0][1] * 100
    
    st.markdown("---")
    if prediccion == 1:
        st.success(f"🌟 **¡Potencial de Wrapped Top 10 Global!** Probabilidad estimada: **{probabilidad:.1f}%**")
    else:
        st.info(f"📊 **Perfil fuera del Top 10 Wrapped.** Probabilidad de entrar: **{probabilidad:.1f}%**")
        
    st.caption(f"Precisión global del árbol de decisión en el conjunto de prueba: **{acc:.2%}**")
