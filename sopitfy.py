import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Configuración de página e Imagen
st.set_page_config(page_title="Spotify Top 10 Decision Tree", page_icon="🎵", layout="wide")
st.image("spotify.png", use_container_width=True)

# 2. Cargar y preparar datos
@st.cache_data
def load_data():
    df = pd.read_csv("most_streamed_spotify_2025_cleaned_v2.csv")
    
    # Target Binario: 1 si está en el Top 10 del ranking diario, 0 si no
    df['top_10_viral'] = (df['daily_streams_rank'] <= 10).astype(int)
    
    return df

df = load_data()

# 3. Definir las Variables de Entrada (6 variables)
features = [
    'spotify_streams_total',       # 1. Reproducciones totales
    'daily_streams',               # 2. Reproducciones diarias
    'daily_stream_share_pct',      # 3. Porcentaje de cuota diaria
    'billed_artist_count',         # 4. Cantidad de artistas
    'is_collaboration_int',        # 5. Indicador de colaboración (0 o 1)
    'wrapped_global_top10_rank'    # 6. Posición en Wrapped Top 10
]

X = df[features]
y = df['top_10_viral']

# 4. Entrenar el Árbol de Decisión
# Usamos max_depth=4 para evitar el sobreajuste (overfitting) y mantener el árbol interpretable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = DecisionTreeClassifier(max_depth=4, random_state=42, criterion='gini').fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))

# 5. Interfaz de Usuario
st.title("🌲 Árbol de Decisión: Predictor Top 10 Viral")
st.write("Evalúa las métricas de la canción para descubrir si la estructura del árbol la clasifica dentro del **Top 10 Diario**.")

# Formulario organizado en 2 columnas
col1, col2 = st.columns(2)

with col1:
    totales = st.number_input("1. Streams Totales (spotify_streams_total):", value=300000000, step=10000000)
    diarias = st.number_input("2. Streams Diarios (daily_streams):", value=850000, step=25000)
    cuota = st.number_input("3. Cuota Diaria % (daily_stream_share_pct):", value=0.35, step=0.01)

with col2:
    artistas = st.number_input("4. Cantidad de Artistas (billed_artist_count):", min_value=1, max_value=5, value=1)
    colaboracion = st.checkbox("5. ¿Es Colaboración? (is_collaboration_int)")
    wrapped_rank = st.number_input("6. Ranking Top 10 Wrapped (0 si no aplica):", min_value=0, max_value=10, value=0)

# 6. Predicción
if st.button("🔮 Consultar al Árbol de Decisión"):
    input_data = [[
        totales, 
        diarias, 
        cuota, 
        artistas, 
        int(colaboracion), 
        wrapped_rank
    ]]
    
    prediccion = model.predict(input_data)[0]
    probabilidad = model.predict_proba(input_data)[0][1] * 100
    
    st.markdown("---")
    if prediccion == 1:
        st.success(f"🏆 **¡Éxito Viral! Entra al codiciado Top 10.** Probabilidad estimada: **{probabilidad:.1f}%**")
    else:
        st.info(f"📊 **Se queda fuera del Top 10.** Probabilidad de entrar: **{probabilidad:.1f}%**")
        
    st.caption(f"Precisión global del Árbol de Decisión en datos de prueba: **{acc:.2%}**")
