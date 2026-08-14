import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Configuración de página e Imagen
st.set_page_config(page_title="Spotify Decision Tree 5-Var", page_icon="🎵", layout="wide")
st.image("spotify.png", use_container_width=True)

# 2. Cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv("most_streamed_spotify_2025_cleaned_v2.csv")
    # Target Binario: 1 si supera los 250M de streams totales, 0 si no
    df['gran_escala'] = (df['spotify_streams_total'] > 250000000).astype(int)
    return df

df = load_data()

# 3. Definir EXACTAMENTE 5 Variables de Entrada
features = [
    'daily_streams',             # 1. Reproducciones diarias actuales
    'daily_streams_rank',        # 2. Posición en el ranking diario
    'daily_stream_share_pct',    # 3. Cuota de mercado diaria
    'billed_artist_count',       # 4. Cantidad de artistas principales
    'is_collaboration_int'       # 5. Indicador binario de colaboración (0 o 1)
]

X = df[features]
y = df['gran_escala']

# 4. Entrenar el Árbol de Decisión
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = DecisionTreeClassifier(max_depth=4, random_state=42, criterion='entropy').fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))

# 5. Interfaz de Usuario
st.title("🌳 Árbol de Decisión: Clasificador Gran Escala (+250M Streams)")
st.write("Evalúa si el comportamiento diario de la canción indica que superará los **250 millones de reproducciones acumuladas**.")

# Formulario organizado en 2 columnas
col1, col2 = st.columns(2)

with col1:
    diarias = st.number_input("1. Streams Diarios (daily_streams):", value=500000, step=25000)
    rank_diario = st.number_input("2. Ranking Diario (daily_streams_rank):", min_value=1, value=35)
    cuota = st.number_input("3. Cuota Diaria % (daily_stream_share_pct):", value=0.22, step=0.01)

with col2:
    artistas = st.number_input("4. Cantidad de Artistas (billed_artist_count):", min_value=1, max_value=5, value=1)
    colaboracion = st.checkbox("5. ¿Es Colaboración? (is_collaboration_int)")

# 6. Predicción
if st.button("🔮 Evaluar con Árbol de Decisión"):
    input_data = [[
        diarias, 
        rank_diario, 
        cuota, 
        artistas, 
        int(colaboracion)
    ]]
    
    prediccion = model.predict(input_data)[0]
    probabilidad = model.predict_proba(input_data)[0][1] * 100
    
    st.markdown("---")
    if prediccion == 1:
        st.success(f"🚀 **¡Lanzamiento de Gran Escala!** Probabilidad calculada: **{probabilidad:.1f}%**")
    else:
        st.info(f"📈 **Escala Moderada/Estándar.** Probabilidad calculada: **{probabilidad:.1f}%**")
        
    st.caption(f"Precisión global del Árbol de Decisión en datos de prueba: **{acc:.2%}**")
