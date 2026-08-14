import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Configuración de página e Imagen
st.set_page_config(page_title="Spotify XGBoost 6-Var Classifier", page_icon="🎵", layout="wide")
st.image("spotify.png", use_container_width=True)

# 2. Cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv("most_streamed_spotify_2025_cleaned_v2.csv")
    
    # Feature Engineering (Variable 6)
    df['streams_per_artist'] = df['spotify_streams_total'] / df['billed_artist_count']
    
    # Target Binario: 1 si está en el Top 30 diario, 0 si no
    df['is_top30'] = (df['daily_streams_rank'] <= 30).astype(int)
    
    return df

df = load_data()

# 3. Definir EXACTAMENTE 6 Variables de Entrada
features = [
    'spotify_streams_total',       # 1. Reproducciones totales acumuladas
    'daily_streams',               # 2. Reproducciones diarias
    'daily_stream_share_pct',      # 3. Cuota diaria de mercado %
    'billed_artist_count',         # 4. Cantidad de artistas
    'is_collaboration_int',        # 5. Indicador binario de colaboración (0 o 1)
    'streams_per_artist'           # 6. Promedio de streams acumulados por artista
]

X = df[features]
y = df['is_top30']

# 4. Entrenar el Modelo (XGBoost)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = XGBClassifier(
    n_estimators=100, 
    learning_rate=0.08, 
    max_depth=4, 
    random_state=42, 
    eval_metric='logloss'
).fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))

# 5. Interfaz de Usuario
st.title("⚡ Clasificador XGBoost: Predictor Top 30 (6 Variables)")
st.write("Ingresa los **6 parámetros** para evaluar si la canción pertenece al **Top 30 Diario**.")

# Formulario organizado en 3 columnas
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### 📈 Volúmenes")
    totales = st.number_input("1. Streams Totales (spotify_streams_total):", value=220000000, step=10000000)
    diarias = st.number_input("2. Streams Diarios (daily_streams):", value=550000, step=25000)

with col2:
    st.markdown("##### 🏆 Mercado")
    cuota = st.number_input("3. Cuota Diaria % (daily_stream_share_pct):", value=0.24, step=0.01)
    artistas = st.number_input("4. Cantidad de Artistas (billed_artist_count):", min_value=1, max_value=5, value=1)

with col3:
    st.markdown("##### 👥 Formato y Métricas Derivadas")
    es_colab = st.checkbox("5. ¿Es Colaboración? (is_collaboration_int)")
    
    # Cálculo automático de la Variable 6
    streams_p_artist = totales / artistas
    st.info(f"6. Streams/Artista (auto): **{streams_p_artist:,.0f}**")

# 6. Predicción
if st.button("🔮 Clasificar con XGBoost", use_container_width=True):
    input_data = [[
        totales, 
        diarias, 
        cuota, 
        artistas, 
        int(es_colab), 
        streams_p_artist
    ]]
    
    prediccion = model.predict(input_data)[0]
    probabilidad = model.predict_proba(input_data)[0][1] * 100
    
    st.markdown("---")
    if prediccion == 1:
        st.success(f"🔥 **¡Pertenece al Top 30!** Probabilidad estimada: **{probabilidad:.1f}%**")
    else:
        st.info(f"📉 **Fuera del Top 30.** Probabilidad estimada de entrar: **{probabilidad:.1f}%**")
        
    st.caption(f"Precisión global del modelo XGBoost en test: **{acc:.2%}**")
