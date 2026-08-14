import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 1. Configuración de página e Imagen
st.set_page_config(page_title="Spotify KNN Classifier", page_icon="🎵")
st.image("spotify.png", use_container_width=True)

# 2. Cargar datos
@st.cache_data
def load_data():
    return pd.read_csv("most_streamed_spotify_2025_cleaned_v2.csv")

df = load_data()

# 3. Entrenar el Modelo KNN (4 Variables de Rendimiento -> Target: Colaboración)
features = ['spotify_streams_total', 'daily_streams', 'daily_streams_rank', 'daily_stream_share_pct']
X = df[features]
y = df['is_collaboration_int']

# Escalado necesario para KNN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KNeighborsClassifier(n_neighbors=5).fit(X_scaled, y)

# 4. Interfaz de Usuario
st.title("🎯 Clasificador KNN: Detección de Formato")
st.write("Predice si la canción tiene perfil de **Colaboración** basándose en el comportamiento de sus métricas.")

# 4 Variables de Entrada
col1, col2 = st.columns(2)

with col1:
    totales = st.number_input("1. Reproducciones Totales:", value=180000000, step=10000000)
    diarias = st.number_input("2. Reproducciones Diarias:", value=320000, step=20000)

with col2:
    rank_diario = st.number_input("3. Ranking Diario:", min_value=1, value=90)
    cuota = st.number_input("4. Cuota Diaria (%):", value=0.16, step=0.01)

# 5. Predicción
if st.button("🔮 Clasificar Formato"):
    input_scaled = scaler.transform([[totales, diarias, rank_diario, cuota]])
    
    prediccion = model.predict(input_scaled)[0]
    probabilidad = model.predict_proba(input_scaled)[0][1] * 100
    
    st.markdown("---")
    if prediccion == 1:
        st.success(f"🤝 **Perfil de Colaboración.** Probabilidad: **{probabilidad:.1f}%**")
    else:
        st.info(f"👤 **Perfil de Solista.** Probabilidad de ser colaboración: **{probabilidad:.1f}%**")
