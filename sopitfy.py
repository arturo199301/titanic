import streamlit as st
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

# 1. Configuración de página e Imagen
st.set_page_config(page_title="Spotify Viral Classifier", page_icon="🎵")
st.image("spotify.png", use_container_width=True)

# 2. Cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv("most_streamed_spotify_2025_cleaned_v2.csv")
    # Target: 1 si la cuota de mercado diaria supera el 0.25%, 0 en caso contrario
    df['impacto_viral'] = (df['daily_stream_share_pct'] > 0.25).astype(int)
    return df

df = load_data()

# 3. Entrenar el Modelo (SGDClassifier con 3 Variables)
features = ['spotify_streams_total', 'daily_streams', 'daily_streams_rank']
X = df[features]
y = df['impacto_viral']

# Escalado de características (recomendado para SGD)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42).fit(X_scaled, y)

# 4. Interfaz de Usuario
st.title("¿Canción Viral de Alto Impacto en spotify 2025?")
st.write("Predice si la canción superará el **0.25% de cuota de mercado diaria**.")

# 3 Variables de Entrada
col1, col2 = st.columns(2)

with col1:
    totales = st.number_input("1. Reproducciones Totales:", value=300000000, step=10000000)
    diarias = st.number_input("2. Reproducciones Diarias:", value=600000, step=25000)

with col2:
    rank_diario = st.number_input("3. Ranking Diario):", min_value=1, value=20)

# 5. Predicción
if st.button("🔮 Clasificar Impacto"):
    input_scaled = scaler.transform([[totales, diarias, rank_diario]])
    
    prediccion = model.predict(input_scaled)[0]
    probabilidad = model.predict_proba(input_scaled)[0][1] * 100
    
    st.markdown("---")
    if prediccion == 1:
        st.success(f"**¡Alto Impacto Viral!** Probabilidad: **{probabilidad:.1f}%**")
    else:
        st.info(f"**Impacto Estándar.** Probabilidad de ser alto impacto: **{probabilidad:.1f}%**")
