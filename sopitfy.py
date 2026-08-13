import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from PIL import Image
import os

# Configuración de la página
st.set_page_config(
    page_title="Predicción Spotify 2025",
    page_icon="🎵",
    layout="wide"
)

# 1. Mostrar la imagen del logo si está disponible
if os.path.exists('spotify.png'):
    st.image('spotify.png', width=300)
elif os.path.exists('input_file_0.png'):
    st.image('input_file_0.png', width=300)

st.title("🎵 Predicción de Reproducciones Totales en Spotify")
st.write("Esta aplicación utiliza un modelo de **Random Forest Regressor** para estimar el total de reproducciones acumuladas.")

# 2. Cargar el dataset limpio
@st.cache_data
def load_data():
    if os.path.exists('most_streamed_spotify_2025_cleaned_v2.csv'):
        return pd.read_csv('most_streamed_spotify_2025_cleaned_v2.csv')
    else:
        return pd.read_csv('most_streamed_spotify_2025.csv')

df = load_data()

# 3. Preparación de variables y modelo
features = ['daily_streams', 'daily_streams_rank', 'daily_stream_share_pct', 'billed_artist_count', 'is_collaboration']

# Asegurar que is_collaboration sea entero para el modelo
X = df[features].copy()
X['is_collaboration'] = X['is_collaboration'].astype(int)
y = df['spotify_streams_total']

# Dividir y entrenar el modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluación
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Mostrar métricas en la barra lateral
st.sidebar.header("📊 Métricas del Modelo")
st.sidebar.metric("Precisión (R²)", f"{r2:.2f}")
st.sidebar.metric("Error Medio (MAE)", f"{mae:,.0f}")

# 4. Formulario de predicción
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎛️ Parámetros de la Canción")
    daily_streams = st.number_input("Reproducciones Diarias Promedio", min_value=0, value=2_000_000, step=100_000)
    daily_streams_rank = st.number_input("Ranking Diario", min_value=1, max_value=1000, value=10)
    daily_stream_share_pct = st.slider("Porcentaje de Participación Diaria (%)", min_value=0.0, max_value=1.0, value=0.15, step=0.01)
    billed_artist_count = st.number_input("Número de Artistas", min_value=1, max_value=10, value=1)
    is_collab = st.checkbox("¿Es una colaboración?")

with col2:
    st.subheader("🔮 Resultado")
    if st.button("Predecir Reproducciones Totales"):
        input_df = pd.DataFrame({
            'daily_streams': [daily_streams],
            'daily_streams_rank': [daily_streams_rank],
            'daily_stream_share_pct': [daily_stream_share_pct],
            'billed_artist_count': [billed_artist_count],
            'is_collaboration': [int(is_collab)]
        })
        
        prediccion = model.predict(input_df)[0]
        st.success("**Streams Totales Estimados:**")
        st.title(f"{prediccion:,.0f}")

st.divider()
st.subheader("📋 Datos Utilizados")
st.dataframe(df.head(10), use_container_width=True)
