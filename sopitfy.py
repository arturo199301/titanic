import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# 1. Encabezado con imagen y título
st.image("spotify.png", width=150)
st.title("🎵 Predicción Completa de Streams")

# 2. Cargar datos
df = pd.read_csv("most_streamed_spotify_2025_cleaned_v2.csv")

# 3. Definir características (argumentos) y variable a predecir
features = ['daily_streams', 'daily_streams_rank', 'billed_artist_count', 'is_collaboration']

X = df[features].copy()
X['is_collaboration'] = X['is_collaboration'].astype(int)  # Convertir True/False a 1/0
y = df['spotify_streams_total']

# 4. Entrenar modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 5. Entradas del usuario (Argumentos de entrada)
daily_streams = st.number_input("Reproducciones diarias:", value=1_500_000, step=100_000)
rank = st.number_input("Ranking diario (puesto en lista):", min_value=1, value=10)
artists =st.number_input("Cantidad de artistas en la canción:", min_value=1, max_value=5, value=1)
is_collab = st.checkbox("¿Es una colaboración entre varios artistas?")

# 6. Botón de predicción
if st.button("Predecir Total"):
    # Crear vector con todos los argumentos
    datos_entrada = pd.DataFrame([{
        'daily_streams': daily_streams,
        'daily_streams_rank': rank,
        'billed_artist_count': artists,
        'is_collaboration': int(is_collab)
    }])
    
    prediccion = model.predict(datos_entrada)[0]
    st.success(f"**Streams totales estimados:** {prediccion:,.0f}")
