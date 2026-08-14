import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Configuración de página e Imagen
st.set_page_config(page_title="Spotify Market Share Classifier", page_icon="🎵")
st.image("spotify.png", use_container_width=True)

# 2. Cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv("most_streamed_spotify_2025_cleaned_v2.csv")
    # Target: 1 si el porcentaje de cuota diaria es superior a la mediana, 0 si es inferior
    mediana_cuota = df['daily_stream_share_pct'].median()
    df['cuota_alta'] = (df['daily_stream_share_pct'] > mediana_cuota).astype(int)
    return df

df = load_data()

# 3. Entrenar el Modelo (Random Forest con 3 Variables)
features = ['daily_streams_rank', 'billed_artist_count', 'is_collaboration_int']
X = df[features]
y = df['cuota_alta']

model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)

# 4. Interfaz de Usuario
st.title("🎯 Clasificador: ¿Alta Cuota de Mercado?")
st.write("Predice si la canción obtendrá una cuota de mercado superior al promedio.")

# 3 Variables de Entrada
rank_diario = st.number_input("1. Ranking diario (daily_streams_rank):", min_value=1, value=50)
artistas = st.number_input("2. Cantidad de artistas (billed_artist_count):", min_value=1, max_value=5, value=1)
es_colaboracion = st.checkbox("3. ¿Es una colaboración? (is_collaboration_int)")

# 5. Predicción
if st.button("🔮 Evaluar Cuota"):
    input_data = [[rank_diario, artistas, int(es_colaboracion)]]
    
    prediccion = model.predict(input_data)[0]
    probabilidad = model.predict_proba(input_data)[0][1] * 100
    
    st.markdown("---")
    if prediccion == 1:
        st.success(f"📊 **¡Cuota de Mercado Alta!** Probabilidad: **{probabilidad:.1f}%**")
    else:
        st.info(f"📉 **Cuota de Mercado Normal/Baja.** Probabilidad de cuota alta: **{probabilidad:.1f}%**")
