import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Configuración de página e Imagen
st.set_page_config(page_title="Spotify 7-Var Classifier", page_icon="🎵", layout="wide")
st.image("spotify.png", use_container_width=True)

# 2. Cargar y preparar datos (7 Variables)
@st.cache_data
def load_data():
    df = pd.read_csv("most_streamed_spotify_2025_cleaned_v2.csv")
    
    # Feature Engineering (Variable 7: Promedio de streams diarios por artista)
    df['daily_streams_per_artist'] = df['daily_streams'] / df['billed_artist_count']
    
    # Target Binario: 1 si es éxito consistente (Top 50 Y >200M streams), 0 si no
    df['exito_consistente'] = (
        (df['daily_streams_rank'] <= 50) & 
        (df['spotify_streams_total'] > 200000000)
    ).astype(int)
    
    return df

df = load_data()

# 3. Definir EXACTAMENTE 7 Variables de Entrada
features = [
    'spotify_streams_total',         # 1. Escuchas acumuladas totales
    'daily_streams',                 # 2. Escuchas registradas en el día
    'daily_streams_rank',            # 3. Posición en el ranking diario
    'daily_stream_share_pct',        # 4. Cuota diaria de mercado %
    'billed_artist_count',           # 5. Cantidad de artistas
    'is_collaboration_int',          # 6. Indicador de colaboración (0 o 1)
    'daily_streams_per_artist'       # 7. Promedio de escuchas diarias por artista
]

X = df[features]
y = df['exito_consistente']

# 4. Entrenar el Modelo (Random Forest)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))

# 5. Interfaz de Usuario
st.title("🎯 Clasificador Random Forest: Éxito Consistente (7 Variables)")
st.write("Ingresa los **7 parámetros** para evaluar si la canción se clasifica como un **Éxito Consistente**.")

# Formulario organizado en 3 columnas
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### 📈 Volúmenes")
    totales = st.number_input("1. Streams Totales (spotify_streams_total):", value=250000000, step=10000000)
    diarias = st.number_input("2. Streams Diarios (daily_streams):", value=600000, step=25000)

with col2:
    st.markdown("##### 🏆 Ranking y Mercado")
    rank_diario = st.number_input("3. Ranking Diario (daily_streams_rank):", min_value=1, value=25)
    cuota = st.number_input("4. Cuota Diaria % (daily_stream_share_pct):", value=0.25, step=0.01)

with col3:
    st.markdown("##### 👥 Artistas y Formato")
    artistas = st.number_input("5. Cantidad de Artistas (billed_artist_count):", min_value=1, max_value=5, value=2)
    es_colab = st.checkbox("6. ¿Es Colaboración? (is_collaboration_int)", value=True)
    
    # Cálculo automático de la Variable 7
    daily_p_artist = diarias / artistas
    st.info(f"7. Diarias/Artista (auto): **{daily_p_artist:,.0f}**")

# 6. Predicción
if st.button("🔮 Clasificar Canción", use_container_width=True):
    input_data = [[
        totales, 
        diarias, 
        rank_diario, 
        cuota, 
        artistas, 
        int(es_colab), 
        daily_p_artist
    ]]
    
    prediccion = model.predict(input_data)[0]
    probabilidad = model.predict_proba(input_data)[0][1] * 100
    
    st.markdown("---")
    if prediccion == 1:
        st.success(f"🔥 **¡Éxito Consistente!** Probabilidad estimada: **{probabilidad:.1f}%**")
    else:
        st.info(f"📉 **Desempeño Estándar.** Probabilidad estimada: **{probabilidad:.1f}%**")
        
    st.caption(f"Precisión global del modelo en test: **{acc:.2%}**")
