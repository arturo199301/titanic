import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Configuración de página e Imagen
st.set_page_config(page_title="Spotify SVM 5-Var Classifier", page_icon="🎵", layout="wide")
st.image("spotify.png", use_container_width=True)

# 2. Cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv("most_streamed_spotify_2025_cleaned_v2.csv")
    # Target Binario: 1 si la cuota de mercado diaria supera el 0.30%, 0 en caso contrario
    df['dominio_masivo'] = (df['daily_stream_share_pct'] > 0.30).astype(int)
    return df

df = load_data()

# 3. Definir EXACTAMENTE 5 Variables de Entrada
features = [
    'spotify_streams_total', 
    'daily_streams', 
    'daily_streams_rank', 
    'billed_artist_count', 
    'is_collaboration_int'
]

X = df[features]
y = df['dominio_masivo']

# 4. Escalar Datos y Entrenar SVM
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVC habilitando probability=True para obtener la probabilidad estimada
model = SVC(kernel='rbf', probability=True, random_state=42).fit(X_train_scaled, y_train)

acc = accuracy_score(y_test, model.predict(X_test_scaled))

# 5. Interfaz de Usuario
st.title("⚔️ Clasificador SVM: Predictor Dominio Masivo (>0.30% Cuota)")
st.write("Evalúa si la canción alcanzará **Dominio Masivo Diario** usando Máquinas de Vectores de Soporte.")

# Formulario en 2 columnas
col1, col2 = st.columns(2)

with col1:
    totales = st.number_input("1. Streams Totales (spotify_streams_total):", value=300000000, step=10000000)
    diarias = st.number_input("2. Streams Diarios (daily_streams):", value=750000, step=25000)
    rank_diario = st.number_input("3. Ranking Diario (daily_streams_rank):", min_value=1, value=15)

with col2:
    artistas = st.number_input("4. Cantidad de Artistas (billed_artist_count):", min_value=1, max_value=5, value=1)
    colaboracion = st.checkbox("5. ¿Es Colaboración? (is_collaboration_int)")

# 6. Predicción
if st.button("🔮 Clasificar con SVM"):
    raw_input = [[
        totales, 
        diarias, 
        rank_diario, 
        artistas, 
        int(colaboracion)
    ]]
    
    scaled_input = scaler.transform(raw_input)
    
    prediccion = model.predict(scaled_input)[0]
    probabilidad = model.predict_proba(scaled_input)[0][1] * 100
    
    st.markdown("---")
    if prediccion == 1:
        st.success(f"🔥 **¡Dominio Masivo Diario!** Probabilidad estimada: **{probabilidad:.1f}%**")
    else:
        st.info(f"📊 **Cuota Estándar.** Probabilidad de dominio masivo: **{probabilidad:.1f}%**")
        
    st.caption(f"Precisión global del modelo SVM en test: **{acc:.2%}**")
