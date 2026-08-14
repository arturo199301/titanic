import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =============================================================================
# 1. CONFIGURACIÓN INICIAL Y ESTILOS DE LA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Spotify Music Classifier Pro",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Imagen de encabezado
st.image("spotify.png", use_container_width=True)

# CSS Personalizado para tarjetas y métricas
st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #1DB954;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. CARGA Y PREPROCESAMIENTO DE DATOS
# =============================================================================
@st.cache_data
def load_data():
    # Cargar dataset
    df = pd.read_csv("most_streamed_spotify_2025_cleaned_v2.csv")
    
    # Feature Engineering (Variables Derivadas)
    df['streams_per_artist'] = df['spotify_streams_total'] / df['billed_artist_count']
    df['daily_per_artist'] = df['daily_streams'] / df['billed_artist_count']
    
    # Variable Target (Clasificación Binaria: 1 = Hit Top 50, 0 = Estándar)
    df['target_hit'] = (df['daily_streams_rank'] <= 50).astype(int)
    
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error al cargar el archivo de datos: {e}")
    st.stop()

# =============================================================================
# 3. CONFIGURACIÓN DEL MODELO Y ENTRENAMIENTO
# =============================================================================
# Definición de las 6 variables de entrada
FEATURE_COLUMNS = [
    'spotify_streams_total',
    'daily_streams',
    'daily_stream_share_pct',
    'billed_artist_count',
    'is_collaboration_int',
    'streams_per_artist'
]

X = df[FEATURE_COLUMNS]
y = df['target_hit']

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Modelo RandomForest
@st.cache_resource
def train_model(X_tr, y_tr):
    model = RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42)
    model.fit(X_tr, y_tr)
    return model

model = train_model(X_train, y_train)

# Evaluación rápida
y_pred = model.predict(X_test)
model_acc = accuracy_score(y_test, y_pred)

# =============================================================================
# 4. BARRA LATERAL (SIDEBAR) - NAVEGACIÓN Y CONFIGURACIÓN
# =============================================================================
with st.sidebar:
    st.header("⚙️ Configuración del Modelo")
    st.write("Métricas generales del entrenamiento:")
    st.metric(label="Precisión en Test (Accuracy)", value=f"{model_acc:.2%}")
    st.metric(label="Total de Datos", value=f"{len(df):,} filas")
    st.markdown("---")
    st.info("💡 **Tip:** Ajusta los hiperparámetros en las pestañas superiores para predecir.")

# =============================================================================
# 5. ESTRUCTURA PRINCIPAL BASADA EN PESTAÑAS (TABS)
# =============================================================================
st.title("🎯 Sistema de Clasificación y Analítica Spotify")
st.write("Plataforma estructurada de *Machine Learning* para la predicción del desempeño de temas musicales.")

tab_pred, tab_analytics, tab_data = st.tabs([
    "🔮 Predictor Interactivo", 
    "📊 Interpretabilidad y Métricas", 
    "📁 Explorador de Datos"
])

# -----------------------------------------------------------------------------
# TAB 1: PREDICTOR INTERACTIVO
# -----------------------------------------------------------------------------
with tab_pred:
    st.subheader("📋 Formulario de Evaluación de Canción")
    st.caption("Ingresa los 6 parámetros del tema para estimar si clasificará dentro del Top 50.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 📈 Reproducciones")
            totales = st.number_input(
                "Streams Totales Acumulados:", 
                value=250000000, 
                step=10000000,
                help="Historico total de reproducciones en Spotify"
            )
            diarias = st.number_input(
                "Streams Diarios Estimados:", 
                value=600000, 
                step=25000
            )

        with col2:
            st.markdown("##### 🏆 Mercado")
            cuota = st.number_input(
                "Cuota Diaria de Mercado (%):", 
                value=0.25, 
                step=0.01,
                min_value=0.0,
                max_value=100.0
            )
            artistas = st.number_input(
                "Cantidad de Artistas:", 
                min_value=1, 
                max_value=10, 
                value=2
            )

        with col3:
            st.markdown("##### 👥 Formato y Métricas Calculadas")
            es_colab = st.checkbox("¿Es Colaboración entre Artistas?", value=True)
            
            # Cálculo automático de variable derivada
            streams_p_artist = totales / artistas
            st.markdown(f"""
                <div class="metric-card">
                    <small>Streams/Artista (Automático)</small><br>
                    <strong>{streams_p_artist:,.0f}</strong>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        submit_button = st.form_submit_button("🚀 Ejecutar Predicción", use_container_width=True)

    # Procesamiento del resultado al presionar el botón del formulario
    if submit_button:
        input_data = [[
            totales, 
            diarias, 
            cuota, 
            artistas, 
            int(es_colab), 
            streams_p_artist
        ]]
        
        prediccion = model.predict(input_data)[0]
        probabilidades = model.predict_proba(input_data)[0]
        prob_top50 = probabilidades[1] * 100

        st.markdown("---")
        st.subheader("📌 Resultado del Análisis")

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            if prediccion == 1:
                st.success("🔥 **¡Clasificación: HIT TOP 50!**")
                st.write("El modelo detecta patrones sólidos de alto impacto constante.")
            else:
                st.info("📉 **Clasificación: Rendimiento Estándar**")
                st.write("El tema mantiene un desempeño dentro del rango promedio fuera del Top 50.")

        with res_col2:
            st.metric(label="Probabilidad de Pertenece al Top 50", value=f"{prob_top50:.1f}%")
            st.progress(prob_top50 / 100)

# -----------------------------------------------------------------------------
# TAB 2: INTERPRETABILIDAD Y MÉTRICAS
# -----------------------------------------------------------------------------
with tab_analytics:
    st.subheader("📊 Diagnóstico del Modelo")
    st.write("Análisis detallado de cómo el algoritmo toma decisiones y evalúa los datos.")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("##### Importancia de las Variables (Feature Importance)")
        importances = pd.DataFrame({
            'Variable': FEATURE_COLUMNS,
            'Importancia': model.feature_importances_
        }).sort_values('Importancia', ascending=True)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(importances['Variable'], importances['Importancia'], color='#1DB954')
        ax.set_xlabel("Peso de Importancia")
        st.pyplot(fig)

    with col_chart2:
        st.markdown("##### Matriz de Confusión (Datos de Prueba)")
        cm = confusion_matrix(y_test, y_pred)
        
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax_cm,
                    xticklabels=['Estándar', 'Top 50'],
                    yticklabels=['Estándar', 'Top 50'])
        ax_cm.set_ylabel('Real')
        ax_cm.set_xlabel('Predicho')
        st.pyplot(fig_cm)

# -----------------------------------------------------------------------------
# TAB 3: EXPLORADOR DE DATOS
# -----------------------------------------------------------------------------
with tab_data:
    st.subheader("📁 Inspección del Dataset")
    st.write("Muestra representativa de los datos utilizados para el entrenamiento:")
    st.dataframe(df[FEATURE_COLUMNS + ['target_hit']].head(20), use_container_width=True)
