import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------
# Configuración inicial de la página
# ---------------------------------------------------------
st.set_page_config(
    page_title="Predicción de Éxito en Spotify",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Clasificador de Canciones: ¿Será un Top 100 en Spotify?")
st.markdown("""
Esta aplicación utiliza un modelo de Machine Learning (**Random Forest Classifier**) 
para predecir si una canción logrará estar en el **Top 100** en función de sus métricas de reproducción diaria y colaboración.
""")

# ---------------------------------------------------------
# Cargar y Preparar los Datos
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('most_streamed_spotify_2025_cleaned_v2.csv')
    # Definimos la variable objetivo: 1 si el rank está entre las primeras 100, 0 en otro caso
    df['is_top_100'] = (df['rank'] <= 100).astype(int)
    return df

df = load_data()

# ---------------------------------------------------------
# Barra Lateral (Sidebar): Parámetros interactivos
# ---------------------------------------------------------
st.sidebar.header("⚙️ Configuración del Modelo")

# Hiperparámetros del modelo
n_estimators = st.sidebar.slider("Número de Árboles (n_estimators)", min_value=10, max_value=200, value=100, step=10)
max_depth = st.sidebar.slider("Profundidad Máxima del Árbol", min_value=1, max_value=20, value=10)
test_size = st.sidebar.slider("Proporción de Datos de Prueba (Test Size)", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

# ---------------------------------------------------------
# Entrenamiento del Modelo
# ---------------------------------------------------------
features = ['daily_streams', 'billed_artist_count', 'is_collaboration_int', 'daily_stream_share_pct']
X = df[features]
y = df['is_top_100']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# ---------------------------------------------------------
# Panel Principal: Secciones con Pestañas
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Exploración de Datos", "🧠 Rendimiento del Modelo", "🔮 Realizar Predicción"])

# --- Pestaña 1: Exploración ---
with tab1:
    st.subheader("Muestra de Datos del Dataset")
    st.dataframe(df[['rank', 'track', 'artist', 'daily_streams', 'billed_artist_count', 'is_top_100']].head(10))
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Distribución de Canciones Top 100 vs Otros**")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='is_top_100', palette='Set2', ax=ax)
        ax.set_xticklabels(['Fuera del Top 100 (0)', 'En el Top 100 (1)'])
        ax.set_ylabel("Cantidad de Canciones")
        st.pyplot(fig)

    with col2:
        st.markdown("**Relación entre Reproducciones Diarias y Rango**")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='daily_streams', y='rank', hue='is_top_100', palette='Set1', ax=ax)
        ax.invert_yaxis()  # El rango 1 va arriba
        ax.set_xlabel("Reproducciones Diarias")
        ax.set_ylabel("Ranking")
        st.pyplot(fig)

# --- Pestaña 2: Evaluación del Modelo ---
with tab2:
    st.subheader("Métricas de Rendimiento")
    st.metric(label="Exactitud del Modelo (Accuracy)", value=f"{acc * 100:.2f}%")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("**Matriz de Confusión**")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Predicho 0', 'Predicho 1'],
                    yticklabels=['Real 0', 'Real 1'])
        st.pyplot(fig)
        
    with col_b:
        st.markdown("**Importancia de las Características**")
        importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
        fig, ax = plt.subplots()
        importances.plot(kind='barh', color='skyblue', ax=ax)
        ax.set_title("Importancia en las Predicciones")
        st.pyplot(fig)

# --- Pestaña 3: Predicción con Datos del Usuario ---
with tab3:
    st.subheader("Ingresa los datos de una nueva canción para predecir:")
    
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        user_daily_streams = st.number_input(
            "Reproducciones Diarias Estimadas (daily_streams)", 
            min_value=0, max_value=5000000, value=500000, step=10000
        )
        user_artists = st.number_input(
            "Número de Artistas Acreditados", 
            min_value=1, max_value=5, value=1
        )
        
    with col_input2:
        user_collab = st.selectbox(
            "¿Es una Colaboración?", 
            options=["No", "Sí"]
        )
        user_share = st.slider(
            "Porcentaje del cuota diaria global (daily_stream_share_pct)", 
            min_value=0.0, max_value=1.0, value=0.15, step=0.01
        )
        
    is_collab_int = 1 if user_collab == "Sí" else 0
    
    # Botón para ejecutar la predicción
    if st.button("🚀 Predecir Posición"):
        input_data = pd.DataFrame([[
            user_daily_streams, user_artists, is_collab_int, user_share
        ]], columns=features)
        
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        st.divider()
        if prediction == 1:
            st.success(f"🎉 **¡La canción tiene altas probabilidades de entrar al TOP 100!** (Probabilidad: {probability*100:.1f}%)")
        else:
            st.warning(f"📉 **Es poco probable que entre al Top 100.** (Probabilidad de ingresar: {probability*100:.1f}%)")
