import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Spotify Clustering", layout="centered")
st.title("🎧 Agrupamiento de Canciones (K-Means)")

# 1. Cargar datos
@st.cache_data
def load_data():
    return pd.read_csv("most_streamed_spotify_2025_cleaned_v2.csv")

df = load_data()

# 2. Selección de variables y clustering
features = ['spotify_streams_total', 'daily_streams', 'daily_stream_share_pct']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Definir 3 clusters (ej. "Bajo", "Medio", "Alto rendimiento")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

st.subheader("Segmentación del Catálogo")

# 3. Entrada para predecir el cluster de una nueva canción
st.write("### Evalúa el perfil de tu canción:")
total_streams = st.number_input("Reproducciones Totales:", value=int(df['spotify_streams_total'].median()))
daily_streams = st.number_input("Reproducciones Diarias:", value=int(df['daily_streams'].median()))
share_pct = st.number_input("Cuota Diaria (%):", value=float(df['daily_stream_share_pct'].mean()))

if st.button("📌 Asignar Grupo"):
    new_data = scaler.transform([[total_streams, daily_streams, share_pct]])
    cluster_pred = kmeans.predict(new_data)[0]
    
    st.success(f"La canción pertenece al **Grupo {cluster_pred + 1}**")

# 4. Visualización de los Clusters
st.markdown("---")
st.subheader("Mapa de Grupos")
fig, ax = plt.subplots(figsize=(6, 4))
sns.scatterplot(
    data=df, 
    x='daily_streams', 
    y='spotify_streams_total', 
    hue='Cluster', 
    palette='viridis', 
    ax=ax
)
ax.set_title("Segmentación por Reproducciones Diarias vs Totales")
st.pyplot(fig)
