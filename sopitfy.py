import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Imagen y Título
st.image("spotify.png", width=150)
st.title("Predicción de Streams")

# 2. Cargar datos y entrenar modelo en 3 líneas
df = pd.read_csv("most_streamed_spotify_2025_cleaned_v2.csv")
X = df[["daily_streams"]]
y = df["spotify_streams_total"]
model = LinearRegression().fit(X, y)

# 3. Entrada de usuario y Predicción
daily = st.number_input("Ingresa reproducciones diarias:", value=1_000_000)

if st.button("Predecir"):
    prediccion = model.predict([[daily]])[0]
    st.success(f"Total estimado: {prediccion:,.0f} streams")
