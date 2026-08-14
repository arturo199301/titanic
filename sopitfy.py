import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# 1. Configuración y Título Sencillo
st.set_page_config(page_title="Predictor Sencillo", page_icon="🎵")
st.title("🎵 Predictor Rápido de Éxito Musical")
st.markdown("Ingresa los datos y el modelo te dirá si la canción tiene potencial de éxito.")

# ---------------------------------------------------------
# 2. Generar Datos de Ejemplo (Sin usar CSV externo)
# ---------------------------------------------------------
@st.cache_data # Para que no regenere datos cada vez
def generar_datos():
    # Creamos 100 canciones de ejemplo manualmente
    np.random.seed(42)
    reproducciones = np.random.randint(50000, 3000000, 100) # De 50k a 3M
    compartidos = np.random.randint(1000, 500000, 100)      # De 1k a 500k
    
    # Una regla sencilla para el ejemplo: si tiene muchas repros y compartidos, es éxito (1)
    exito = ((reproducciones > 1500000) & (compartidos > 100000)).astype(int)
    
    df = pd.DataFrame({
        'Reproducciones': reproducciones,
        'Compartidos': compartidos,
        'Es_Exito': exito # Esta es nuestra columna objetivo (0 o 1)
    })
    return df

df = generar_datos()

# ---------------------------------------------------------
# 3. Entrenar el Modelo (Automático al iniciar)
# ---------------------------------------------------------
# Definir características (X) y objetivo (y)
X = df[['Reproducciones', 'Compartidos']]
y = df['Es_Exito']

# Usamos el modelo más sencillo con configuración por defecto
modelo = RandomForestClassifier(random_state=42)
modelo.fit(X, y)

# ---------------------------------------------------------
# 4. Interfaz de Usuario para Predecir
# ---------------------------------------------------------
st.subheader("🔮 Ingresa datos de la nueva canción:")

# Campos de entrada numéricos
input_repros = st.number_input("Reproducciones Diarias", min_value=0, value=1000000, step=50000)
input_shares = st.number_input("Veces Compartida", min_value=0, value=50000, step=5000)

# Botón para ejecutar la magia
if st.button("🚀 Predecir Resultado"):
    # Preparar el dato de entrada como una tabla pequeña
    entrada_usuario = pd.DataFrame([[input_repros, input_shares]], columns=['Reproducciones', 'Compartidos'])
    
    # Realizar predicción
    prediccion = modelo.predict(entrada_usuario)[0]
    
    # Mostrar resultado final
    st.divider()
    if prediccion == 1:
        st.success("🎉 **¡Es muy probable que sea un ÉXITO!**")
        st.balloons() # ¡Efecto divertido!
    else:
        st.warning("📉 **Es poco probable que sea un éxito masivo.**")

# (Opcional) Ver datos de ejemplo
if st.checkbox("Ver datos de ejemplo usados para entrenar"):
    st.write(df.head())
