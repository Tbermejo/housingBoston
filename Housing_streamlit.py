import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import gzip
import pickle

# Cargar el modelo entrenado
with open("model_trained_regressor.pkl.gz", "rb") as f:
    modelo = pickle.load(f)

# Definir los nombres de las variables del dataset
column_names = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS",
    "RAD", "TAX", "PTRATIO", "B", "LSTAT"
]

# Crear la interfaz en Streamlit
st.title("Predicción del Precio de Viviendas en Boston 🏡")

# Crear inputs para cada variable
valores_usuario = []
for col in column_names:
    if col == "CHAS":  # Variable categórica (0 o 1)
        valor = st.radio(f"{col} (Cerca del río Charles)", [0, 1])
    else:
        valor = st.number_input(f"{col}", min_value=0.0, format="%.2f")
    
    valores_usuario.append(valor)

# Botón de predicción
if st.button("Predecir Precio"):
    entrada = np.array(valores_usuario).reshape(1, -1)  # Convertir en array 2D
    prediccion = modelo.predict(entrada)  # Hacer la predicción
    st.success(f"🏠 Precio estimado: ${prediccion[0] * 1000:,.2f}")  # Formato en dólares
