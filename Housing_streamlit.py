import streamlit as st
import numpy as np
import gzip
import pickle
import os

# Función para cargar el modelo entrenado
@st.cache_resource
def load_model():
    filename = "model_trained_regressor.pkl.gz"

    if not os.path.exists(filename):
        st.error(f"⚠️ Error: No se encontró el archivo '{filename}'.")
        return None
    
    try:
        with gzip.open(filename, 'rb') as f:
            modelo = pickle.load(f)
        return modelo
    except Exception as e:
        st.error(f"⚠️ Error al cargar el modelo: {e}")
        return None

# Cargar el modelo al iniciar la aplicación
modelo = load_model()

# Definir nombres, descripciones y rangos de las variables
variables_info = {
    "CRIM": {"desc": "Tasa de criminalidad per cápita", "min": 0.0, "max": 100.0},
    "ZN": {"desc": "Proporción de terreno residencial", "min": 0.0, "max": 100.0},
    "INDUS": {"desc": "Proporción de terreno no comercial", "min": 0.0, "max": 30.0},
    "CHAS": {"desc": "Cerca del río Charles (0: No, 1: Sí)", "min": 0, "max": 1},
    "NOX": {"desc": "Concentración de óxidos de nitrógeno (ppm)", "min": 0.3, "max": 0.9},
    "RM": {"desc": "Número promedio de habitaciones", "min": 3.0, "max": 9.0},
    "AGE": {"desc": "Proporción de viviendas antiguas (%)", "min": 0.0, "max": 100.0},
    "DIS": {"desc": "Distancia a centros de empleo", "min": 0.5, "max": 12.0},
    "RAD": {"desc": "Accesibilidad a carreteras radiales", "min": 1, "max": 24},
    "TAX": {"desc": "Tasa de impuesto a la propiedad", "min": 100, "max": 800},
    "PTRATIO": {"desc": "Ratio de alumnos por profesor", "min": 12.0, "max": 22.0},
    "B": {"desc": "Índice de población afroamericana", "min": 0.0, "max": 400.0},
    "LSTAT": {"desc": "Porcentaje de población de bajos ingresos", "min": 1.0, "max": 40.0}
}

# Crear la interfaz en Streamlit
st.title("🏡 Predicción del Precio de Viviendas en Boston")

st.write(
    "Ingrese los valores de cada variable para estimar el precio de una vivienda en Boston. "
    "Cada variable tiene un rango de valores basado en los datos originales del conjunto de datos."
)

# Crear inputs para cada variable con descripciones y rangos
valores_usuario = []
for col, info in variables_info.items():
    if col == "CHAS":  # Variable categórica (0 o 1)
        valor = st.radio(f"{col} - {info['desc']}", [0, 1])
    else:
        valor = st.slider(
            f"{col} - {info['desc']}",
            min_value=info["min"],
            max_value=info["max"],
            value=(info["min"] + info["max"]) / 2,  # Valor por defecto en el centro del rango
        )
    
    valores_usuario.append(valor)

# Botón de predicción
if st.button("Predecir Precio"):
    if modelo is not None:
        entrada = np.array(valores_usuario).reshape(1, -1)
        try:
            prediccion = modelo.predict(entrada)  # Hacer la predicción
            st.success(f"🏠 Precio estimado: ${prediccion[0] * 1000:,.2f}")  # Formato en dólares
        except Exception as e:
            st.error(f"⚠️ Error al hacer la predicción: {e}")
    else:
        st.error("⚠️ No se pudo hacer la predicción porque el modelo no está cargado.")

