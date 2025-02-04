import streamlit as st
import numpy as np
import pickle  # Para cargar el modelo guardado

# Cargar el modelo entrenado
def load_model():
    filename = "model_trained_regressor.pkl.gz"
    with gzip.open(filename, 'rb') as f:
        modelo = pickle.load(f)
    return modelo


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


