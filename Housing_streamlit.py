import streamlit as st
import numpy as np
import gzip
import pickle
import os
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


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
        if not hasattr(modelo, "predict"):
            st.error("⚠️ El modelo cargado no es válido o no tiene el método 'predict()'.")
            return None
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

# --- 🏡 INTERFAZ PRINCIPAL ---
st.title("🏡 Predicción del Precio de Viviendas en Boston")

st.write(
    "Ingrese los valores de cada variable para estimar el precio de una vivienda en Boston. "
    "Cada variable tiene un rango de valores basado en los datos originales del conjunto de datos."
)

# --- 📊 BARRA LATERAL: Información del modelo ---
st.sidebar.header("📊 Parámetros del Modelo")

if modelo is not None:
    modelo_tipo = type(modelo).__name__  # Obtener tipo de modelo
    st.sidebar.write(f"📌 **Tipo de modelo:** {modelo_tipo}")

    try:
        # Obtener hiperparámetros del modelo
        params = modelo.get_params()
        st.sidebar.write("### 🔧 Hiperparámetros Ajustados:")
        
        for key, value in params.items():
            st.sidebar.write(f"🔹 **{key}:** {value}")
    
    except Exception as e:
        st.sidebar.error(f"⚠️ Error al obtener los hiperparámetros del modelo: {e}")

else:
    st.sidebar.warning("⚠️ Modelo no cargado. No se pueden mostrar los parámetros.")


# --- 📝 INPUTS DE VARIABLES ---
valores_usuario = []
for col, info in variables_info.items():
    if col == "CHAS":  # Variable categórica (0 o 1)
        valor = st.radio(f"{col} - {info['desc']}", [0, 1])
    else:
        valor = st.slider(
            f"{col} - {info['desc']}",
            min_value=float(info["min"]),
            max_value=float(info["max"]),
            value=(info["min"] + info["max"]) / 2
        )
    
    valores_usuario.append(valor)

# --- 🔮 BOTÓN DE PREDICCIÓN ---
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
