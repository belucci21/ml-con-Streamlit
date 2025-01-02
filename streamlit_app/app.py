import streamlit as st
import pandas as pd
import numpy as np
import joblib
from helpers.notebook_helper_functions import impute

# Cargar el modelo y los objetos auxiliares
model = joblib.load('helpers/model.pkl')
imputer = joblib.load('helpers/imputer.pkl')
transformer = joblib.load('helpers/quantile_transformer.pkl')

# Título de la aplicación
st.title("Predicción con Machine Learning")

# Entrada de datos
st.write("Ingrese las características para hacer una predicción:")
input_data = st.text_input("Datos (separados por comas)", "5.1, 3.5, 1.4, 0.2")

if st.button("Predecir"):
    try:
        # Convertir los datos de entrada en un array numpy
        features = np.array([float(x) for x in input_data.split(",")]).reshape(1, -1)

        # Imputar y transformar los datos
        transformed_features = transformer.transform(features)
        imputed_features = imputer.transform(transformed_features)

        # Hacer la predicción
        prediction = model.predict(imputed_features)
        st.success(f"Predicción: {prediction[0]}")
    except Exception as e:
        st.error(f"Error al procesar los datos: {str(e)}")

import streamlit as st
import joblib
import numpy as np

# Cargar el modelo y los preprocesadores
model = joblib.load('model.pkl')
imputer = joblib.load('imputer.pkl')
quantile_transformer = joblib.load('quantile_transformer.pkl')

# Configuración de la página
st.title("Clasificador Iris")
st.write("""
Ingrese las características de una flor Iris para predecir su especie.
""")

# Crear formulario para entrada de datos
with st.form("prediction_form"):
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)
    
    # Botón para hacer la predicción
    submit_button = st.form_submit_button("Predecir")

# Realizar predicción al enviar el formulario
if submit_button:
    try:
        # Crear un array con los valores ingresados
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Aplicar preprocesamiento
        input_data = imputer.transform(input_data)
        input_data = quantile_transformer.transform(input_data)
        
        # Hacer la predicción
        prediction = model.predict(input_data)
        species = ["Setosa", "Versicolor", "Virginica"][prediction[0]]
        
        # Mostrar el resultado
        st.success(f"La especie predicha es: **{species}**")
    except Exception as e:
        st.error("Ocurrió un error al realizar la predicción. Verifique los valores ingresados.")
