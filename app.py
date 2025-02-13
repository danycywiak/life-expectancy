import streamlit as st
import pickle
import pandas as pd

# Cargar modelo y scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler_data = pickle.load(f)
scaler = scaler_data["scaler"]

# Función para obtener los datos de un país
def get_country_data(country: str, filepath: str = "Life_Expectancy_Data.csv"):
    dataset = pd.read_csv(filepath, na_values=["N/A", "Unknown"], keep_default_na=True)
    country_data = dataset.loc[dataset["Country"] == country].copy()

    if country_data.empty:
        return None, None

    country_data["Status"] = country_data["Status"].map(lambda x: 0 if x == "Developing" else 1)
    country_data = country_data.drop(columns=["Country", "Continent"])
    country_data = country_data.fillna(country_data.mean())

    actual_life_expectancy = country_data["Life_expectancy"].values[0]
    features = country_data.drop(columns=["Life_expectancy", "Year"])

    return features, actual_life_expectancy

# Título de la app
st.title("Predicción de Esperanza de Vida 🌍")

# Input del usuario
country = st.text_input("Escribe el nombre de un país:", "")

# Botón de predicción
if st.button("Predecir"):
    if country:
        features, actual_value = get_country_data(country)
        
        if features is not None:
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)

            st.success(f"📢 Predicción para {country}: {prediction[0][0]:.2f} años")
            st.info(f"🎯 Valor real en {country}: {actual_value:.2f} años")
        else:
            st.error(f"⚠️ El país '{country}' no está en el dataset.")
    else:
        st.warning("⚠️ Ingresa un país antes de predecir.")

                   


