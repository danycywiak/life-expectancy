import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import pickle 
import typer

app = typer.Typer()

###CLEAN DATA####
#_________________________________________________________#
@app.command()
def clean_data(filepath: str = "input/Life_Expectancy_Data.csv", logs: bool = True, dropExtraFeatures: bool = True):
    """Carga y limpia los datos, luego los guarda en clean_data.pkl"""
    
    dataset = pd.read_csv(filepath, na_values=["N/A", "Unknown"], keep_default_na=True)

    dataset['Status'] = dataset['Status'].map(lambda x: 0 if x == "Developing" else 1)


    for column in dataset.columns:
        if column in ["Country", "Continent"]:
            continue
        dataset[column] = dataset[column].fillna(dataset[column].mean())

    dataset.to_csv("input/clean_data.csv")

    dataset = dataset.drop(columns=['Country', 'Continent'])
    dataset.to_csv("input/clean_data_no_countries.csv")
    print(" Datos limpios guardados.")


####SELECT COUNTRY TO PREDICT ITS LIFE EXPECTANCY####

#_________________________________________________________#
def get_country_data(country: str, logs: bool = False):
    """Filtra los datos de un país específico para la predicción"""

    dataset = pd.read_csv("./input/clean_data.csv")
    country_data = dataset[dataset["Country"] == country]

    if country_data.empty:
        print(f" El país '{country}' no está en el dataset.")
        return None, None

    actual_life_expectancy = country_data["Life_expectancy"].values[0]
    features = country_data.drop(columns=["Life_expectancy", "Year"])

    return features, actual_life_expectancy



####TRAIN MODEL#####
#_________________________________________________________#
@app.command()
def train_model(test_size: float = 0.2, epochs: int = 30, logs: bool = True):
    """Entrena el modelo y guarda model.pkl y scaler.pkl"""

    dataset = pd.read_csv("./input/clean_data_no_countries.csv")

    features = dataset.drop(columns=["Life_expectancy", "Year"])
    labels = dataset["Life_expectancy"]

    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=test_size, random_state=23
    )

    scaler = MinMaxScaler()
    features_train = scaler.fit_transform(features_train)
    features_test = scaler.transform(features_test)

    model = Sequential([
        Dense(64, activation='relu', input_shape=(features_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    model.fit(features_train, labels_train, epochs=epochs, validation_data=(features_test, labels_test), verbose=1)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("scaler.pkl", "wb") as f:
        pickle.dump({"scaler": scaler}, f)

    print("Modelo y scaler guardados correctamente.")



###CALL get_country_data AND PREDICT VALUE####
#_________________________________________________________#
@app.command()
def predict(country: str, logs: bool = True):
    """Predice la esperanza de vida de un país con el modelo entrenado"""

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler_data = pickle.load(f)
    scaler = scaler_data["scaler"]

    features, actual_life_expectancy = get_country_data(country, logs=False)

    if features is None or features.empty:
        print(f" El país '{country}' no está en el dataset o no tiene datos.")
        return None

    features = features.drop(columns=["Country", "Continent"])

    if features is None:
        print(f" El país '{country}' no está en el dataset.")
        return

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)

    print(f"Prediccion para {country}: {prediction[0][0]:.2f} años")
    if actual_life_expectancy is not None:
        print(f"Valor real en {country}: {actual_life_expectancy:.2f} años")


@app.command()
def run():
    """Ejecuta el script"""
    clean_data()
    train_model()
    predict("Argentina")


if __name__ == "__main__":
    app()
