import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import typer

app = typer.Typer()

### CLEAN DATA ###
#_________________________________________________________#
@app.command()
def preprocess_input_data(filepath: str = "Life_Expectancy_Data.csv"):
    """Carga y limpia los datos, luego los guarda en clean_data.csv"""
    
    dataset = pd.read_csv(filepath, na_values=["N/A", "Unknown"], keep_default_na=True)
    dataset['Status'] = dataset['Status'].map(lambda x: 0 if x == "Developing" else 1)
    dataset = dataset.fillna(dataset.mean())
    dataset.to_csv("clean_data.csv", index=False)
    print("✅ Datos limpios guardados en 'clean_data.csv'.")


### SELECT COUNTRY TO PREDICT ###
#_________________________________________________________#
def get_country_data(country: str):
    """Filtra los datos de un país específico para la predicción"""

    dataset = pd.read_csv("clean_data.csv")
    country_data = dataset.loc[dataset["Country"] == country].copy()

    if country_data.empty:
        print(f"⚠ El país '{country}' no está en el dataset.")
        return None, None

    # Eliminar columnas
    features = country_data.drop(columns=["Life_expectancy", "Country", "Year", "Continent"])
    actual_life_expectancy = country_data["Life_expectancy"].values[0]

    return features, actual_life_expectancy


### TRAIN MODEL ###
#_________________________________________________________#
@app.command()
def train(test_size: float = 0.2, epochs: int = 30):
    """Entrena el modelo y guarda model.h5 y scaler.pkl"""

    dataset = pd.read_csv("clean_data.csv")
    features = dataset.drop(columns=["Life_expectancy", "Year", "Country", "Continent"])
    labels = dataset["Life_expectancy"]

    # Split into train and test
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=test_size, random_state=23
    )

    # Escalar 
    scaler = MinMaxScaler()
    features_train = scaler.fit_transform(features_train)
    features_test = scaler.transform(features_test)

    # Model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(features_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    model.fit(features_train, labels_train, epochs=epochs, validation_data=(features_test, labels_test), verbose=1)

    # Guardar el modelo en formato .h5
    model.save("model.h5")

    # Guardar el escalador
    with open("scaler.pkl", "wb") as f:
        pickle.dump({"scaler": scaler}, f)

    print("✅ Modelo y scaler guardados correctamente.")


### PREDICT LIFE EXPECTANCY ###
#_________________________________________________________#
@app.command()
def predict(country: str):
    """Predice la esperanza de vida de un país con el modelo entrenado"""

    # Cargar el modelo guardado en .h5
    model = load_model("model.h5")

    # Cargar el scaler
    with open("scaler.pkl", "rb") as f:
        scaler_data = pickle.load(f)
    scaler = scaler_data["scaler"]

    # Obtener datos del país seleccionado
    features, actual_life_expectancy = get_country_data(country)

    if features is None:
        print(f"El país '{country}' no está en el dataset.")
        return

    # Escalar los datos y hacer la predicción
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)

    print(f"Predicción para {country}: {prediction[0][0]:.2f} años")
    if actual_life_expectancy is not None:
        print(f"Valor real en {country}: {actual_life_expectancy:.2f} años")


### RUN EVERYTHING EXCEPT PREDICTION ###
#_________________________________________________________#
@app.command()
def run():
    """Ejecuta todo el flujo: Preprocesamiento y entrenamiento"""
    preprocess_input_data()
    train()


if __name__ == "__main__":
    app()
