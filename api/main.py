import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
# Configuramos MLflow para que utilice un directorio local llamado "mlruns" para almacenar los experimentos, modelos y métricas.
mlflow.set_tracking_uri("file:./mlruns")  

# Creamos una aplicación FastAPI para servir el modelo de predicción de consumo energético. 
# La API tiene un endpoint para realizar predicciones basadas en las características de entrada, 
# y maneja la carga del modelo desde MLflow, así como la validación de los datos de entrada y la generación 
# de la respuesta con la predicción realizada por el modelo.
app = FastAPI(
    title="Energy Consumption Prediction API",
    description="Predice el consumo energético horario (MW) dado un instante temporal",
    version="1.0.0"
)


MODEL_NAME = "energy_gb_final_combined" 

# carga el modelo registrado en MLflow con el nombre especificado en MODEL_NAME. Si el modelo se carga correctamente, se imprime un mensaje de éxito;
try:
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/latest") # cogemos la ultima version del modelo con ese nombre registrado en MLflow
    print(f" Modelo '{MODEL_NAME}' cargado correctamente")
except Exception as e:
    print(f" Error cargando modelo: {e}")
    model = None

# Datos que recibe la api, lo que tendrá que enviar el usuario para realizar predicciones
class EnergyInput(BaseModel):
    datetime_str: str        # Formato: "2024-01-15 14:00:00"
    lag_1: float             # Consumo hora anterior (MW)
    lag_24: float            # Consumo misma hora ayer (MW)
    lag_168: float           # Consumo misma hora hace una semana (MW)
    rolling_mean_24: float   # Media móvil últimas 24h (MW)
    rolling_mean_168: float  # Media móvil última semana (MW)

    # para que aparezca un ejemplo claro en la documentación automática de FastAPI,
    #  definimos un ejemplo de entrada con valores típicos de consumo energético, lo que facilita a los usuarios entender qué formato 
    # y tipo de datos deben enviar para obtener una predicción válida.
    model_config = {
        "json_schema_extra": {
            "example": {
                "datetime_str": "2024-01-15 14:00:00",
                "lag_1": 35000.0,
                "lag_24": 34500.0,
                "lag_168": 33800.0,
                "rolling_mean_24": 34200.0,
                "rolling_mean_168": 33500.0,
            }
        }
    }

# Datos que devuelve la api, lo que el usuario recibirá como respuesta a su solicitud de predicción
class EnergyOutput(BaseModel):
    predicted_mw: float
    datetime_input: str
    model_used: str

# endpoint raíz para verificar que la API está corriendo correctamente, lo que permite a los usuarios
# y desarrolladores confirmar que el servicio está activo y listo para recibir solicitudes.
@app.get("/")
def root():
    return {"status": "ok", "message": "Energy Prediction API running"}

# endpoint de salud para verificar que el modelo se ha cargado correctamente, lo que es crucial para 
# asegurar que la API puede realizar predicciones antes de aceptar solicitudes de predicción.
@app.get("/health")
def health():
    return {"model_loaded": model is not None, "model_name": MODEL_NAME}

# endpoint de predicción que recibe los datos de entrada, valida el formato de la fecha, genera las características necesarias para el modelo,
# realiza la predicción utilizando el modelo cargado, y devuelve la predicción junto con información adicional sobre la entrada y el modelo utilizado.
@app.post("/predict", response_model=EnergyOutput)
def predict(data: EnergyInput):
    if model is None: # comprobacion de que el modelo existe
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    try:
        dt = datetime.strptime(data.datetime_str, "%Y-%m-%d %H:%M:%S") # validamos que la fecha esté en el formato correcto, 
    except ValueError:
        raise HTTPException(status_code=400, detail="Formato inválido. Usa: YYYY-MM-DD HH:MM:SS")
    # generamos un dataframe con las características necesarias para el modelo a partir de los datos de entrada, 
    # lo que permite al modelo realizar la predicción basada en las características temporales y de consumo energético proporcionadas por el usuario.
    features = pd.DataFrame([{
        'hour':             dt.hour,
        'dayofweek':        dt.weekday(),
        'month':            dt.month,
        'year':             dt.year,
        'quarter':          (dt.month - 1) // 3 + 1,
        'dayofyear':        dt.timetuple().tm_yday,
        'is_weekend':       int(dt.weekday() >= 5),
        'lag_1':            data.lag_1,
        'lag_24':           data.lag_24,
        'lag_168':          data.lag_168,
        'rolling_mean_24':  data.rolling_mean_24,
        'rolling_mean_168': data.rolling_mean_168,
    }])
    features = features.astype({
        'hour': 'int32',
        'dayofweek': 'int32',
        'month': 'int32',
        'year': 'int32',
        'quarter': 'int32',
        'dayofyear': 'int32',
        'is_weekend': 'int64',
        'lag_1': 'float64',
        'lag_24': 'float64',
        'lag_168': 'float64',
        'rolling_mean_24': 'float64',
        'rolling_mean_168': 'float64',
    })

    # realizamos la predicción utilizando el modelo cargado, lo que permite obtener una estimación del consumo energético
    # (en MW) para el instante temporal y las características proporcionadas por el usuario.
    prediction = model.predict(features)[0]

    # Devolvemos la predicción junto con información adicional sobre la entrada y el modelo utilizado, 
    # lo que proporciona al usuario no solo el valor predicho, sino también contexto sobre cuándo se realizó la predicción 
    # y qué modelo se utilizó para generar el resultado.
    return EnergyOutput(
        predicted_mw=round(float(prediction), 2),
        datetime_input=data.datetime_str,
        model_used=MODEL_NAME,
    )