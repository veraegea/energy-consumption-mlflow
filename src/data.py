import kagglehub
import pandas as pd
from pathlib import Path

# Cargamos los datos utilizados desde Kaggle, y realizamos un preprocesamiento básico para convertir la columna de fecha a datetime, 
# ordenar el dataframe por fecha y renombrar la columna de consumo energético para facilitar su uso en el análisis posterior.
def load_data():
    path = kagglehub.dataset_download('robikscube/hourly-energy-consumption')
    df = pd.read_csv(Path(path) / 'PJM_Load_hourly.csv')
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)
    df = df.rename(columns={'PJM_Load_MW': 'energy_mw'})
    return df