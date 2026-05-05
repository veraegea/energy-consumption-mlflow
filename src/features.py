import pandas as pd

# En esta función, a partir del dataframe original, se crean nuevas características temporales 
# (como la hora, el día de la semana, el mes, etc.) y características basadas en lags y medias móviles del consumo energético. 
# Estas nuevas características pueden ayudar a los modelos de machine learning a capturar patrones temporales y estacionales en los datos de consumo energético.
def create_features(df: pd.DataFrame):
    df = df.copy()
    df['hour']       = df['Datetime'].dt.hour
    df['dayofweek']  = df['Datetime'].dt.dayofweek
    df['month']      = df['Datetime'].dt.month
    df['year']       = df['Datetime'].dt.year
    df['quarter']    = df['Datetime'].dt.quarter
    df['dayofyear']  = df['Datetime'].dt.dayofyear
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    # Lag features: la clave para series temporales
    df['lag_1']   = df['energy_mw'].shift(1)
    df['lag_24']  = df['energy_mw'].shift(24)
    df['lag_168'] = df['energy_mw'].shift(168)  # semana anterior

    # Rolling means (con shift(1) para no usar datos del momento actual)
    df['rolling_mean_24']  = df['energy_mw'].shift(1).rolling(24).mean()
    df['rolling_mean_168'] = df['energy_mw'].shift(1).rolling(168).mean()

    return df.dropna()

# defino las columnas que se usarán como características para el modelo, incluyendo tanto las características temporales 
# como las basadas en lags y medias móviles.
FEATURE_COLS = [
    'hour', 'dayofweek', 'month', 'year', 'quarter',
    'dayofyear', 'is_weekend',
    'lag_1', 'lag_24', 'lag_168',
    'rolling_mean_24', 'rolling_mean_168',
]

# Esta función realiza una división temporal de los datos en conjuntos de entrenamiento y prueba, respetando el orden cronológico de los datos. 
# Esto es crucial para evitar el data leakage en problemas de series temporales, donde el modelo no debería tener acceso a información futura 
# durante el entrenamiento. La función devuelve las características y las etiquetas para ambos conjuntos.
def temporal_train_test_split(df: pd.DataFrame, test_ratio: float = 0.2):
    """Split temporal SIN shuffle: respeta el orden cronológico."""
    split_idx = int(len(df) * (1 - test_ratio))
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]
    return (
        train[FEATURE_COLS], test[FEATURE_COLS],
        train['energy_mw'], test['energy_mw']
    )