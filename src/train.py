import argparse
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor # para los modelos de machine learning que se entrenarán
from sklearn.linear_model import Ridge # para el modelo lineal de referencia
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score #métricas de evaluación para los modelos
from mlflow.models.signature import infer_signature # para guardar en MLflow la estructura esperada de entrada y salida del modelo.

# Este script es el núcleo del proceso de entrenamiento y evaluación de modelos. 
# Aquí se definen las funciones para crear los modelos, ejecutar los experimentos y registrar los resultados en MLflow.
# Esto permite que el script encuentre archivos que están en la misma carpeta
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Importamos las funciones de carga de datos y creación de características desde los módulos correspondientes
from data import load_data
from features import create_features, temporal_train_test_split, FEATURE_COLS

# Funcion para obtener el modelo según el nombre y los hiperparámetros especificados. 
# Esto permite una fácil experimentación con diferentes modelos y configuraciones.
def get_model(model_name, n_estimators, max_depth, learning_rate, alpha):
    if model_name == 'ridge':
        return Ridge(alpha=alpha)
    if model_name == 'random_forest':
        return RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=42, n_jobs=-1
        )
    if model_name == 'gradient_boosting':
        return GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, random_state=42
        )
    raise ValueError(f'Modelo no soportado: {model_name}')

# Esta función es la encargada de ejecutar un experimento completo: entrenar el modelo, hacer predicciones, calcular métricas y registrar todo en MLflow.
def run_experiment(model, model_name, params, X_train, X_test, y_train, y_test, notes=''):
    with mlflow.start_run(run_name=model_name): # abrimos la ejecucion de MLflow
        # Tags descriptivos para organizar y filtrar los experimentos en MLflow UI
        mlflow.set_tag('model_type', model_name)
        mlflow.set_tag('dataset', 'PJM_Load_hourly')
        mlflow.set_tag('split', 'temporal_80_20')
        if notes:
            mlflow.set_tag('notes', notes)

        # Log de parámetros (guardamos tanto los hiperparámetros del modelo como información sobre el tamaño de los conjuntos de entrenamiento y prueba, 
        # y el número de características utilizadas)
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_param('train_rows', len(X_train))
        mlflow.log_param('test_rows', len(X_test))
        mlflow.log_param('n_features', len(FEATURE_COLS))

        # Entrenamiento
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Métricas
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        mape = float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100)

        # Log de métricas (guardamos las métricas de evaluación del modelo para compararlas entre diferentes ejecuciones en MLflow UI)
        mlflow.log_metric('MAE', mae)
        mlflow.log_metric('RMSE', rmse)
        mlflow.log_metric('R2', r2)
        mlflow.log_metric('MAPE', mape)

        # Log del modelo con firma (guardamos el modelo entrenado en MLflow, junto con la firma que describe la estructura esperada
        # de entrada y salida del modelo, lo que facilita su uso posterior para inferencia)
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path='model',
            signature=signature,
            input_example=X_test.head(3),
            registered_model_name=f'energy_{model_name}',
        )

        # Guardamos las features importantes (si aplica), lo que puede ayudar a entender qué características están 
        # influyendo más en las predicciones del modelo.
        if hasattr(model, 'feature_importances_'):
            imp = pd.DataFrame({
                'feature': FEATURE_COLS,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            imp.to_csv('feature_importances.csv', index=False)
            mlflow.log_artifact('feature_importances.csv')

        # Guardamos las predicciones vs valores reales en un CSV, lo que permite un análisis posterior de los resultados y 
        # la generación de gráficos de predicciones vs realidad.
        results = pd.DataFrame({'y_true': y_test.values, 'y_pred': y_pred})
        results.to_csv('predictions.csv', index=False)
        mlflow.log_artifact('predictions.csv')

        print(f'\n{"="*55}')
        print(f'Run: {model_name}')
        print(f'  MAE:  {mae:,.2f} MW')
        print(f'  RMSE: {rmse:,.2f} MW')
        print(f'  R²:   {r2:.4f}')
        print(f'  MAPE: {mape:.2f}%')
        print(f'{"="*55}')

# Este bloque se ejecuta cuando se corre el script directamente. Aquí se configuran los experimentos a realizar,
#  incluyendo la carga y preparación de los datos, y la ejecución de varios experimentos con diferentes modelos y 
# configuraciones de hiperparámetros, registrando todo en MLflow.
if __name__ == '__main__':
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment('energy_consumption_prediction') # establecemos el nombre del experimento en MLflow, lo que permite organizar y comparar las ejecuciones relacionadas con la predicción de consumo energético.
    # cargamos los datos, creamos las características y realizamos la división temporal entre entrenamiento y prueba.
    print('Cargando y preparando datos...')
    df = load_data()
    df = create_features(df)
    X_train, X_test, y_train, y_test = temporal_train_test_split(df)
    print(f'Train: {len(X_train)} muestras | Test: {len(X_test)} muestras')

    # ── Experimento 1: Ridge (baseline, sin lags) Modelo linea de referencia 
    run_experiment(
        model=Ridge(alpha=1.0),
        model_name='ridge_baseline',
        params={'alpha': 1.0},
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        notes='Baseline lineal — referencia mínima'
    )

    # ── Experimento 2: Random Forest básico (poco profundo)
    run_experiment(
        model=RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1),
        model_name='rf_v1_shallow',
        params={'n_estimators': 50, 'max_depth': 8},
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        notes='RF poco profundo — detectar señal inicial'
    )

    # ── Experimento 3: Random Forest mejorado (Modelo más potente, pero con cierto control para evitar sobreajuste)
    run_experiment(
        model=RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1),
        model_name='rf_v2_medium',
        params={'n_estimators': 100, 'max_depth': 15, 'min_samples_split': 5},
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        notes='RF con más profundidad y min_samples_split'
    )

    # ── Experimento 4: Random Forest sin límite de profundidad
    run_experiment(
        model=RandomForestRegressor(n_estimators=200, max_depth=None, min_samples_split=2, random_state=42, n_jobs=-1),
        model_name='rf_v3_deep',
        params={'n_estimators': 200, 'max_depth': 'None', 'min_samples_split': 2},
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        notes='RF sin límite — evaluar overfitting'
    )

    # ── Experimento 5: Gradient Boosting estándar
    run_experiment(
        model=GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42),
        model_name='gb_v1_standard',
        params={'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 4},
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        notes='GB estándar'
    )

    # ── Experimento 6: Gradient Boosting afinado (más arboles, learning rate más bajo y subsample para reducir varianza)
    run_experiment(
        model=GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8, random_state=42),
        model_name='gb_v2_tuned',
        params={'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 5, 'subsample': 0.8},
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        notes='GB con learning rate bajo y subsample — reduce varianza'
    )

    print('\n Todos los experimentos completados.')
    print('Ejecuta: mlflow ui')