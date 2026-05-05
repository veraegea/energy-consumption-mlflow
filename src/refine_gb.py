import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import mlflow
from sklearn.ensemble import GradientBoostingRegressor

from data import load_data
from features import create_features, temporal_train_test_split
from train import run_experiment


if __name__ == '__main__':
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment('energy_consumption_prediction')

    print('Cargando y preparando datos...')
    df = load_data()
    df = create_features(df)
    X_train, X_test, y_train, y_test = temporal_train_test_split(df)
    print(f'Train: {len(X_train)} muestras | Test: {len(X_test)} muestras')

    # Experimento 7: Gradient Boosting afinado iteración 2 (ajustes adicionales basados en resultados anteriores)
    # Lr más bajo + más árboles, para intentar mejorar la capacidad de generalización sin sobreajustar.
    run_experiment(
        model=GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.02,
            max_depth=5,
            subsample=0.8,
            random_state=42
        ),
        model_name='gb_r1_lr002_n500',
        params={
            'n_estimators': 500,
            'learning_rate': 0.02,
            'max_depth': 5,
            'subsample': 0.8
        },
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        notes='Iteracion 2 — lr mas bajo con mas arboles'
    )

    # Experimento 8: Gradient Boosting con depth menor para reducir complejidad y evaluar si mejora la generalización.
    run_experiment(
        model=GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=42
        ),
        model_name='gb_r2_depth3',
        params={
            'n_estimators': 300,
            'learning_rate': 0.05,
            'max_depth': 3,
            'subsample': 0.8
        },
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        notes='Iteracion 2 — depth menor para reducir complejidad'
    )

    # Experimento 9: Gradient Boosting con depth mayor para capturar más complejidad, evaluando el impacto en el rendimiento y posible sobreajuste.
    run_experiment(
        model=GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            random_state=42
        ),
        model_name='gb_r3_depth7',
        params={
            'n_estimators': 300,
            'learning_rate': 0.05,
            'max_depth': 7,
            'subsample': 0.8
        },
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        notes='Iteracion 2 — depth mayor para capturar mas complejidad'
    )
    # Experimento 10: Gradient Boosting con subsample menor para regularizar el modelo, evaluando si reduce la varianza y mejora la generalización.
    run_experiment(
        model=GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.6,
            random_state=42
        ),
        model_name='gb_r4_sub06',
        params={
            'n_estimators': 300,
            'learning_rate': 0.05,
            'max_depth': 5,
            'subsample': 0.6
        },
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        notes='Iteracion 2 — subsample menor para regularizar'
    )

    # Experimento 11: Combinación de ajustes para intentar maximizar el rendimiento, basándonos en los resultados anteriores.
    run_experiment(
    model=GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.7,
        random_state=42
    ),
    model_name='gb_final_combined',
    params={
        'n_estimators': 400,
        'learning_rate': 0.05,
        'max_depth': 7,
        'subsample': 0.7
    },
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    notes='Modelo final combinando depth alto y subsample moderado'
)

    print('\n Refinamiento completado. Recarga MLflow UI para comparar.')