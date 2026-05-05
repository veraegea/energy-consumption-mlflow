#  Energy Consumption Prediction with MLflow

Predicción del consumo eléctrico horario utilizando el dataset **PJM Load (Kaggle)**, con seguimiento de experimentos mediante **MLflow** y despliegue del modelo final en producción mediante **FastAPI**.

---

##  Objetivo

El objetivo del proyecto es:

- Entrenar distintos modelos de Machine Learning
- Comparar su rendimiento mediante MLflow
- Ajustar hiperparámetros de forma iterativa
- Seleccionar el mejor modelo en base a métricas
- Desplegar el modelo en una API para inferencia

---

##  Dataset

- Fuente: Kaggle — PJM Hourly Energy Consumption
- Granularidad: datos horarios
- Tamaño: varios miles de observaciones
- Tipo de problema: **regresión en series temporales**

---

##  Pipeline del proyecto

1. **Carga de datos**
2. **Feature engineering**
   - Variables temporales (hora, día, mes…)
   - Lags (`lag_1`, `lag_24`, `lag_168`)
   - Medias móviles
3. **Split temporal (sin shuffle)**
4. **Entrenamiento de modelos**
5. **Tracking con MLflow**
6. **Refinamiento de hiperparámetros**
7. **Selección del mejor modelo**
8. **Despliegue con FastAPI**

---

##  Cómo ejecutar

```bash
# 1. Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate   # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Entrenar modelos (tanda 1)
set PYTHONPATH=src
python src/train.py

# 4. Refinamiento de modelos (tanda 2)
python src/refine_gb.py

# 5. Lanzar MLflow UI
mlflow ui
# → http://localhost:5000

# 6. Lanzar API
uvicorn api.main:app --reload
# → http://localhost:8000/docs

# 7. Probar predicción con curl
curl -X POST http://127.0.0.1:8000/predict ^
-H "Content-Type: application/json" ^
-d "{\"datetime_str\":\"2024-01-15 14:00:00\",\"lag_1\":35000,\"lag_24\":34500,\"lag_168\":33800,\"rolling_mean_24\":34200,\"rolling_mean_168\":33500}"
