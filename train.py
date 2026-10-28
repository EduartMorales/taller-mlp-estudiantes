"""
train.py - Servicio de entrenamiento del MLP
Taller MLP: predicción de aprobación de estudiantes

Pasos: carga -> imputación con mediana -> verificación de balanceo ->
división train/test estratificada -> estandarización -> búsqueda de
arquitectura del MLP (GridSearchCV) -> evaluación -> guardado del modelo.

Uso:  python train.py
"""
import json
from datetime import datetime

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

RUTA_DATOS = "dataset_estudiantes_balanceado.csv"
RUTA_MODELO = "modelo_mlp.pkl"

# Se excluye Calificacion_Final: es combinación lineal de estas variables
# y define directamente la clase (fuga de datos).
FEATURES = ["Tareas", "Examenes", "Asistencia",
            "Estudio_Horas", "Sueno_Horas", "Extracurriculares"]
OBJETIVO = "Aprobado"


MEJORES_PARAMS = {"hidden_layer_sizes": (32, 16), "activation": "tanh", "alpha": 0.0001}


def entrenar(ruta_datos: str = RUTA_DATOS, ruta_modelo: str = RUTA_MODELO,
             rapido: bool = False) -> dict:
    """rapido=False: busca la mejor arquitectura con GridSearchCV (más lento).
    rapido=True: entrena directamente con los mejores hiperparámetros encontrados
    (para reentrenar desde el servidor gratuito, que tiene poca CPU)."""
    # 1. Carga
    df = pd.read_csv(ruta_datos)

    # 2. Imputación con la mediana de cada columna
    df = df.dropna(subset=[OBJETIVO])
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())

    # 3. Verificación del balanceo
    balance = df[OBJETIVO].value_counts(normalize=True).round(3).to_dict()

    # 4. División estratificada (80/20)
    X, y = df[FEATURES], df[OBJETIVO].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # 5. Estandarización (media 0, varianza 1) ajustada solo con entrenamiento
    scaler = StandardScaler().fit(X_train)
    X_train_sc = scaler.transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # 6. MLP con búsqueda de la mejor arquitectura
    param_grid = {
        "hidden_layer_sizes": [(8,), (16,), (16, 8), (32, 16)],
        "activation": ["relu", "tanh"],
        "alpha": [0.0001, 0.001, 0.01],
    }
    if rapido:
        mlp = MLPClassifier(max_iter=3000, random_state=42, **MEJORES_PARAMS)
        mlp.fit(X_train_sc, y_train)
        mejores, f1_cv = MEJORES_PARAMS, None
    else:
        grid = GridSearchCV(
            MLPClassifier(max_iter=3000, random_state=42),
            param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring="f1", n_jobs=-1)
        grid.fit(X_train_sc, y_train)
        mlp = grid.best_estimator_
        mejores, f1_cv = grid.best_params_, round(float(grid.best_score_), 4)

    # 7. Evaluación en el set de prueba independiente
    y_pred = mlp.predict(X_test_sc)
    metricas = {
        "fecha_entrenamiento": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "registros": int(len(df)),
        "balance_clases": {str(k): v for k, v in balance.items()},
        "modo": "rapido" if rapido else "busqueda_grid",
        "mejores_hiperparametros": {k: str(v) for k, v in mejores.items()},
        "f1_validacion_cruzada": f1_cv,
        "accuracy_prueba": round(float(accuracy_score(y_test, y_pred)), 4),
        "matriz_confusion": confusion_matrix(y_test, y_pred).tolist(),
        "reporte": classification_report(
            y_test, y_pred, target_names=["Reprobado", "Aprobado"],
            output_dict=True, digits=4),
    }

    # 8. Guardado
    joblib.dump({"modelo": mlp, "scaler": scaler, "features": FEATURES,
                 "medianas": X_train.median().to_dict(), "metricas": metricas},
                ruta_modelo)
    return metricas


if __name__ == "__main__":
    m = entrenar()
    print("Mejores hiperparámetros:", m["mejores_hiperparametros"])
    print("Accuracy en prueba:", m["accuracy_prueba"])
    print("Matriz de confusión:", m["matriz_confusion"])
    print(json.dumps(m["reporte"], indent=2, ensure_ascii=False))
    print(f"Modelo guardado en {RUTA_MODELO}")
