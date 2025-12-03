# ml.py (actualizado para evitar explosion de cardinalidad)
import os
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


class TopKCategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Transformer que para cada columna categórica reemplaza las categorías
    menos frecuentes por '__OTHER__', manteniendo solo las top_k más frecuentes.
    Funciona con DataFrames y retorna un DataFrame con las columnas transformadas.
    """
    def __init__(self, top_k: int = 20):
        self.top_k = int(top_k)
        self.top_categories_ = {}  # dict: col -> set(top_k)

    def fit(self, X, y=None):
        if not hasattr(X, "columns"):
            X = pd.DataFrame(X)
        for col in X.columns:
            counts = X[col].value_counts(dropna=True)
            top = counts.head(self.top_k).index.tolist()
            self.top_categories_[col] = set(top)
        return self

    def transform(self, X):
        if not hasattr(X, "columns"):
            X = pd.DataFrame(X)
        Xt = X.copy()
        for col in Xt.columns:
            top_set = self.top_categories_.get(col, set())
            # replace NaN with a placeholder so imputer can work later
            Xt[col] = Xt[col].fillna("__MISSING__").astype(object)
            # map values not in top_set to '__OTHER__'
            Xt[col] = Xt[col].apply(lambda v: v if v in top_set else "__OTHER__")
        return Xt


def _build_preprocessor(X: pd.DataFrame, top_k=20):
    """
    Construye preprocesador reduciendo la cardinalidad categórica a top_k por columna.
    Devuelve el ColumnTransformer listo para usar.
    """
    # detect columns
    numeric_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # numeric pipeline
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # categorical pipeline: primero TopK, luego imputer y one-hot
    # TopKCategoricalEncoder espera un DataFrame con solo las columnas categóricas,
    # por eso la colocamos como primer step en un Pipeline que se aplicará a esas columnas.
    categorical_pipeline = Pipeline([
        ("topk", TopKCategoricalEncoder(top_k=top_k)),
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))  # sparse False for model compatibility
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ], remainder="drop", sparse_threshold=0.0)  # ensure dense output (we handle size via top_k)

    return preprocessor, numeric_cols, categorical_cols


def train_and_compare_models(csv_path: str, target_col: str="", test_size: float = 0.2,
                             random_state: int = 42, id_proceso: str = "", top_k: int = 20):
    """
    Entrena 3 algoritmos de regresión y genera reportes gráficos.
    top_k: número máximo de categorías por columna que se mantendrán (las demás pasarán a '__OTHER__').
    Retorna diccionario con métricas, ganador y rutas a imágenes.
    """
    if csv_path.endswith("amazon_limpio_api.csv"):
        target_col = "Amount"
    elif csv_path.endswith("intl_limpio_api.csv"):
        target_col = "GROSS AMT"
    elif csv_path.endswith("pl_limpio_api.csv"):
        target_col = "TP 2"    

    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Columna objetivo '{target_col}' no encontrada en el csv.")

    # separar X / y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # forzamos y numeric si es posible
    try:
        y = pd.to_numeric(y, errors="raise").astype(float)
    except Exception:
        # intentar coaccionar (si falla, avisamos)
        try:
            y = pd.to_numeric(y, errors="coerce").astype(float)
            if y.isna().all():
                raise ValueError("La columna objetivo no puede convertirse a numérico.")
        except Exception as e:
            raise ValueError(f"No se pudo convertir la columna objetivo a numérico: {e}")

    preprocessor, numeric_cols, categorical_cols = _build_preprocessor(X, top_k=top_k)

    # partición
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # modelos a evaluar
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=random_state)
    }

    results = {}
    # carpeta de reportes por proceso
    proc_dir = REPORTS_DIR / (id_proceso or "default")
    proc_dir.mkdir(parents=True, exist_ok=True)

    try:
        for name, model in models.items():
            pipeline = Pipeline([
                ("preproc", preprocessor),
                ("estimator", model)
            ])

            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)

            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae = float(mean_absolute_error(y_test, preds))
            r2 = float(r2_score(y_test, preds))

            results[name] = {
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "model_object": pipeline
            }

    except MemoryError as mem_err:
        raise MemoryError(
            "Se agotó la memoria durante el preprocesamiento/entrenamiento. "
            "Prueba reducir `top_k` (ej. a 10) para limitar la cardinalidad, o usar columnas menos categóricas. "
            f"Error original: {mem_err}"
        )

    # elegir ganador por RMSE más bajo
    winner_name = min(results.items(), key=lambda x: x[1]["rmse"])[0]
    winner = results[winner_name]["model_object"]

    # Guardar modelos (opcional)
    models_dir = proc_dir / "models"
    models_dir.mkdir(exist_ok=True)
    for name, info in results.items():
        try:
            joblib.dump(info["model_object"], models_dir / f"{name}.joblib")
        except Exception:
            # si falla guardar (por ejemplo por permisos), continuamos sin interrumpir
            pass

    # Generar gráficos
    # 1) Bar chart comparando RMSE
    rmse_values = [results[name]["rmse"] for name in results]
    model_names = list(results.keys())
    fig1_path = proc_dir / "rmse_comparison.png"
    plt.figure(figsize=(8, 5))
    plt.bar(model_names, rmse_values)
    plt.ylabel("RMSE")
    plt.title("Comparación de RMSE entre modelos")
    plt.tight_layout()
    plt.savefig(fig1_path)
    plt.close()

    # 2) Residuals plot del ganador
    preds_winner = winner.predict(X_test)
    residuals = y_test - preds_winner
    fig2_path = proc_dir / "residuals_winner.png"
    plt.figure(figsize=(8, 5))
    plt.scatter(preds_winner, residuals, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicho")
    plt.ylabel("Residual (Real - Predicho)")
    plt.title(f"Residuals del modelo ganador: {winner_name}")
    plt.tight_layout()
    plt.savefig(fig2_path)
    plt.close()

    # 3) Predicho vs Real plot para el ganador
    fig3_path = proc_dir / "pred_vs_real_winner.png"
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds_winner, alpha=0.6)
    lims = [min(y_test.min(), preds_winner.min()), max(y_test.max(), preds_winner.max())]
    plt.plot(lims, lims, "--", color="gray")
    plt.xlabel("Real")
    plt.ylabel("Predicho")
    plt.title(f"Predicho vs Real ({winner_name})")
    plt.tight_layout()
    plt.savefig(fig3_path)
    plt.close()

    # Cleanup: remove model_object from results antes de serializar
    serializable_results = {}
    for name, info in results.items():
        serializable_results[name] = {
            "rmse": info["rmse"],
            "mae": info["mae"],
            "r2": info["r2"],
        }

    output = {
        "metrics": serializable_results,
        "winner": winner_name,
        "figures": {
            "rmse_comparison": str(fig1_path),
            "residuals_winner": str(fig2_path),
            "pred_vs_real_winner": str(fig3_path)
        },
        "models_dir": str(models_dir),
        "top_k_used": int(top_k)
    }

    # Escribir un resumen JSON para este proceso
    summary_path = proc_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    return output
