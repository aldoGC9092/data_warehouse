import pandas as pd
import os
from etl import transformar_amazon, transformar_intl, transformar_pl, cargar_datos_limpios

def ejecutar_etl_desde_api(filepath: str):
    nombre = os.path.basename(filepath).lower()
    df = pd.read_csv(filepath, low_memory=False)

    if "amazon" in nombre:
        df_limpio = transformar_amazon(df)
        nombre_salida = "amazon_limpio_api.csv"
    elif "intl" in nombre or "international" in nombre:
        df_limpio = transformar_intl(df)
        nombre_salida = "intl_limpio_api.csv"
    elif "march" in nombre:
        df_limpio = transformar_pl(df)
        nombre_salida = "pl_limpio_api.csv"
    else:
        return {
            "estado": "error",
            "mensaje": "Archivo no reconocido",
            "errores": 1,
            "filas_transformadas": 0
        }
    
    if df_limpio is None:
        return {
            "estado": "error",
            "mensaje": "Transformación fallida: DataFrame vacío o inválido",
            "errores": 1,
            "filas_transformadas": 0
        }

    cargar_datos_limpios(df_limpio, nombre_salida)

    return {
        "estado": "completado",
        "mensaje": f"Transformación exitosa: {nombre_salida}",
        "errores": 0,
        "filas_transformadas": len(df_limpio)
    }