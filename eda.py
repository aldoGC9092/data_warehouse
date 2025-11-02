import pandas as pd
import numpy as np
import os

archivo_amazon = os.path.join(os.getcwd(), 'data', 'Amazon Sale Report.csv') 
archivo_intl = os.path.join(os.getcwd(), 'data', 'International sale Report.csv')  
archivo_pl = os.path.join(os.getcwd(), 'data', 'P_L_March 2021.csv') 

def separador(titulo):
    print("\n" + "="*80)
    print(f"--- {titulo} ---")
    print("="*80 + "\n")

# --- Función de Análisis de Outliers (Valores Atípicos) ---
def analizar_outliers_iqr(df, columna):
    if not pd.api.types.is_numeric_dtype(df[columna]):
        print(f"-> HALLAZGO: La columna '{columna}' es de tipo '{df[columna].dtype}' (texto).")
        print("   No se pueden calcular outliers numéricos. Esto requiere limpieza (Etapa 3).")
        return
        
    print(f"\n--- Análisis de Outliers (IQR) para '{columna}' ---")
    
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1
    
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    
    outliers = df[(df[columna] < limite_inferior) | (df[columna] > limite_superior)]
    
    print(f"Medidas de dispersión (IQR):")
    print(f"  Q1 (Percentil 25): {Q1:.2f}")
    print(f"  Q3 (Percentil 75): {Q3:.2f}")
    print(f"  Rango Intercuartílico (IQR): {IQR:.2f}")
    print(f"Límite inferior (para outliers): {limite_inferior:.2f}")
    print(f"Límite superior (para outliers): {limite_superior:.2f}")
    
    if outliers.empty:
        print("\n-> HALLAZGO: No se encontraron outliers (valores atípicos) significativos.")
    else:
        print(f"\n-> HALLAZGO: Se encontraron {len(outliers)} outliers (valores atípicos).")
        print("   Mostrando 5 ejemplos de estos outliers:")
        print(outliers[[columna]].head())

# --- ETAPA 1: RECOLECCIÓN (Carga de Datos) ---
# Se verifica cada archivo individualmente para un mejor diagnóstico.

separador("ETAPA 1: RECOLECCIÓN DE DATOS")

# Carga de Archivo 1: Amazon
try:
    df_amazon = pd.read_csv(archivo_amazon, low_memory=False) 
    print(f"Archivo '{archivo_amazon}' cargado exitosamente ({len(df_amazon)} filas).")
except FileNotFoundError:
    print(f"\n¡ERROR FATAL! No se encontró el archivo '{archivo_amazon}'.")
    print("Asegúrate de que esté en la misma carpeta que este script .py")
    exit()

# Carga de Archivo 2: International Sales
try:
    df_intl = pd.read_csv(archivo_intl)
    print(f"Archivo '{archivo_intl}' cargado exitosamente ({len(df_intl)} filas).")
except FileNotFoundError:
    print(f"\n¡ERROR FATAL! No se encontró el archivo '{archivo_intl}'.")
    print("Asegúrate de que esté en la misma carpeta que este script .py")
    exit()

# Carga de Archivo 3: P&L March
try:
    df_pl = pd.read_csv(archivo_pl)
    print(f"Archivo '{archivo_pl}' cargado exitosamente ({len(df_pl)} filas).")
except FileNotFoundError:
    print(f"\n¡ERROR FATAL! No se encontró el archivo '{archivo_pl}'.")
    print("Asegúrate de que esté en la misma carpeta que este script .py")
    exit()

# --- ETAPA 2: ANÁLISIS EXPLORATORIO DE DATOS (EDA) ---

separador("INICIO DE ETAPA 2: ANÁLISIS EXPLORATORIO (EDA)")

# --- 1. ANÁLISIS: Amazon Sale Report.csv ---
separador("1. Análisis del Dataset: 'Amazon Sale Report'")

print("--- Información General (.info()) ---")
df_amazon.info()

print("\n--- Estadísticas Descriptivas (.describe()) ---")
print(df_amazon.describe(include='all'))

analizar_outliers_iqr(df_amazon, 'Amount')
analizar_outliers_iqr(df_amazon, 'Qty')

# --- 2. ANÁLISIS: International sale Report.csv ---
separador("2. Análisis del Dataset: 'International sale Report'")

print("--- Información General (.info()) ---")
df_intl.info()

print("\n--- Estadísticas Descriptivas (.describe()) ---")
print(df_intl.describe(include='all'))

analizar_outliers_iqr(df_intl, 'GROSS AMT') 

# --- 3. ANÁLISIS: P L March 2021.csv ---
separador("3. Análisis del Dataset: 'P L March 2021'")

print("--- Información General (.info()) ---")
df_pl.info()

print("\n--- Estadísticas Descriptivas (.describe()) ---")
print(df_pl.describe(include='all'))

analizar_outliers_iqr(df_pl, 'Weight')       
analizar_outliers_iqr(df_pl, 'TP 2')          
analizar_outliers_iqr(df_pl, 'Final MRP Old') 
analizar_outliers_iqr(df_pl, 'TP 1')          

separador("FIN DEL ANÁLISIS EXPLORATORIO")
