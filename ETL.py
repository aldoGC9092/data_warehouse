import pandas as pd
import numpy as np
import os

# --- ETAPA 1: RECOLECCIÓN (Definición de rutas) ---
# Nombres de los archivos de entrada (los datos "sucios")
ARCHIVOS_ENTRADA = {
    'amazon': 'Amazon Sale Report.csv',
    'intl': 'International sale Report.csv',
    'pl': 'P L March 2021.csv'
}

# Nombres para los archivos de salida (los datos "limpios")
ARCHIVOS_SALIDA = {
    'amazon': 'amazon_sales_limpio.csv',
    'intl': 'international_sales_limpio.csv',
    'pl': 'inventario_costos_limpio.csv'
}

# --- Funciones Auxiliares ---
def separador(titulo):
    """Imprime un separador bonito en la consola."""
    print("\n" + "="*80)
    print(f"--- {titulo} ---")
    print("="*80 + "\n")

# --- FASE DE EXTRACCIÓN (E) ---
def extraer_datos(nombre_archivo):
    """
    Función modular para extraer datos de un archivo CSV.
    (E)xtracción del ETL.
    """
    ruta = os.path.join(os.getcwd(), 'data', nombre_archivo)
    try:
        # Usamos low_memory=False para el warning de Amazon, no afecta a los otros
        df = pd.read_csv(ruta, low_memory=False)
        print(f"[OK] Archivo '{nombre_archivo}' cargado exitosamente.")
        return df
    except FileNotFoundError:
        print(f"[ERROR] ¡Archivo no encontrado! -> {nombre_archivo}")
        print("Asegúrate de que esté en la misma carpeta que el script.")
        return None
    except Exception as e:
        print(f"[ERROR] Inesperado al leer '{nombre_archivo}': {e}")
        return None

# --- FASE DE TRANSFORMACIÓN (T) ---

def transformar_amazon(df):
    """
    Aplica todas las correcciones del EDA al DataFrame de Amazon.
    """
    if df is None:
        return None
        
    print("  -> Iniciando transformación de 'Amazon Sale Report'...")
    
    # Hallazgo 1 (EDA): Eliminar columna 'Unnamed: 22'
    if 'Unnamed: 22' in df.columns:
        df = df.drop('Unnamed: 22', axis=1)
        print("     - Columna 'Unnamed: 22' eliminada.")
    
    # Hallazgo 2 (EDA): Convertir 'Date' a formato datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%m-%d-%y')
    print("     - Columna 'Date' convertida a datetime.")
    
    # Hallazgo 3 (EDA): Convertir 'ship-postal-code' de float a string
    df['ship-postal-code'] = df['ship-postal-code'].fillna(0).astype(int).astype(str)
    print("     - Columna 'ship-postal-code' convertida a string.")
    
    # Hallazgo 4 (EDA): Rellenar 'Amount' nulos (asociados a 'Qty' 0)
    df['Amount'] = df['Amount'].fillna(0.0)
    print("     - Nulos en 'Amount' rellenados con 0.0.")
    
    # Hallazgo 5 (EDA): Rellenar 'promotion-ids' nulos
    df['promotion-ids'] = df['promotion-ids'].fillna('Sin Promocion')
    print("     - Nulos en 'promotion-ids' rellenados.")

    # Limpieza Extra: Rellenar nulos en 'fulfilled-by' y 'Courier Status'
    df['fulfilled-by'] = df['fulfilled-by'].fillna('Desconocido')
    df['Courier Status'] = df['Courier Status'].fillna('Desconocido')
        
    print("  -> [OK] Transformación de 'Amazon Sale Report' completada.")
    return df

def transformar_intl(df):
    """
    Aplica todas las correcciones del EDA al DataFrame Internacional.
    """
    if df is None:
        return None
        
    print("  -> Iniciando transformación de 'International sale Report'...")
    
    # Hallazgo 1 (EDA): Convertir columnas 'object' (texto) a numéricas
    # Usamos errors='coerce' para convertir datos "sucios" (texto) en NaN (Nulo)
    df['GROSS AMT'] = pd.to_numeric(df['GROSS AMT'], errors='coerce')
    df['RATE'] = pd.to_numeric(df['RATE'], errors='coerce')
    df['PCS'] = pd.to_numeric(df['PCS'], errors='coerce')
    print("     - Columnas 'GROSS AMT', 'RATE', 'PCS' convertidas a numérico.")
    
    # Ahora rellenamos los nulos que 'coerce' pudo haber creado
    df['GROSS AMT'] = df['GROSS AMT'].fillna(0.0)
    df['RATE'] = df['RATE'].fillna(0.0)
    df['PCS'] = df['PCS'].fillna(0) # Piezas como entero
    
    # Hallazgo 2 (EDA): Convertir 'DATE' a formato datetime
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce') # 'coerce' por si hay fechas malas
    print("     - Columna 'DATE' convertida a datetime.")
    
    # Hallazgo 3 (EDA): Eliminar columna redundante 'Months'
    if 'Months' in df.columns:
        df = df.drop('Months', axis=1)
        print("     - Columna 'Months' eliminada.")
        
    # Hallazgo 4 (EDA): Rellenar nulos en columnas de texto
    df['CUSTOMER'] = df['CUSTOMER'].fillna('Desconocido')
    df['Style'] = df['Style'].fillna('Desconocido')
    df['SKU'] = df['SKU'].fillna('Desconocido')
    print("     - Nulos en 'CUSTOMER', 'Style' y 'SKU' rellenados.")

    print("  -> [OK] Transformación de 'International sale Report' completada.")
    return df

def transformar_pl(df):
    """
    Aplica todas las correcciones del EDA al DataFrame de Inventario (P&L).
    """
    if df is None:
        return None
        
    print("  -> Iniciando transformación de 'P L March 2021'...")

    # Hallazgo 1 (EDA): Reemplazar el texto 'Nill' por un valor nulo estándar (np.nan)
    # Esto es CRUCIAL antes de intentar convertir a número
    print(f"     - Se encontraron {df[df == 'Nill'].count().sum()} valores 'Nill'.")
    df = df.replace('Nill', np.nan)
    print("     - Valores 'Nill' reemplazados por Nulos (NaN).")
    
    # Hallazgo 2 (EDA): Convertir todas las columnas de precio y peso a numérico
    columnas_a_convertir = [
        'Weight', 'TP 1', 'TP 2', 'MRP Old', 'Final MRP Old', 'Ajio MRP', 
        'Amazon MRP', 'Amazon FBA MRP', 'Flipkart MRP', 'Limeroad MRP', 
        'Myntra MRP', 'Paytm MRP', 'Snapdeal MRP'
    ]
    
    for col in columnas_a_convertir:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    print(f"     - {len(columnas_a_convertir)} columnas de precio/peso convertidas a numérico.")
    
    # Rellenamos los nulos (originales o creados por 'coerce')
    df[columnas_a_convertir] = df[columnas_a_convertir].fillna(0.0)
    
    # Hallazgo 3 (EDA): Rellenar nulos en 'Catalog' y 'Category'
    df['Catalog'] = df['Catalog'].fillna('Desconocido')
    df['Category'] = df['Category'].fillna('Desconocido')
    print("     - Nulos en 'Catalog' y 'Category' rellenados.")
    
    print("  -> [OK] Transformación de 'P L March 2021' completada.")
    return df

# --- FASE DE CARGA (L) ---
def cargar_datos_limpios(df, nombre_archivo_limpio):
    """
    Guarda el DataFrame limpio en un nuevo archivo CSV (Etapa 5).
    (L)oad del ETL.
    """
    if df is None:
        print(f"[ERROR] No se pudo guardar '{nombre_archivo_limpio}' porque el DataFrame está vacío (None).")
        return
    
    directorio_salida = os.path.join(os.getcwd(), 'clean_data')
    # Crea la carpeta si no existe
    os.makedirs(directorio_salida, exist_ok=True)
        
    try:
        ruta_salida = os.path.join(directorio_salida, nombre_archivo_limpio)
        # index=False evita que se guarde el índice de pandas como una columna
        # encoding='utf-8-sig' es robusto para compatibilidad con Excel
        df.to_csv(ruta_salida, index=False, encoding='utf-8-sig')
        print(f"[OK] ¡Datos limpios guardados exitosamente en '{nombre_archivo_limpio}'!")
    except Exception as e:
        print(f"[ERROR] al guardar el archivo '{nombre_archivo_limpio}': {e}")

# --- Orquestador Principal del Pipeline ---
def main():
    """
    Función principal que orquesta todo el proceso ETL.
    Ejecuta las fases E, T y L de forma modular.
    """
    separador("INICIO DEL PIPELINE ETL (ETAPA 3 Y 5)")
    
    # --- ETAPA 3: Extracción ---
    separador("FASE 1: (E) EXTRACCIÓN")
    df_amazon_sucio = extraer_datos(ARCHIVOS_ENTRADA['amazon'])
    df_intl_sucio = extraer_datos(ARCHIVOS_ENTRADA['intl'])
    df_pl_sucio = extraer_datos(ARCHIVOS_ENTRADA['pl'])
    
    # --- ETAPA 3: Transformación ---
    separador("FASE 2: (T) TRANSFORMACIÓN")
    df_amazon_limpio = transformar_amazon(df_amazon_sucio)
    df_intl_limpio = transformar_intl(df_intl_sucio)
    df_pl_limpio = transformar_pl(df_pl_sucio)
    
    # --- ETAPA 5: Carga (Guardado) ---
    separador("FASE 3: (L) CARGA / GUARDADO")
    cargar_datos_limpios(df_amazon_limpio, ARCHIVOS_SALIDA['amazon'])
    cargar_datos_limpios(df_intl_limpio, ARCHIVOS_SALIDA['intl'])
    cargar_datos_limpios(df_pl_limpio, ARCHIVOS_SALIDA['pl'])
    
    separador("PIPELINE ETL COMPLETADO")
    print("Se han generado 3 nuevos archivos CSV con los datos limpios en tu carpeta.")

# solo se ejecute cuando corres el script directamente.
if __name__ == "__main__":
    main()