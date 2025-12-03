import uvicorn, os

from uuid import uuid4
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from etl_adapter import ejecutar_etl_desde_api
from etl import cargar_datos_limpios
from ml import train_and_compare_models

etl_registry = {}

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def formularioArchivo(request: Request):
    # convert registry into a list for rendering
    registros = [
        {
        "id": k,
        "filename": v.get("filename"),
        "status": v.get("status", "unknown"),
        }
        for k, v in etl_registry.items()
    ]

    return templates.TemplateResponse("formulario.html", {"request": request, "registros": registros})


@app.post("/api/subir")
async def subeArchivo(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos CSV")
    
    UPLOAD_DIR = "data"
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    id_proceso = str(uuid4())
    ruta = os.path.join(os.getcwd(), UPLOAD_DIR, file.filename)
    content = await file.read()
    with open(ruta, "wb") as f:        
        f.write(content)

    # resultado = ejecutar_etl_desde_api(ruta)
    # resultado["id"] = id_proceso
    etl_registry[id_proceso] = {
        "id": id_proceso,
        "filename": file.filename,
        "uploaded_path": str(ruta),
        "status": "subido",
        "mensaje": "Archivo subido, pendiente de ejecutar ETL",
        "filas_transformadas": None,
        "errores": None,
        "etl_result": None,
        "clean_data_path": None
    }

    return RedirectResponse(url=f"/api/estado/{id_proceso}", status_code=303)


@app.post("/api/run_etl/{id_proceso}")
async def run_etl(id_proceso: str):
    registro = etl_registry.get(id_proceso)

    if not registro:
        raise HTTPException(status_code=404, detail="ID de proceso no encontrado")
    
    if registro.get("status") == "guardado" or registro.get("status") == "etl terminado":
        return JSONResponse({"status": "etl ya fue procesado"})
    
    if registro.get("status") not in "subido":
        return JSONResponse({"status": "error con el archivo subido"})
    
    if registro.get("status") == "ejecutando":
        return JSONResponse({"status": "ya esta siendo ejecutado el etl"})   
    
    
    uploaded_path = registro.get("uploaded_path")
    if not uploaded_path or not os.path.exists(uploaded_path):
        raise HTTPException(status_code=400, detail="Archivo no encontrado")
    
    registro["status"] = "ejecutando"
    registro["mensaje"] = "ETL en progreso"

    try:
        resultado = ejecutar_etl_desde_api(uploaded_path)
    except Exception as e:
        registro["status"] = "etl fallado"
        registro["mensaje"] = f"ETL falló: {e}"
        registro["errores"] = str(e)
        registro["etl_result"] = None
        return JSONResponse({"status": "failed", "error": str(e)}, status_code=500)
    
    registro["etl_result"] = resultado
    registro["filas_transformadas"] = resultado.get("filas_transformadas")
    registro["errores"] = resultado.get("errores")
    registro["mensaje"] = resultado.get("mensaje", "ETL finalizado")
    registro["status"] = "etl terminado"

    return RedirectResponse(url=f"/api/estado/{id_proceso}", status_code=303)


@app.post("/api/save_cleaned/{id_proceso}")
async def save_cleaned(id_proceso: str):
    registro = etl_registry.get(id_proceso)
        
    if not registro:
        raise HTTPException(status_code=404, detail="ID de proceso no encontrado")
    else: resultado_etl = registro.get("etl_result", {})

    # Si no se ha ejecutado
    if registro.get("status") not in "etl terminado":
        raise HTTPException(status_code=400, detail="ETL no ejecutado todavía")
    
    if registro.get("status") == "guardado":
        raise HTTPException(status_code=400, detail="Ya fue guardado")
    
    try:
        ruta_salida = cargar_datos_limpios(resultado_etl.get("datos_limpios"), resultado_etl.get("salida"))
        registro["clean_data_path"] = ruta_salida
    except Exception as e:
        registro["status"] = "fallado"
        registro["mensaje"] = f"Guardar etl falló: {e}"
        registro["errores"] = str(e)
        registro["etl_result"] = None
        return JSONResponse({"status": "failed", "error": str(e)}, status_code=500)
    
    registro["mensaje"] = "Transformacion guardada exitosamente"
    registro["status"] = "guardado"

    return RedirectResponse(url=f"/api/estado/{id_proceso}", status_code=303)


@app.get("/api/estado/{id_proceso}")
def estado_etl(request: Request, id_proceso: str):
    estado = etl_registry.get(id_proceso)
    if not estado:
        raise HTTPException(status_code=404, detail="ID de proceso no encontrado")
    
    return templates.TemplateResponse("resultado.html", {
        "request": request,
        "id": id_proceso,
        "mensaje": estado.get("mensaje"),
        "filas": estado.get("filas_transformadas"),
        "errores": estado.get("errores")
    })


# Nuevo endpoint para entrenar modelos para un proceso existente
@app.post("/api/train/{id_proceso}")
async def train_models(id_proceso: str, target_col: str, test_size: float = 0.2):
    registro = etl_registry.get(id_proceso)
    if not registro:
        return JSONResponse({"status": "error", "detail": "Proceso no encontrado"}, status_code=404)

    # verificar que ETL ya terminó y tenemos ruta de salida
    if registro.get("status") != "guardado":
        return JSONResponse({"status": "error", "detail": "ETL no ejecutado o incompleto"}, status_code=400)

    ruta_salida = registro.get("clean_data_path")
    if not ruta_salida or not os.path.exists(ruta_salida):
        return JSONResponse({"status": "error", "detail": "Archivo transformado no encontrado"}, status_code=500)

    # Ejecutar entrenamiento
    try:
        ml_results = train_and_compare_models(
            ruta_salida, 
            target_col=target_col, 
            test_size=test_size, 
            id_proceso=id_proceso)
    except Exception as e:
        registro["status"] = "ml_failed"
        registro["ml_error"] = str(e)
        return JSONResponse({"status": "error", "detail": f"Training failed: {e}"}, status_code=500)

    # Guardar resultados en el registro
    registro["ml_results"] = ml_results
    registro["status"] = "ml_completed"
    return JSONResponse({"status": "ok", "ml_results": ml_results})


# Endpoint para obtener resumen JSON del ML
@app.get("/api/ml_results/{id_proceso}")
async def get_ml_results(id_proceso: str):
    registro = etl_registry.get(id_proceso)
    if not registro:
        return JSONResponse({"status": "error", "detail": "Proceso no encontrado"}, status_code=404)
    ml_results = registro.get("ml_results")
    if not ml_results:
        return JSONResponse({"status": "error", "detail": "Resultados ML no disponibles"}, status_code=404)
    return JSONResponse({"status": "ok", "ml_results": ml_results})


@app.get("/ml/view/{id_proceso}")
async def view_ml_results(request: Request, id_proceso: str):
    registro = etl_registry.get(id_proceso)
    if not registro or "ml_results" not in registro:
        return templates.TemplateResponse("ml_results.html", {"request": request, "id_proceso": id_proceso, "metrics": {}, "winner": "no disponible"})
    ml = registro["ml_results"]
    return templates.TemplateResponse("ml_results.html", {"request": request, "id_proceso": id_proceso, "metrics": ml["metrics"], "winner": ml["winner"]})

if __name__ == '__main__':
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
