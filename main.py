import uvicorn, os

from uuid import uuid4
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from etl_adapter import ejecutar_etl_desde_api
from etl import cargar_datos_limpios

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
        cargar_datos_limpios(resultado_etl.get("datos_limpios"), resultado_etl.get("salida"))
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



if __name__ == '__main__':
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
