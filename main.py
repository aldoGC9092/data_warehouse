import uvicorn, os

from uuid import uuid4
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from etl_adapter import ejecutar_etl_desde_api

etl_registry = {}

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def formularioArchivo(request: Request):
    formulario = './static/formulario.html'
    if not os.path.exists(formulario):
        return {"mensaje": "not found"}
    
    success = request.query_params.get("success")
    with open(formulario, "r", encoding="utf-8") as f:
        html_content = f.read()

    if success == "true":
        html_content = f"<p style='color:green;'>Archivo subido correctamente</p>\n" + html_content
    
    return HTMLResponse(content=html_content)


@app.post("/api/subir")
async def subeArchivo(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos CSV")
    
    id_proceso = str(uuid4())
    ruta = os.path.join(os.getcwd(), UPLOAD_DIR, file.filename)
    content = await file.read()
    with open(ruta, "wb") as f:        
        f.write(content)

    resultado = ejecutar_etl_desde_api(ruta)
    resultado["id"] = id_proceso
    etl_registry[id_proceso] = resultado

    return RedirectResponse(url=f"/api/estado/{id_proceso}", status_code=303)


@app.get("/api/estado/{process_id}")
def estado_etl(request: Request, process_id: str):
    estado = etl_registry.get(process_id)
    if not estado:
        raise HTTPException(status_code=404, detail="ID de proceso no encontrado")
    
    return templates.TemplateResponse("resultado.html", {
        "request": request,
        "id": process_id,
        "mensaje": estado.get("mensaje"),
        "filas": estado.get("filas_transformadas"),
        "errores": estado.get("errores")
    })



if __name__ == '__main__':
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
