from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.api.routes import alerts, blacklist, recognition
from app.config import settings
from app.db.init_db import init_db_and_index
from app.models.alert import Alerta
from app.models.person import PersonaVetada
from app.services.face_engine import face_engine
from app.services.ws_manager import ws_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db_and_index()
    yield


app = FastAPI(
    title="Módulo 3: Reconocimiento Facial — Control de Vigilancia en Estadios",
    description=(
        "Microservicio de visión artificial para detectar rostros en accesos controlados, "
        "generar embeddings biométricos (InsightFace/ArcFace + ONNX Runtime) y buscarlos en "
        "milisegundos con FAISS contra la lista de personas vetadas."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir capturas, fotos de referencia y recursos estáticos del frontend modular
app.mount("/data", StaticFiles(directory="data"), name="data")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(recognition.router)
app.include_router(blacklist.router)
app.include_router(alerts.router)


@app.get("/", include_in_schema=False)
def dashboard_ui():
    return FileResponse("app/static/index.html")


@app.get("/health", tags=["Sistema"])
def health_check():
    return {
        "status": "ok",
        "module": settings.APP_NAME,
        "engine_ready": face_engine.is_ready,
        "engine_backend": face_engine.backend_mode,
        "indexed_faces_in_faiss": face_engine.total_indexed,
        "similarity_threshold": settings.SIMILARITY_THRESHOLD,
    }


@app.get("/api/v1/stats", tags=["Sistema"])
def module_stats(db: Session = Depends(get_db)):
    total_vetados = db.query(PersonaVetada).count()
    vetados_activos = db.query(PersonaVetada).filter(PersonaVetada.activo.is_(True)).count()
    total_alertas = db.query(Alerta).count()
    alertas_pendientes = db.query(Alerta).filter(Alerta.revisada.is_(False)).count()

    return {
        "total_vetados": total_vetados,
        "vetados_activos": vetados_activos,
        "vectores_en_faiss": face_engine.total_indexed,
        "total_alertas": total_alertas,
        "alertas_pendientes": alertas_pendientes,
        "umbral_similitud": settings.SIMILARITY_THRESHOLD,
        "motor_inferencia": face_engine.backend_mode,
    }


@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """Canal WebSocket en tiempo real para notificar detecciones de personas vetadas al Dashboard."""
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


