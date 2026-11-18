"""
Script para pre-descargar los modelos ONNX de InsightFace (buffalo_l: RetinaFace + ArcFace)
antes de iniciar el servidor en producción.
"""
from app.config import settings
from app.services.face_engine import face_engine


def main() -> None:
    print(f"Descargando y preparando modelo InsightFace '{settings.MODEL_NAME}' en './models'...")
    face_engine.load_model()
    print(f"Motor listo. Modo activo: {face_engine.backend_mode}")


if __name__ == "__main__":
    main()

