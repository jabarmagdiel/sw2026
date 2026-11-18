from sqlalchemy.orm import Session

from app.db.database import Base, SessionLocal, engine
import app.models  # noqa: F401 - Registra modelos SQLAlchemy
from app.services.blacklist_service import BlacklistService
from app.services.face_engine import face_engine


def init_db_and_index() -> int:
    """Crea las tablas SQL si no existen, inicializa el motor facial y reconstruye el índice FAISS."""
    Base.metadata.create_all(bind=engine)
    face_engine.load_model()

    db: Session = SessionLocal()
    try:
        indexed_count = BlacklistService.sync_faiss_from_db(db)
        return indexed_count
    finally:
        db.close()

