from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from app.db.database import Base


class Alerta(Base):
    __tablename__ = "alertas"

    id = Column(Integer, primary_key=True, index=True)
    persona_vetada_id = Column(Integer, ForeignKey("personas_vetadas.id"), nullable=False, index=True)
    camara_id = Column(String(100), nullable=False, default="CAM-ACCESO-01", index=True)
    zona = Column(String(100), nullable=False, default="Entrada Principal")
    similitud = Column(Float, nullable=False)
    bbox = Column(String(100), nullable=True)  # Guardado como "x1,y1,x2,y2"
    screenshot_path = Column(String(500), nullable=True)
    revisada = Column(Boolean, default=False, nullable=False, index=True)
    notas = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)

    persona_vetada = relationship("PersonaVetada", back_populates="alertas")

