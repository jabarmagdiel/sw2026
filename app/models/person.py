from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, Integer, LargeBinary, String
from sqlalchemy.orm import relationship

from app.db.database import Base


class PersonaVetada(Base):
    __tablename__ = "personas_vetadas"

    id = Column(Integer, primary_key=True, index=True)
    cedula = Column(String(50), unique=True, index=True, nullable=False)
    nombre = Column(String(100), nullable=False)
    apellido = Column(String(100), nullable=False, default="")
    motivo_veto = Column(String(500), nullable=False)
    nivel_riesgo = Column(String(30), nullable=False, default="ALTO")  # BAJO, MEDIO, ALTO, CRITICO
    foto_path = Column(String(500), nullable=True)
    embedding = Column(LargeBinary, nullable=False)  # Vector float32 de 512 dims (2048 bytes)
    activo = Column(Boolean, default=True, nullable=False, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    alertas = relationship("Alerta", back_populates="persona_vetada", cascade="all, delete-orphan")

