from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field

from app.schemas.person import PersonaVetadaResponse


class AlertaReviewRequest(BaseModel):
    revisada: bool = Field(default=True, description="Estado de revisión de la alerta")
    notas: Optional[str] = Field(default=None, max_length=500, description="Observaciones del guardia/operador")


class AlertaResponse(BaseModel):
    id: int
    persona_vetada_id: int
    camara_id: str
    zona: str
    similitud: float
    bbox: Optional[str] = None
    screenshot_path: Optional[str] = None
    revisada: bool
    notas: Optional[str] = None
    created_at: datetime
    persona_vetada: Optional[PersonaVetadaResponse] = None

    model_config = ConfigDict(from_attributes=True)

