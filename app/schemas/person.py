from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class PersonaVetadaBase(BaseModel):
    cedula: str = Field(..., min_length=3, max_length=50, description="CI / Documento de identidad")
    nombre: str = Field(..., min_length=1, max_length=100, description="Nombre de la persona")
    apellido: str = Field(default="", max_length=100, description="Apellido de la persona")
    motivo_veto: str = Field(..., min_length=3, max_length=500, description="Motivo de prohibición de ingreso")
    nivel_riesgo: str = Field(default="ALTO", description="BAJO, MEDIO, ALTO o CRITICO")


class PersonaVetadaUpdate(BaseModel):
    nombre: Optional[str] = Field(default=None, min_length=1, max_length=100)
    apellido: Optional[str] = Field(default=None, max_length=100)
    motivo_veto: Optional[str] = Field(default=None, min_length=3, max_length=500)
    nivel_riesgo: Optional[str] = Field(default=None)
    activo: Optional[bool] = Field(default=None)


class PersonaVetadaResponse(PersonaVetadaBase):
    id: int
    foto_path: Optional[str] = None
    activo: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

