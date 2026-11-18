from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.schemas.person import PersonaVetadaResponse, PersonaVetadaUpdate
from app.services.blacklist_service import BlacklistService
from app.services.face_engine import face_engine

router = APIRouter(prefix="/api/v1/vetados", tags=["Lista de Vetados (Blacklist)"])


@router.post("", response_model=PersonaVetadaResponse, status_code=status.HTTP_201_CREATED)
async def registrar_vetado(
    cedula: str = Form(..., description="Cédula de identidad / Documento"),
    nombre: str = Form(..., description="Nombre"),
    apellido: str = Form(default="", description="Apellido"),
    motivo_veto: str = Form(..., description="Motivo de prohibición de ingreso al estadio"),
    nivel_riesgo: str = Form(default="ALTO", description="BAJO, MEDIO, ALTO o CRITICO"),
    foto: UploadFile = File(..., description="Fotografía clara del rostro"),
    db: Session = Depends(get_db),
):
    """
    Registra una persona vetada con su fotografía, extrae su vector biométrico (ArcFace 512-d)
    y lo indexa inmediatamente en FAISS.
    """
    content = await foto.read()
    try:
        image = face_engine.decode_image_bytes(content)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    return BlacklistService.create_vetado(
        db=db,
        cedula=cedula,
        nombre=nombre,
        apellido=apellido,
        motivo_veto=motivo_veto,
        nivel_riesgo=nivel_riesgo,
        image=image,
    )


@router.get("", response_model=List[PersonaVetadaResponse])
def listar_vetados(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    solo_activos: bool = Query(True, description="Filtrar únicamente personas con veto activo"),
    db: Session = Depends(get_db),
):
    """Lista las personas registradas en la base de datos de vetados."""
    return BlacklistService.list_vetados(db=db, skip=skip, limit=limit, solo_activos=solo_activos)


@router.get("/{persona_id}", response_model=PersonaVetadaResponse)
def obtener_vetado(persona_id: int, db: Session = Depends(get_db)):
    """Obtiene el detalle de una persona vetada por su ID."""
    return BlacklistService.get_vetado(db=db, persona_id=persona_id)


@router.put("/{persona_id}", response_model=PersonaVetadaResponse)
def actualizar_vetado(
    persona_id: int,
    payload: PersonaVetadaUpdate,
    db: Session = Depends(get_db),
):
    """Actualiza datos de una persona vetada y sincroniza el índice FAISS si cambia su estado."""
    return BlacklistService.update_vetado(db=db, persona_id=persona_id, data=payload)


@router.delete("/{persona_id}", response_model=PersonaVetadaResponse)
def desactivar_vetado(persona_id: int, db: Session = Depends(get_db)):
    """Desactiva (soft-delete) a una persona vetada y la retira del índice FAISS activo."""
    return BlacklistService.deactivate_vetado(db=db, persona_id=persona_id)

