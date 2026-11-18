from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.schemas.alert import AlertaResponse, AlertaReviewRequest
from app.services.alert_service import AlertService

router = APIRouter(prefix="/api/v1/alertas", tags=["Alertas de Seguridad"])


@router.get("", response_model=List[AlertaResponse])
def listar_alertas(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    camara_id: Optional[str] = Query(None, description="Filtrar por ID de cámara"),
    revisada: Optional[bool] = Query(None, description="Filtrar por estado de revisión"),
    db: Session = Depends(get_db),
):
    """Lista las alertas generadas cuando se detecta una persona vetada en los accesos."""
    return AlertService.list_alerts(
        db=db, skip=skip, limit=limit, camara_id=camara_id, revisada=revisada
    )


@router.get("/{alerta_id}", response_model=AlertaResponse)
def obtener_alerta(alerta_id: int, db: Session = Depends(get_db)):
    """Obtiene el detalle de una alerta específica incluyendo los datos del vetado."""
    return AlertService.get_alert(db=db, alerta_id=alerta_id)


@router.patch("/{alerta_id}/revisar", response_model=AlertaResponse)
def marcar_alerta_revisada(
    alerta_id: int,
    payload: AlertaReviewRequest,
    db: Session = Depends(get_db),
):
    """Marca una alerta como revisada por el personal de seguridad y agrega notas."""
    return AlertService.review_alert(db=db, alerta_id=alerta_id, data=payload)

