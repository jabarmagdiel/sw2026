import base64
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import HTTPException, status
from sqlalchemy.orm import Session, joinedload

from app.config import settings
from app.models.alert import Alerta
from app.schemas.alert import AlertaReviewRequest


class AlertService:
    @staticmethod
    def annotate_multiface_frame(
        frame: np.ndarray,
        faces_info: List[Dict[str, Any]],
        highlight_bbox: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Dibuja todas las cajas delimitadoras de múltiples rostros detectados simultáneamente en el frame.
        faces_info: lista de dicts con keys: bbox, is_match, label
        """
        annotated = frame.copy()
        for info in faces_info:
            bx1, by1, bx2, by2 = info["bbox"][:4]
            is_match = info.get("is_match", False)
            label = info.get("label", "ROSTRO")
            is_highlighted = highlight_bbox is not None and info["bbox"][:4] == highlight_bbox[:4]

            color = (0, 0, 255) if is_match else (80, 200, 30)
            thickness = 3 if (is_match or is_highlighted) else 2
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, thickness)

            # Etiqueta con fondo legible
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            ty1 = max(by1 - th - 10, 0)
            ty2 = ty1 + th + 8
            cv2.rectangle(annotated, (bx1, ty1), (bx1 + tw + 10, ty2), color, -1)
            cv2.putText(
                annotated,
                label,
                (bx1 + 5, ty2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2,
            )
        return annotated

    @staticmethod
    def encode_frame_preview_base64(frame: np.ndarray, max_width: int = 640) -> Optional[str]:
        """Codifica un fotograma anotado como Base64 JPEG ligero para líneas de tiempo de video."""
        try:
            h, w = frame.shape[:2]
            if w > max_width:
                scale = max_width / float(w)
                frame = cv2.resize(frame, (max_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                return None
            b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
            return f"data:image/jpeg;base64,{b64}"
        except Exception:
            return None

    @staticmethod
    def create_alert(
        db: Session,
        persona_vetada_id: int,
        camara_id: str,
        zona: str,
        similitud: float,
        bbox: List[int],
        frame: Optional[np.ndarray] = None,
        all_faces_info: Optional[List[Dict[str, Any]]] = None,
        cooldown_seconds: float = 0.0,
    ) -> Alerta:
        if cooldown_seconds > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(seconds=cooldown_seconds)
            recent = (
                db.query(Alerta)
                .filter(
                    Alerta.persona_vetada_id == persona_vetada_id,
                    Alerta.camara_id == camara_id,
                    Alerta.created_at >= cutoff,
                )
                .order_by(Alerta.created_at.desc())
                .first()
            )
            if recent is not None:
                if similitud > recent.similitud:
                    recent.similitud = round(float(similitud), 4)
                    db.commit()
                    db.refresh(recent)
                return recent

        screenshot_path = None
        if frame is not None:
            os.makedirs(settings.SCREENSHOTS_DIR, exist_ok=True)
            if all_faces_info:
                annotated = AlertService.annotate_multiface_frame(
                    frame, all_faces_info, highlight_bbox=bbox
                )
            else:
                annotated = frame.copy()
                x1, y1, x2, y2 = bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"VETADO ({similitud:.2f})"
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
            filename = f"alert_{persona_vetada_id}_{uuid.uuid4().hex[:8]}.jpg"
            screenshot_path = os.path.join(settings.SCREENSHOTS_DIR, filename).replace("\\", "/")
            cv2.imwrite(screenshot_path, annotated)

        bbox_str = ",".join(str(v) for v in bbox)
        alerta = Alerta(
            persona_vetada_id=persona_vetada_id,
            camara_id=camara_id,
            zona=zona,
            similitud=round(float(similitud), 4),
            bbox=bbox_str,
            screenshot_path=screenshot_path,
            revisada=False,
        )
        db.add(alerta)
        db.commit()
        db.refresh(alerta)
        return alerta

    @staticmethod
    def list_alerts(
        db: Session,
        skip: int = 0,
        limit: int = 50,
        camara_id: Optional[str] = None,
        revisada: Optional[bool] = None,
    ) -> List[Alerta]:
        query = db.query(Alerta).options(joinedload(Alerta.persona_vetada))
        if camara_id:
            query = query.filter(Alerta.camara_id == camara_id)
        if revisada is not None:
            query = query.filter(Alerta.revisada.is_(revisada))
        return query.order_by(Alerta.created_at.desc()).offset(skip).limit(limit).all()

    @staticmethod
    def get_alert(db: Session, alerta_id: int) -> Alerta:
        alerta = (
            db.query(Alerta)
            .options(joinedload(Alerta.persona_vetada))
            .filter(Alerta.id == alerta_id)
            .first()
        )
        if not alerta:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alerta con ID {alerta_id} no encontrada.",
            )
        return alerta

    @staticmethod
    def review_alert(db: Session, alerta_id: int, data: AlertaReviewRequest) -> Alerta:
        alerta = AlertService.get_alert(db, alerta_id)
        alerta.revisada = data.revisada
        if data.notas is not None:
            alerta.notas = data.notas
        db.commit()
        db.refresh(alerta)
        return alerta

