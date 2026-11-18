import os
import uuid
from typing import List, Optional

import cv2
import numpy as np
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.config import settings
from app.models.person import PersonaVetada
from app.schemas.person import PersonaVetadaUpdate
from app.services.face_engine import face_engine


class BlacklistService:
    @staticmethod
    def sync_faiss_from_db(db: Session) -> int:
        """Sincroniza todos los vetados activos de la base de datos hacia el índice FAISS."""
        personas = db.query(PersonaVetada).filter(PersonaVetada.activo.is_(True)).all()
        items = [
            (p.id, face_engine.bytes_to_embedding(p.embedding))
            for p in personas
            if p.embedding is not None
        ]
        return face_engine.rebuild_index(items)

    @staticmethod
    def create_vetado(
        db: Session,
        cedula: str,
        nombre: str,
        apellido: str,
        motivo_veto: str,
        nivel_riesgo: str,
        image: np.ndarray,
    ) -> PersonaVetada:
        existing = db.query(PersonaVetada).filter(PersonaVetada.cedula == cedula).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Ya existe una persona registrada con la cédula/CI '{cedula}'.",
            )

        faces = face_engine.detect_and_embed(image)
        if not faces:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No se detectó ningún rostro claro en la fotografía enviada.",
            )

        # Se selecciona el rostro principal (mayor confianza)
        primary_face = faces[0]
        embedding_bytes = face_engine.embedding_to_bytes(primary_face.embedding)

        # Guardar imagen de referencia en disco
        os.makedirs(settings.VETADOS_DIR, exist_ok=True)
        safe_cedula = "".join(c for c in cedula if c.isalnum() or c in ("-", "_"))
        filename = f"{safe_cedula}_{uuid.uuid4().hex[:8]}.jpg"
        foto_path = os.path.join(settings.VETADOS_DIR, filename).replace("\\", "/")
        cv2.imwrite(foto_path, image)

        persona = PersonaVetada(
            cedula=cedula,
            nombre=nombre,
            apellido=apellido,
            motivo_veto=motivo_veto,
            nivel_riesgo=nivel_riesgo.upper(),
            foto_path=foto_path,
            embedding=embedding_bytes,
            activo=True,
        )
        db.add(persona)
        db.commit()
        db.refresh(persona)

        face_engine.add_to_index(persona.id, primary_face.embedding)
        return persona

    @staticmethod
    def list_vetados(
        db: Session, skip: int = 0, limit: int = 50, solo_activos: bool = True
    ) -> List[PersonaVetada]:
        query = db.query(PersonaVetada)
        if solo_activos:
            query = query.filter(PersonaVetada.activo.is_(True))
        return query.order_by(PersonaVetada.created_at.desc()).offset(skip).limit(limit).all()

    @staticmethod
    def get_vetado(db: Session, persona_id: int) -> PersonaVetada:
        persona = db.query(PersonaVetada).filter(PersonaVetada.id == persona_id).first()
        if not persona:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Persona vetada con ID {persona_id} no encontrada.",
            )
        return persona

    @staticmethod
    def update_vetado(db: Session, persona_id: int, data: PersonaVetadaUpdate) -> PersonaVetada:
        persona = BlacklistService.get_vetado(db, persona_id)
        update_data = data.model_dump(exclude_unset=True)

        for field, value in update_data.items():
            if field == "nivel_riesgo" and value is not None:
                value = value.upper()
            setattr(persona, field, value)

        db.commit()
        db.refresh(persona)

        # Actualizar índice FAISS si cambió su estado activo
        if "activo" in update_data:
            if persona.activo:
                emb = face_engine.bytes_to_embedding(persona.embedding)
                face_engine.add_to_index(persona.id, emb)
            else:
                face_engine.remove_from_index(persona.id)

        return persona

    @staticmethod
    def deactivate_vetado(db: Session, persona_id: int) -> PersonaVetada:
        persona = BlacklistService.get_vetado(db, persona_id)
        persona.activo = False
        db.commit()
        db.refresh(persona)
        face_engine.remove_from_index(persona.id)
        return persona

