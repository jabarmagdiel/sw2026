import os
import tempfile
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.config import settings
from app.models.alert import Alerta
from app.models.person import PersonaVetada
from app.schemas.recognition import (
    FaceMatchItem,
    FaceNoMatchItem,
    MatchedPersonInfo,
    RecognitionResponse,
    RecognizeBase64Request,
    VideoFrameDetection,
    VideoPersonSummary,
    VideoRecognitionResponse,
)
from app.services.alert_service import AlertService
from app.services.face_engine import face_engine
from app.services.ws_manager import ws_manager

router = APIRouter(prefix="/api/v1/recognize", tags=["Reconocimiento Facial en Accesos"])


def _format_timestamp(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = seconds - (mins * 60)
    return f"{mins:02d}:{secs:04.1f}"


def _process_frame(
    db: Session,
    image: np.ndarray,
    camara_id: str,
    zona: str,
    threshold: Optional[float] = None,
    include_crops: bool = True,
    cooldown_seconds: float = 0.0,
) -> RecognitionResponse:
    t0 = time.perf_counter()
    detected_faces = face_engine.detect_and_embed(image)

    matches: List[FaceMatchItem] = []
    no_matches: List[FaceNoMatchItem] = []

    if not detected_faces:
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return RecognitionResponse(
            camara_id=camara_id,
            zona=zona,
            faces_detected=0,
            alert_triggered=False,
            matches=[],
            no_matches=[],
            processing_time_ms=elapsed_ms,
        )

    # Búsqueda vectorial por lotes (batch) en FAISS para todos los rostros detectados simultáneamente
    embeddings = [dface.embedding for dface in detected_faces]
    batch_search_results = face_engine.search_batch(embeddings, threshold=threshold, top_k=5)

    # Pre-cargar personas candidatas activas desde la BD en una sola consulta
    candidate_ids = {
        cand_id
        for candidates, _ in batch_search_results
        for cand_id, _ in candidates
    }
    active_persons_map: Dict[int, PersonaVetada] = {}
    if candidate_ids:
        active_persons = (
            db.query(PersonaVetada)
            .filter(PersonaVetada.id.in_(candidate_ids), PersonaVetada.activo.is_(True))
            .all()
        )
        active_persons_map = {p.id: p for p in active_persons}

    # Resolver asignaciones rostro -> persona vetada priorizando mayor similitud cuando hay múltiples rostros
    face_assignments: Dict[int, tuple[PersonaVetada, float]] = {}
    used_person_ids: set[int] = set()

    # Ordenar índices de rostros por su mejor candidato válido para asignación óptima multirrostro
    scored_face_indices = []
    for idx, (candidates, _) in enumerate(batch_search_results):
        best_valid = next(
            ((cid, sc) for cid, sc in candidates if cid in active_persons_map),
            None,
        )
        scored_face_indices.append((idx, best_valid[1] if best_valid else -1.0))
    scored_face_indices.sort(key=lambda item: item[1], reverse=True)

    for idx, _ in scored_face_indices:
        candidates, _ = batch_search_results[idx]
        for cand_id, score in candidates:
            if cand_id in active_persons_map and cand_id not in used_person_ids:
                face_assignments[idx] = (active_persons_map[cand_id], score)
                used_person_ids.add(cand_id)
                break

    # Preparar metadatos de todos los rostros para anotar capturas multirrostro
    all_faces_info = []
    for idx, dface in enumerate(detected_faces):
        if idx in face_assignments:
            p, sc = face_assignments[idx]
            all_faces_info.append(
                {
                    "bbox": dface.bbox,
                    "is_match": True,
                    "label": f"VETADO #{idx + 1}: {p.nombre.upper()} ({sc * 100:.0f}%)",
                }
            )
        else:
            all_faces_info.append(
                {
                    "bbox": dface.bbox,
                    "is_match": False,
                    "label": f"AUTORIZADO #{idx + 1}",
                }
            )

    for idx, dface in enumerate(detected_faces):
        _, best_score = batch_search_results[idx]
        crop_b64 = face_engine.crop_face_base64(image, dface.bbox) if include_crops else None

        if idx in face_assignments:
            matched_person, matched_score = face_assignments[idx]
            alerta = AlertService.create_alert(
                db=db,
                persona_vetada_id=matched_person.id,
                camara_id=camara_id,
                zona=zona,
                similitud=matched_score,
                bbox=dface.bbox,
                frame=image,
                all_faces_info=all_faces_info,
                cooldown_seconds=cooldown_seconds,
            )
            matches.append(
                FaceMatchItem(
                    face_index=idx,
                    bbox=dface.bbox,
                    det_score=dface.det_score,
                    face_crop_base64=crop_b64,
                    match=MatchedPersonInfo(
                        persona_id=matched_person.id,
                        cedula=matched_person.cedula,
                        nombre=matched_person.nombre,
                        apellido=matched_person.apellido,
                        motivo_veto=matched_person.motivo_veto,
                        nivel_riesgo=matched_person.nivel_riesgo,
                        similitud=matched_score,
                        alerta_id=alerta.id,
                        foto_path=matched_person.foto_path,
                    ),
                )
            )
            ws_manager.notify_from_sync(
                {
                    "event": "ALERTA_PERSONA_VETADA",
                    "alerta_id": alerta.id,
                    "camara_id": camara_id,
                    "zona": zona,
                    "similitud": matched_score,
                    "bbox": dface.bbox,
                    "faces_in_frame": len(detected_faces),
                    "screenshot_path": alerta.screenshot_path,
                    "persona": {
                        "id": matched_person.id,
                        "cedula": matched_person.cedula,
                        "nombre": f"{matched_person.nombre} {matched_person.apellido}".strip(),
                        "motivo_veto": matched_person.motivo_veto,
                        "nivel_riesgo": matched_person.nivel_riesgo,
                    },
                }
            )
        else:
            no_matches.append(
                FaceNoMatchItem(
                    face_index=idx,
                    bbox=dface.bbox,
                    det_score=dface.det_score,
                    best_score=best_score,
                    face_crop_base64=crop_b64,
                )
            )

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return RecognitionResponse(
        camara_id=camara_id,
        zona=zona,
        faces_detected=len(detected_faces),
        alert_triggered=len(matches) > 0,
        matches=matches,
        no_matches=no_matches,
        processing_time_ms=elapsed_ms,
    )


@router.post("", response_model=RecognitionResponse)
async def reconocer_frame_multipart(
    file: UploadFile = File(..., description="Frame/imagen capturada por la cámara de acceso"),
    camara_id: str = Form(default="CAM-ACCESO-01", description="ID de la cámara"),
    zona: str = Form(default="Entrada Principal", description="Zona de acceso controlada"),
    threshold: Optional[float] = Form(default=None, description="Umbral opcional de similitud (ej. 0.45)"),
    db: Session = Depends(get_db),
):
    """
    Analiza una imagen enviada por formulario `multipart/form-data`, detecta todos los rostros en simultáneo,
    compara sus embeddings en lote contra el índice FAISS de vetados y genera alertas si hay coincidencia.
    """
    content = await file.read()
    try:
        image = face_engine.decode_image_bytes(content)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    return _process_frame(db=db, image=image, camara_id=camara_id, zona=zona, threshold=threshold)


@router.post("/base64", response_model=RecognitionResponse)
def reconocer_frame_base64(
    payload: RecognizeBase64Request,
    db: Session = Depends(get_db),
):
    """
    Analiza un frame enviado en formato JSON Base64 (cámara en vivo o video en reproducción),
    detectando y verificando múltiples rostros a la vez.
    """
    try:
        image = face_engine.decode_base64_image(payload.image_base64)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Imagen Base64 inválida: {exc}",
        )

    return _process_frame(
        db=db,
        image=image,
        camara_id=payload.camara_id,
        zona=payload.zona,
        threshold=payload.threshold,
        cooldown_seconds=settings.ALERT_COOLDOWN_SECONDS,
    )


@router.post("/video", response_model=VideoRecognitionResponse)
async def reconocer_video_multipart(
    file: UploadFile = File(..., description="Archivo de video (MP4, AVI, MOV, WEBM, MKV) para análisis multirrostro"),
    camara_id: str = Form(default="CAM-VIDEO-FORENSE", description="ID de la cámara o fuente de video"),
    zona: str = Form(default="Análisis de Video Grabado", description="Zona o referencia del video"),
    sample_every_n_frames: int = Form(
        default=settings.VIDEO_SAMPLE_EVERY_N_FRAMES,
        ge=1,
        le=120,
        description="Analizar 1 de cada N fotogramas del video",
    ),
    max_frames: int = Form(
        default=settings.MAX_VIDEO_FRAMES_TO_ANALYZE,
        ge=1,
        le=600,
        description="Máximo de fotogramas a analizar en el video",
    ),
    threshold: Optional[float] = Form(default=None, description="Umbral opcional de similitud (ej. 0.45)"),
    db: Session = Depends(get_db),
):
    """
    Procesa un archivo de video completo fotograma a fotograma (según `sample_every_n_frames`),
    detectando múltiples rostros simultáneos en cada fotograma y cruzándolos en lote contra FAISS.
    Devuelve la línea de tiempo con detecciones, rostros simultáneos y resumen de personas vetadas identificadas.
    """
    t0 = time.perf_counter()
    content = await file.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="El archivo de video está vacío.")

    suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No se pudo abrir o decodificar el archivo de video proporcionado.",
            )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0 or fps > 240:
            fps = 25.0

        frames_analyzed = 0
        total_faces_detected = 0
        max_simultaneous_faces = 0
        timeline: List[VideoFrameDetection] = []
        vetados_summary_map: Dict[int, VideoPersonSummary] = {}

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx % sample_every_n_frames == 0:
                frames_analyzed += 1
                timestamp_sec = round(frame_idx / fps, 2)
                timestamp_label = _format_timestamp(timestamp_sec)

                frame_res = _process_frame(
                    db=db,
                    image=frame,
                    camara_id=camara_id,
                    zona=f"{zona} (t={timestamp_label})",
                    threshold=threshold,
                    include_crops=True,
                    cooldown_seconds=settings.ALERT_COOLDOWN_SECONDS,
                )

                n_faces = frame_res.faces_detected
                total_faces_detected += n_faces
                if n_faces > max_simultaneous_faces:
                    max_simultaneous_faces = n_faces

                # Actualizar resumen acumulado de personas vetadas únicas en el video
                for m_item in frame_res.matches:
                    m = m_item.match
                    pid = m.persona_id
                    alerta_obj = db.query(Alerta).filter(Alerta.id == m.alerta_id).first()
                    scr_path = alerta_obj.screenshot_path if alerta_obj else None

                    if pid not in vetados_summary_map:
                        vetados_summary_map[pid] = VideoPersonSummary(
                            persona_id=pid,
                            cedula=m.cedula,
                            nombre=m.nombre,
                            apellido=m.apellido,
                            motivo_veto=m.motivo_veto,
                            nivel_riesgo=m.nivel_riesgo,
                            max_similitud=m.similitud,
                            apariciones=1,
                            first_seen_sec=timestamp_sec,
                            last_seen_sec=timestamp_sec,
                            timestamps=[timestamp_label],
                            alerta_id=m.alerta_id,
                            screenshot_path=scr_path,
                            foto_path=m.foto_path,
                            face_crop_base64=m_item.face_crop_base64,
                        )
                    else:
                        summary = vetados_summary_map[pid]
                        summary.apariciones += 1
                        summary.last_seen_sec = timestamp_sec
                        if timestamp_label not in summary.timestamps:
                            summary.timestamps.append(timestamp_label)
                        if m.similitud > summary.max_similitud:
                            summary.max_similitud = m.similitud
                            summary.alerta_id = m.alerta_id
                            if scr_path:
                                summary.screenshot_path = scr_path
                            if m_item.face_crop_base64:
                                summary.face_crop_base64 = m_item.face_crop_base64

                # Si se detectaron rostros en este fotograma, guardar evento en la línea de tiempo (hasta 40 eventos)
                if n_faces > 0 and len(timeline) < 40:
                    faces_info_for_preview = []
                    for m_item in frame_res.matches:
                        faces_info_for_preview.append(
                            {
                                "bbox": m_item.bbox,
                                "is_match": True,
                                "label": f"VETADO: {m_item.match.nombre.upper()} ({m_item.match.similitud * 100:.0f}%)",
                            }
                        )
                    for nm_item in frame_res.no_matches:
                        faces_info_for_preview.append(
                            {
                                "bbox": nm_item.bbox,
                                "is_match": False,
                                "label": f"AUTORIZADO #{nm_item.face_index + 1}",
                            }
                        )
                    annotated_frame = AlertService.annotate_multiface_frame(frame, faces_info_for_preview)
                    preview_b64 = AlertService.encode_frame_preview_base64(annotated_frame, max_width=640)

                    timeline.append(
                        VideoFrameDetection(
                            frame_number=frame_idx,
                            timestamp_sec=timestamp_sec,
                            timestamp_label=timestamp_label,
                            faces_detected=n_faces,
                            alert_triggered=frame_res.alert_triggered,
                            matches=frame_res.matches,
                            no_matches=frame_res.no_matches,
                            frame_preview_base64=preview_b64,
                        )
                    )

                if frames_analyzed >= max_frames:
                    break

            frame_idx += 1

        cap.release()
        actual_total_frames = max(total_frames, frame_idx)
        duration_seconds = round(actual_total_frames / fps, 2) if fps > 0 else 0.0
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        unique_vetados = sorted(
            vetados_summary_map.values(),
            key=lambda x: x.max_similitud,
            reverse=True,
        )

        return VideoRecognitionResponse(
            video_filename=file.filename or "video.mp4",
            camara_id=camara_id,
            zona=zona,
            total_frames=actual_total_frames,
            fps=round(fps, 2),
            duration_seconds=duration_seconds,
            frames_analyzed=frames_analyzed,
            sample_every_n_frames=sample_every_n_frames,
            total_faces_detected=total_faces_detected,
            max_simultaneous_faces=max_simultaneous_faces,
            alert_triggered=len(unique_vetados) > 0,
            unique_vetados_detected=unique_vetados,
            timeline=timeline,
            processing_time_ms=elapsed_ms,
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
