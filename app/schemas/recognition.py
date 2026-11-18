from typing import List, Optional
from pydantic import BaseModel, Field


class RecognizeBase64Request(BaseModel):
    image_base64: str = Field(..., description="Imagen codificada en Base64 (con o sin prefijo data:image/...)")
    camara_id: str = Field(default="CAM-ACCESO-01", description="Identificador de la cámara de acceso")
    zona: str = Field(default="Entrada Principal", description="Zona de acceso del estadio")
    threshold: Optional[float] = Field(default=None, ge=0.1, le=1.0, description="Umbral opcional de similitud coseno")


class MatchedPersonInfo(BaseModel):
    persona_id: int
    cedula: str
    nombre: str
    apellido: str
    motivo_veto: str
    nivel_riesgo: str
    similitud: float
    alerta_id: int
    foto_path: Optional[str] = None


class FaceMatchItem(BaseModel):
    face_index: int
    bbox: List[int] = Field(..., description="[x1, y1, x2, y2]")
    det_score: float = Field(..., description="Confianza de detección del rostro")
    match: MatchedPersonInfo
    face_crop_base64: Optional[str] = Field(default=None, description="Recorte del rostro detectado en Base64 JPEG")


class FaceNoMatchItem(BaseModel):
    face_index: int
    bbox: List[int] = Field(..., description="[x1, y1, x2, y2]")
    det_score: float
    best_score: Optional[float] = Field(default=None, description="Mayor similitud encontrada (bajo el umbral)")
    face_crop_base64: Optional[str] = Field(default=None, description="Recorte del rostro detectado en Base64 JPEG")


class RecognitionResponse(BaseModel):
    camara_id: str
    zona: str
    faces_detected: int
    alert_triggered: bool
    matches: List[FaceMatchItem]
    no_matches: List[FaceNoMatchItem]
    processing_time_ms: float


class VideoPersonSummary(BaseModel):
    persona_id: int
    cedula: str
    nombre: str
    apellido: str
    motivo_veto: str
    nivel_riesgo: str
    max_similitud: float
    apariciones: int
    first_seen_sec: float
    last_seen_sec: float
    timestamps: List[str] = Field(default_factory=list, description="Marcas de tiempo MM:SS.d donde fue detectado")
    alerta_id: int
    screenshot_path: Optional[str] = None
    foto_path: Optional[str] = None
    face_crop_base64: Optional[str] = None


class VideoFrameDetection(BaseModel):
    frame_number: int
    timestamp_sec: float
    timestamp_label: str
    faces_detected: int
    alert_triggered: bool
    matches: List[FaceMatchItem]
    no_matches: List[FaceNoMatchItem]
    frame_preview_base64: Optional[str] = Field(default=None, description="Vista previa anotada del fotograma con todos los rostros")


class VideoRecognitionResponse(BaseModel):
    video_filename: str
    camara_id: str
    zona: str
    total_frames: int
    fps: float
    duration_seconds: float
    frames_analyzed: int
    sample_every_n_frames: int
    total_faces_detected: int
    max_simultaneous_faces: int
    alert_triggered: bool
    unique_vetados_detected: List[VideoPersonSummary]
    timeline: List[VideoFrameDetection]
    processing_time_ms: float

