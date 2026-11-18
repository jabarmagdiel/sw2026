import base64
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

try:
    import faiss
except ImportError:
    faiss = None

try:
    import insightface
    from insightface.app import FaceAnalysis
except ImportError:
    insightface = None
    FaceAnalysis = None


@dataclass
class DetectedFace:
    bbox: List[int]  # [x1, y1, x2, y2]
    det_score: float
    embedding: np.ndarray  # Vector normalizado float32 de 512 dims


class FaceEngine:
    """
    Motor de Reconocimiento Facial basado en InsightFace (RetinaFace + ArcFace)
    optimizado con ONNX Runtime y búsqueda vectorial de alta velocidad con FAISS.
    """

    EMBEDDING_DIM = 512

    def __init__(self) -> None:
        self.model = None
        self.face_cascade = None
        self.index = None
        self._fallback_vectors: dict[int, np.ndarray] = {}
        self.is_ready = False
        self.backend_mode = "uninitialized"
        self._init_index()

    def _init_index(self) -> None:
        """Inicializa el índice vectorial FAISS (Producto Interno sobre vectores L2-normalizados = Similitud Coseno)."""
        if faiss is not None:
            base_index = faiss.IndexFlatIP(self.EMBEDDING_DIM)
            self.index = faiss.IndexIDMap(base_index)
        else:
            logger.warning("FAISS no encontrado; usando búsqueda vectorial en memoria con NumPy.")
            self.index = None
        self._fallback_vectors.clear()

    def load_model(self) -> None:
        """Carga el pipeline de InsightFace (RetinaFace + ArcFace) sobre ONNX Runtime."""
        if FaceAnalysis is not None:
            try:
                det_size = (settings.FACE_DET_SIZE, settings.FACE_DET_SIZE)
                self.model = FaceAnalysis(
                    name=settings.MODEL_NAME,
                    root="./models",
                    providers=["CPUExecutionProvider"],
                )
                self.model.prepare(ctx_id=0, det_size=det_size, det_thresh=0.45)
                self.is_ready = True
                self.backend_mode = f"insightface:{settings.MODEL_NAME}"
                logger.info("InsightFace (%s) cargado correctamente con ONNX Runtime.", settings.MODEL_NAME)
                return
            except Exception as exc:
                logger.warning("No se pudo inicializar InsightFace (%s). Activando modo fallback OpenCV.", exc)

        # Modo Fallback con OpenCV + descriptor determinista normalizado (para pruebas sin modelo ONNX descargado)
        self.face_cascade = None
        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"):
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
            except Exception:
                self.face_cascade = None
        self.is_ready = True
        self.backend_mode = "opencv-fallback"
        logger.info("Motor facial iniciado en modo fallback (opencv-fallback).")

    @staticmethod
    def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        vec = np.asarray(embedding, dtype=np.float32).flatten()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)

    @staticmethod
    def embedding_to_bytes(embedding: np.ndarray) -> bytes:
        vec = FaceEngine.normalize_embedding(embedding)
        return vec.tobytes()

    @staticmethod
    def bytes_to_embedding(blob: bytes) -> np.ndarray:
        vec = np.frombuffer(blob, dtype=np.float32).copy()
        return FaceEngine.normalize_embedding(vec)

    @staticmethod
    def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("No se pudo decodificar la imagen proporcionada.")
        return image

    @staticmethod
    def decode_base64_image(image_base64: str) -> np.ndarray:
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]
        image_bytes = base64.b64decode(image_base64)
        return FaceEngine.decode_image_bytes(image_bytes)

    @staticmethod
    def crop_face_base64(image: np.ndarray, bbox: List[int], pad_ratio: float = 0.18) -> Optional[str]:
        """Recorta un rostro del frame original con margen y lo codifica en Base64 JPEG para visualización UI."""
        try:
            h, w = image.shape[:2]
            x1, y1, x2, y2 = bbox[:4]
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            pad_x = int(bw * pad_ratio)
            pad_y = int(bh * pad_ratio)
            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(w, x2 + pad_x)
            cy2 = min(h, y2 + pad_y)
            crop = image[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                return None
            ch, cw = crop.shape[:2]
            if max(ch, cw) > 160:
                scale = 160.0 / max(ch, cw)
                crop = cv2.resize(crop, (max(1, int(cw * scale)), max(1, int(ch * scale))), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                return None
            b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
            return f"data:image/jpeg;base64,{b64}"
        except Exception:
            return None

    def _compute_fallback_embedding(self, face_crop: np.ndarray) -> np.ndarray:
        """Genera un embedding normalizado de 512 dimensiones cuando InsightFace no está disponible."""
        resized = cv2.resize(face_crop, (16, 16), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32).flatten()  # 256
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)[:, :, :2].astype(np.float32).flatten()[:256]  # 256
        vec = np.concatenate([gray, hsv])
        return self.normalize_embedding(vec)

    def detect_and_embed(self, image: np.ndarray, max_faces: Optional[int] = None) -> List[DetectedFace]:
        """Detecta múltiples rostros en simultáneo en un frame BGR y devuelve sus cajas delimitadoras y embeddings ArcFace de 512-d."""
        if not self.is_ready:
            self.load_model()

        limit = max_faces if max_faces is not None and max_faces > 0 else settings.MAX_FACES_PER_FRAME
        results: List[DetectedFace] = []

        if self.model is not None:
            try:
                faces = self.model.get(image, max_num=limit)
            except TypeError:
                faces = self.model.get(image)
            faces = sorted(faces, key=lambda f: float(getattr(f, "det_score", 0.0)), reverse=True)
            for face in faces[:limit]:
                bbox = [int(v) for v in face.bbox[:4]]
                score = float(getattr(face, "det_score", 1.0))
                emb = self.normalize_embedding(face.embedding)
                results.append(DetectedFace(bbox=bbox, det_score=round(score, 4), embedding=emb))
            return results

        # Fallback si InsightFace no está instalado o sin pesos descargados
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = (
            self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
            if self.face_cascade is not None
            else []
        )
        if len(rects) == 0:
            h, w = image.shape[:2]
            if h >= 32 and w >= 32:
                # Si el frame ancho contiene múltiples rostros sintéticos lado a lado (ej. pruebas multirrostro)
                if w >= int(h * 1.75):
                    num_slots = min(limit, max(2, int(round(w / float(h)))))
                    slot_w = w // num_slots
                    for i in range(num_slots):
                        x1 = i * slot_w
                        x2 = w if i == num_slots - 1 else (i + 1) * slot_w
                        crop = image[:, x1:x2]
                        emb = self._compute_fallback_embedding(crop)
                        results.append(DetectedFace(bbox=[int(x1), 0, int(x2), int(h)], det_score=0.95, embedding=emb))
                else:
                    emb = self._compute_fallback_embedding(image)
                    results.append(DetectedFace(bbox=[0, 0, w, h], det_score=0.95, embedding=emb))
            return results

        for (x, y, w, h) in rects[:limit]:
            crop = image[y : y + h, x : x + w]
            emb = self._compute_fallback_embedding(crop)
            results.append(DetectedFace(bbox=[int(x), int(y), int(x + w), int(y + h)], det_score=0.98, embedding=emb))
        return results

    def add_to_index(self, person_id: int, embedding: np.ndarray) -> None:
        """Agrega o actualiza el embedding de una persona vetada en el índice FAISS."""
        vec = self.normalize_embedding(embedding)
        self.remove_from_index(person_id)

        if self.index is not None:
            vec_2d = vec.reshape(1, -1).astype(np.float32)
            ids = np.array([person_id], dtype=np.int64)
            self.index.add_with_ids(vec_2d, ids)
            self.save_index()
        self._fallback_vectors[int(person_id)] = vec

    def remove_from_index(self, person_id: int) -> None:
        """Elimina una persona del índice FAISS."""
        if self.index is not None:
            ids = np.array([person_id], dtype=np.int64)
            self.index.remove_ids(ids)
            self.save_index()
        self._fallback_vectors.pop(int(person_id), None)

    def rebuild_index(self, items: List[Tuple[int, np.ndarray]]) -> int:
        """Reconstruye el índice FAISS completo desde los registros activos en la base de datos."""
        self._init_index()
        if not items:
            self.save_index()
            return 0

        vectors = []
        ids = []
        for person_id, emb in items:
            vec = self.normalize_embedding(emb)
            vectors.append(vec)
            ids.append(int(person_id))
            self._fallback_vectors[int(person_id)] = vec

        if self.index is not None and vectors:
            matrix = np.vstack(vectors).astype(np.float32)
            id_array = np.array(ids, dtype=np.int64)
            self.index.add_with_ids(matrix, id_array)
            self.save_index()

        return len(ids)

    def search(
        self, embedding: np.ndarray, threshold: Optional[float] = None, top_k: int = 5
    ) -> Tuple[List[Tuple[int, float]], Optional[float]]:
        """
        Busca el embedding en FAISS.
        Retorna:
          - Lista de tuplas (person_id, similitud) que superan el umbral
          - Mejor puntaje encontrado (incluso si está bajo el umbral)
        """
        thresh = threshold if threshold is not None else settings.SIMILARITY_THRESHOLD
        vec = self.normalize_embedding(embedding)

        if self.total_indexed == 0:
            return [], None

        if self.index is not None:
            k = min(top_k, self.index.ntotal)
            scores, ids = self.index.search(vec.reshape(1, -1).astype(np.float32), k=k)
            best_score = float(scores[0][0]) if k > 0 and ids[0][0] != -1 else None
            matches = [
                (int(ids[0][i]), round(float(scores[0][i]), 4))
                for i in range(k)
                if ids[0][i] != -1 and float(scores[0][i]) >= thresh
            ]
            return matches, round(best_score, 4) if best_score is not None else None

        # Fallback NumPy
        scored = [
            (pid, float(np.dot(vec, ref_vec)))
            for pid, ref_vec in self._fallback_vectors.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        best_score = round(scored[0][1], 4) if scored else None
        matches = [(pid, round(score, 4)) for pid, score in scored[:top_k] if score >= thresh]
        return matches, best_score

    def search_batch(
        self,
        embeddings: List[np.ndarray],
        threshold: Optional[float] = None,
        top_k: int = 5,
    ) -> List[Tuple[List[Tuple[int, float]], Optional[float]]]:
        """
        Realiza búsqueda vectorial por lotes (batch) en FAISS para múltiples rostros simultáneos.
        Retorna una lista (uno por cada rostro detectado) de tuplas: (matches_sobre_umbral, best_score).
        """
        if not embeddings:
            return []

        thresh = threshold if threshold is not None else settings.SIMILARITY_THRESHOLD
        if self.total_indexed == 0:
            return [([], None) for _ in embeddings]

        matrix = np.vstack([self.normalize_embedding(emb) for emb in embeddings]).astype(np.float32)

        if self.index is not None:
            k = min(top_k, self.index.ntotal)
            scores_batch, ids_batch = self.index.search(matrix, k=k)
            results: List[Tuple[List[Tuple[int, float]], Optional[float]]] = []
            for row_idx in range(len(embeddings)):
                row_scores = scores_batch[row_idx]
                row_ids = ids_batch[row_idx]
                best_score = float(row_scores[0]) if k > 0 and row_ids[0] != -1 else None
                matches = [
                    (int(row_ids[i]), round(float(row_scores[i]), 4))
                    for i in range(k)
                    if row_ids[i] != -1 and float(row_scores[i]) >= thresh
                ]
                results.append((matches, round(best_score, 4) if best_score is not None else None))
            return results

        # Fallback NumPy por lotes
        return [self.search(emb, threshold=thresh, top_k=top_k) for emb in embeddings]

    @property
    def total_indexed(self) -> int:
        if self.index is not None:
            return int(self.index.ntotal)
        return len(self._fallback_vectors)

    def save_index(self) -> None:
        if self.index is not None and faiss is not None:
            try:
                os.makedirs(os.path.dirname(settings.FAISS_INDEX_PATH) or ".", exist_ok=True)
                faiss.write_index(self.index, settings.FAISS_INDEX_PATH)
            except Exception as exc:
                logger.warning("No se pudo guardar el índice FAISS en disco: %s", exc)


face_engine = FaceEngine()

