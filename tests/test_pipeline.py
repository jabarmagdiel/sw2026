import base64
import io
import os
import tempfile

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.deps import get_db
from app.config import settings
from app.db.database import Base
from app.main import app
from app.services.face_engine import DetectedFace, face_engine

# Base de datos SQLite en memoria aislada para pruebas (no borra datos reales en ./data/)
TEST_ENGINE = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=TEST_ENGINE)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


def _create_synthetic_face_img(seed: int = 42) -> np.ndarray:
    """Genera una matriz BGR determinista con rasgos geométricos tipo rostro."""
    rng = np.random.default_rng(seed)
    img = rng.integers(40, 200, size=(200, 200, 3), dtype=np.uint8)
    cv2.circle(img, (100, 100), 65, (210, 190, 175), -1)
    cv2.circle(img, (75, 85), 10, (30, 30, 30), -1)
    cv2.circle(img, (125, 85), 10, (30, 30, 30), -1)
    cv2.ellipse(img, (100, 130), (25, 12), 0, 0, 180, (40, 40, 150), 3)
    return img


def _create_synthetic_face_bytes(seed: int = 42) -> bytes:
    img = _create_synthetic_face_img(seed=seed)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


@pytest.fixture(autouse=True)
def setup_clean_db(monkeypatch):
    # Evitar sobrescribir el archivo faiss_index.bin de producción durante los tests
    monkeypatch.setattr(settings, "FAISS_INDEX_PATH", "./data/test_faiss_index.bin")
    Base.metadata.drop_all(bind=TEST_ENGINE)
    Base.metadata.create_all(bind=TEST_ENGINE)
    # Forzar modo fallback determinista durante tests sintéticos
    monkeypatch.setattr(face_engine, "model", None)
    monkeypatch.setattr(face_engine, "face_cascade", None, raising=False)
    monkeypatch.setattr(face_engine, "is_ready", True)
    face_engine.rebuild_index([])
    yield
    if os.path.exists("./data/test_faiss_index.bin"):
        try:
            os.remove("./data/test_faiss_index.bin")
        except OSError:
            pass


client = TestClient(app)


def test_health_and_stats():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["indexed_faces_in_faiss"] == 0

    stats = client.get("/api/v1/stats")
    assert stats.status_code == 200
    assert stats.json()["vetados_activos"] == 0


def test_full_blacklist_and_recognition_pipeline():
    face_bytes = _create_synthetic_face_bytes(seed=101)

    # 1. Registrar persona vetada
    files = {"foto": ("vetado1.jpg", io.BytesIO(face_bytes), "image/jpeg")}
    form_data = {
        "cedula": "88776655",
        "nombre": "Carlos",
        "apellido": "Rojas",
        "motivo_veto": "Reincidencia en disturbios en tribuna norte",
        "nivel_riesgo": "CRITICO",
    }
    res_create = client.post("/api/v1/vetados", data=form_data, files=files)
    assert res_create.status_code == 201, res_create.text
    vetado = res_create.json()
    assert vetado["cedula"] == "88776655"
    assert face_engine.total_indexed == 1

    # 2. Reconocer el mismo rostro en la cámara de acceso
    rec_files = {"file": ("cam_frame.jpg", io.BytesIO(face_bytes), "image/jpeg")}
    res_rec = client.post(
        "/api/v1/recognize",
        data={"camara_id": "CAM-NORTE-01", "zona": "Puerta Norte", "threshold": "0.45"},
        files=rec_files,
    )
    assert res_rec.status_code == 200, res_rec.text
    rec_data = res_rec.json()
    assert rec_data["alert_triggered"] is True
    assert len(rec_data["matches"]) == 1
    assert rec_data["matches"][0]["match"]["cedula"] == "88776655"
    assert rec_data["matches"][0]["face_crop_base64"] is not None
    alerta_id = rec_data["matches"][0]["match"]["alerta_id"]

    # 3. Reconocimiento vía Base64 (para integración con otros microservicios)
    b64_str = base64.b64encode(face_bytes).decode("utf-8")
    res_b64 = client.post(
        "/api/v1/recognize/base64",
        json={
            "image_base64": b64_str,
            "camara_id": "CAM-SUR-02",
            "zona": "Puerta Sur",
            "threshold": 0.45,
        },
    )
    assert res_b64.status_code == 200
    assert res_b64.json()["alert_triggered"] is True

    # 4. Consultar y revisar alerta
    res_alerts = client.get("/api/v1/alertas")
    assert res_alerts.status_code == 200
    assert len(res_alerts.json()) >= 1

    res_review = client.patch(
        f"/api/v1/alertas/{alerta_id}/revisar",
        json={"revisada": True, "notas": "Retenido por seguridad en acceso norte"},
    )
    assert res_review.status_code == 200
    assert res_review.json()["revisada"] is True

    # 5. Desactivar vetado y verificar que sale del índice FAISS
    res_del = client.delete(f"/api/v1/vetados/{vetado['id']}")
    assert res_del.status_code == 200
    assert face_engine.total_indexed == 0


def test_multiface_simultaneous_and_video_recognition():
    # Crear 2 rostros sintéticos distintos para vetar y 1 rostro civil autorizado
    face1_img = _create_synthetic_face_img(seed=201)
    face2_img = _create_synthetic_face_img(seed=302)
    face3_img = _create_synthetic_face_img(seed=999)
    # Diferenciar claramente colores/contraste de cada rostro sintético
    face1_img[:, :, 0] = 20
    face2_img[:, :, 2] = 20
    face3_img[:, :] = (240, 240, 240)

    ok1, buf1 = cv2.imencode(".jpg", face1_img)
    ok2, buf2 = cv2.imencode(".jpg", face2_img)
    assert ok1 and ok2

    # Registrar Vetado 1 y Vetado 2
    r1 = client.post(
        "/api/v1/vetados",
        data={"cedula": "111111", "nombre": "Pedro", "apellido": "Uno", "motivo_veto": "Veto 1", "nivel_riesgo": "ALTO"},
        files={"foto": ("v1.jpg", io.BytesIO(buf1.tobytes()), "image/jpeg")},
    )
    r2 = client.post(
        "/api/v1/vetados",
        data={"cedula": "222222", "nombre": "Marco", "apellido": "Dos", "motivo_veto": "Veto 2", "nivel_riesgo": "CRITICO"},
        files={"foto": ("v2.jpg", io.BytesIO(buf2.tobytes()), "image/jpeg")},
    )
    assert r1.status_code == 201 and r2.status_code == 201
    assert face_engine.total_indexed == 2

    # Crear un fotograma grupal con 3 rostros simultáneos lado a lado (600x200)
    multi_frame = np.hstack([face1_img, face2_img, face3_img])
    ok_m, buf_m = cv2.imencode(".jpg", multi_frame)
    assert ok_m

    res_multi = client.post(
        "/api/v1/recognize",
        data={"camara_id": "CAM-MULTIRROSTRO", "zona": "Acceso Central", "threshold": "0.85"},
        files={"file": ("group.jpg", io.BytesIO(buf_m.tobytes()), "image/jpeg")},
    )
    assert res_multi.status_code == 200, res_multi.text
    mdata = res_multi.json()
    assert mdata["faces_detected"] == 3
    assert mdata["alert_triggered"] is True
    assert len(mdata["matches"]) >= 2

    # Crear un archivo de video AVI/MP4 temporal con 6 fotogramas multirrostro
    with tempfile.NamedTemporaryFile(delete=False, suffix=".avi") as tmp_vid:
        vid_path = tmp_vid.name

    try:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(vid_path, fourcc, 10.0, (600, 200))
        for _ in range(6):
            writer.write(multi_frame)
        writer.release()

        with open(vid_path, "rb") as vf:
            video_bytes = vf.read()

        res_vid = client.post(
            "/api/v1/recognize/video",
            data={
                "camara_id": "CCTV-VIDEO-TEST",
                "zona": "Puerta Sur",
                "sample_every_n_frames": "2",
                "max_frames": "10",
                "threshold": "0.85",
            },
            files={"file": ("test_crowd.avi", io.BytesIO(video_bytes), "video/x-msvideo")},
        )
        assert res_vid.status_code == 200, res_vid.text
        vdata = res_vid.json()
        assert vdata["frames_analyzed"] == 3
        assert vdata["max_simultaneous_faces"] == 3
        assert vdata["total_faces_detected"] == 9
        assert vdata["alert_triggered"] is True
        assert len(vdata["unique_vetados_detected"]) >= 2
        assert len(vdata["timeline"]) == 3
    finally:
        if os.path.exists(vid_path):
            os.remove(vid_path)
