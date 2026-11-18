# 🏟️ Módulo 3: Reconocimiento Facial — Vigilancia en Estadios

Microservicio REST en **Python + FastAPI** encargado del **Módulo 3 (Reconocimiento Facial)** para el Sistema de Gestión y Control de Vigilancia en Estadios.

Detecta rostros en zonas de acceso controladas, extrae vectores biométricos de 512 dimensiones (**InsightFace / ArcFace** sobre **ONNX Runtime**) y los compara en milisegundos contra la lista de personas vetadas utilizando **FAISS**.

---

## 🚀 Instalación y Ejecución Rápida

### 1. Crear entorno virtual e instalar dependencias
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. (Opcional) Pre-descargar modelos ONNX de InsightFace (`buffalo_l`)
```powershell
python -m scripts.download_models
```
*(Si se ejecuta por primera vez sin internet o sin los pesos ONNX aún descargados, el motor cuenta con un modo fallback automático con OpenCV para que puedas probar e integrar la API sin bloqueos).*

### 3. Levantar el servidor FastAPI (Puerto `8003`)
```powershell
uvicorn app.main:app --reload --port 8003
```

- **Documentación interactiva (Swagger UI):** `http://localhost:8003/docs`
- **Health Check:** `http://localhost:8003/health`

---

## 🔌 Endpoints Principales para Integración

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | Estado del microservicio, motor de inferencia e índice FAISS |
| `GET` | `/api/v1/stats` | Estadísticas de vetados activos, vectores en FAISS y alertas |
| `POST` | `/api/v1/vetados` | Registrar persona vetada (`multipart/form-data` con foto) |
| `GET` | `/api/v1/vetados` | Listar personas vetadas |
| `GET` | `/api/v1/vetados/{id}` | Obtener detalle de persona vetada |
| `PUT` | `/api/v1/vetados/{id}` | Actualizar datos o estado de veto |
| `DELETE` | `/api/v1/vetados/{id}` | Desactivar veto y retirar del índice FAISS |
| `POST` | `/api/v1/recognize` | Analizar frame de cámara enviado como archivo (`multipart/form-data`) |
| `POST` | `/api/v1/recognize/base64` | Analizar frame enviado en JSON Base64 (ideal para API Gateway) |
| `GET` | `/api/v1/alertas` | Listar alertas generadas en los accesos |
| `GET` | `/api/v1/alertas/{id}` | Detalle de alerta con captura y datos del vetado |
| `PATCH` | `/api/v1/alertas/{id}/revisar` | Marcar alerta como revisada por seguridad |

---

## 🧪 Ejecutar Pruebas Automatizadas
```powershell
pytest -v
```

