from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "facial-recognition-module"
    APP_PORT: int = 8003
    DEBUG: bool = True

    # Base de datos
    DATABASE_URL: str = "sqlite:///./data/facial_recognition.db"

    # Reconocimiento facial
    FACE_DET_SIZE: int = 640
    SIMILARITY_THRESHOLD: float = 0.45
    MAX_FACES_PER_FRAME: int = 50
    MODEL_NAME: str = "buffalo_l"

    # Análisis de video y control de alertas
    VIDEO_SAMPLE_EVERY_N_FRAMES: int = 8
    MAX_VIDEO_FRAMES_TO_ANALYZE: int = 150
    ALERT_COOLDOWN_SECONDS: float = 8.0

    # Almacenamiento e índice FAISS
    FAISS_INDEX_PATH: str = "./data/faiss_index.bin"
    VETADOS_DIR: str = "./data/vetados"
    SCREENSHOTS_DIR: str = "./data/screenshots"
    VIDEOS_DIR: str = "./data/videos"

    # CORS
    ALLOWED_ORIGINS: str = "*"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origins(self) -> List[str]:
        if self.ALLOWED_ORIGINS.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]

    def ensure_dirs(self) -> None:
        Path("./data").mkdir(parents=True, exist_ok=True)
        Path(self.VETADOS_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.SCREENSHOTS_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.VIDEOS_DIR).mkdir(parents=True, exist_ok=True)
        Path("./models").mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()

