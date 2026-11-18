import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class AlertWebSocketManager:
    """Gestiona conexiones WebSocket activas para enviar alertas en tiempo real al centro de control."""

    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast_alert(self, payload: Dict[str, Any]) -> None:
        if not self.active_connections:
            return
        disconnected: List[WebSocket] = []
        for connection in self.active_connections:
            try:
                await connection.send_json(payload)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            self.disconnect(conn)

    def notify_from_sync(self, payload: Dict[str, Any]) -> None:
        """Envía una notificación desde contextos síncronos si hay un event loop activo."""
        if not self.active_connections:
            return
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.broadcast_alert(payload))
        except RuntimeError:
            pass


ws_manager = AlertWebSocketManager()

