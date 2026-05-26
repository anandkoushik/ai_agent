import time
import asyncio
import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class MemoryProvider(ABC):
    """Abstraction interface for session memory."""
    @abstractmethod
    async def get_session(self, session_id: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def update_session(self, session_id: str, key: str, value: Any) -> None:
        pass

    @abstractmethod
    async def cleanup_expired(self) -> None:
        pass

class InMemorySessionProvider(MemoryProvider):
    """
    Temporary thread-safe RAM provider matching future Postgres schema.
    Warning: Context lost on server restart.
    """
    def __init__(self, ttl_seconds: int = 3600):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self.ttl_seconds = ttl_seconds
        
    async def get_session(self, session_id: str) -> Dict[str, Any]:
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                session = {
                    "session_id": session_id,
                    "active_dataset_path": None,
                    "active_inference_path": None,
                    "workflow_stage": "idle",
                    "chat_history": [],
                    "last_accessed": time.time()
                }
                self._sessions[session_id] = session
            else:
                session["last_accessed"] = time.time()
            return dict(session) # Return copy

    async def update_session(self, session_id: str, key: str, value: Any) -> None:
        async with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = {
                    "session_id": session_id,
                    "active_dataset_path": None,
                    "active_inference_path": None,
                    "workflow_stage": "idle",
                    "chat_history": [],
                    "last_accessed": time.time()
                }
            self._sessions[session_id][key] = value
            self._sessions[session_id]["last_accessed"] = time.time()
            
    async def cleanup_expired(self) -> None:
        """TTL Expiration Policy."""
        async with self._lock:
            now = time.time()
            expired_keys = [
                sid for sid, data in self._sessions.items() 
                if now - data["last_accessed"] > self.ttl_seconds
            ]
            for key in expired_keys:
                logger.info(f"Session {key} expired. Cleaning up memory.")
                # Future: trigger disk cleanup of workspaces here
                del self._sessions[key]

# Singleton instance
SessionManager = InMemorySessionProvider()
