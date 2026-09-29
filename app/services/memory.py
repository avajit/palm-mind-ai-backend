import json
from typing import List, Dict, Any, Optional
import redis
from app.config import settings

class MemoryService:
    """
    Manages chat history and booking state in Redis.

    Why Redis?
    Redis provides fast, in-memory key-value storage with built-in TTL (Time-To-Live)
    expiration. This is ideal for ephemeral chat sessions and temporary booking state
    without burdening our relational database.
    """

    def __init__(self) -> None:
        self._redis = redis.Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            protocol=2
        )
        self.ttl_seconds = 86400  # 24 hours

    def _history_key(self, session_id: str) -> str:
        return f"chat_history:{session_id}"

    def _booking_key(self, session_id: str) -> str:
        return f"booking_state:{session_id}"

    def add_message(self, session_id: str, role: str, content: str, max_messages: int = 20) -> None:
        """
        Appends a message (user or assistant) to the session's chat history in Redis.
        Maintains a rolling window of max_messages and updates TTL.
        """
        key = self._history_key(session_id)
        msg = json.dumps({"role": role, "content": content})
        
        # Push to list
        self._redis.rpush(key, msg)
        # Trim to keep only the last N messages
        self._redis.ltrim(key, -max_messages, -1)
        # Refresh TTL
        self._redis.expire(key, self.ttl_seconds)

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Retrieves the chat history for a given session ID.
        Returns a list of dicts: [{"role": "user", "content": "..."}, ...]
        """
        key = self._history_key(session_id)
        raw_msgs = self._redis.lrange(key, 0, -1)
        
        history: List[Dict[str, str]] = []
        for raw in raw_msgs:
            try:
                history.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
        return history

    def get_booking_state(self, session_id: str) -> Dict[str, Any]:
        """
        Retrieves the in-progress interview booking state for a session.
        Returns a dict containing fields like name, email, date, time.
        """
        key = self._booking_key(session_id)
        data = self._redis.hgetall(key)
        return data if data else {}

    def update_booking_state(self, session_id: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        """
        Updates or merges new fields into the session's booking state hash.
        Refreshes TTL.
        """
        key = self._booking_key(session_id)
        # Convert all values to strings for Redis hash storage
        str_fields = {k: str(v) for k, v in fields.items() if v is not None}
        if str_fields:
            for k, v in str_fields.items():
                self._redis.hset(key, k, v)
            self._redis.expire(key, self.ttl_seconds)
        return self.get_booking_state(session_id)

    def clear_booking_state(self, session_id: str) -> None:
        """
        Clears/deletes the booking state after a booking is successfully completed.
        """
        key = self._booking_key(session_id)
        self._redis.delete(key)
