import hashlib
import json
from functools import lru_cache
from typing import TypedDict

from redis import Redis
from redis.exceptions import RedisError

from src.core.config import get_settings
from src.core.logging import logging


class ConversationMessage(TypedDict):
    role: str
    content: str


@lru_cache
def _get_redis_client() -> Redis:
    settings = get_settings()
    return Redis.from_url(settings.redis_url, decode_responses=True)


def _conversation_key(conversation_id: str, user_id: str) -> str:
    identity = f"{len(user_id)}:{user_id}{conversation_id}".encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()
    return f"palm-ai:conversation:{digest}"


def get_conversation_history(
    conversation_id: str,
    user_id: str,
    limit: int | None = None,
) -> list[ConversationMessage]:
    settings = get_settings()
    history_limit = limit if limit is not None else settings.conversation_history_limit
    if history_limit < 1:
        raise ValueError("History limit must be positive.")
    if not conversation_id.strip() or not user_id.strip():
        raise ValueError("Conversation ID and user ID must not be empty.")

    try:
        serialized_messages = _get_redis_client().lrange(
            _conversation_key(conversation_id, user_id),
            -history_limit,
            -1,
        )
    except RedisError as exc:
        raise RuntimeError("Redis conversation memory is unavailable.") from exc

    history = []
    for serialized_message in serialized_messages:
        try:
            message = json.loads(serialized_message)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Redis conversation memory contains invalid data.") from exc
        if (
            not isinstance(message, dict)
            or message.get("role") not in {"user", "assistant"}
            or not isinstance(message.get("content"), str)
        ):
            raise RuntimeError("Redis conversation memory contains invalid data.")
        history.append(
            {"role": message["role"], "content": message["content"]}
        )
    return history


def save_conversation_turn(
    conversation_id: str,
    user_id: str,
    question: str,
    answer: str,
) -> None:
    if not conversation_id.strip() or not user_id.strip():
        raise ValueError("Conversation ID and user ID must not be empty.")
    if not question.strip() or not answer.strip():
        raise ValueError("Question and answer must not be empty.")

    settings = get_settings()
    if settings.conversation_history_limit < 1:
        raise ValueError("History limit must be positive.")
    if settings.conversation_ttl_seconds < 1:
        raise ValueError("Conversation TTL must be positive.")

    key = _conversation_key(conversation_id, user_id)
    messages = [
        json.dumps({"role": "user", "content": question}),
        json.dumps({"role": "assistant", "content": answer}),
    ]
    try:
        pipeline = _get_redis_client().pipeline(transaction=True)
        pipeline.rpush(key, *messages)
        pipeline.ltrim(key, -settings.conversation_history_limit, -1)
        pipeline.expire(key, settings.conversation_ttl_seconds)
        pipeline.execute()
    except RedisError as exc:
        raise RuntimeError("Redis conversation memory is unavailable.") from exc

    logging.info("Saved conversation turn conversation_id=%s", conversation_id)
