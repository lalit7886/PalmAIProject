import pytest
from redis.exceptions import ConnectionError

from src.core.config import get_settings
from src.services import conversation_store


class FakePipeline:
    def __init__(self, client):
        self.client = client
        self.commands = []

    def rpush(self, key, *values):
        self.commands.append(("rpush", key, values))
        return self

    def ltrim(self, key, start, end):
        self.commands.append(("ltrim", key, start, end))
        return self

    def expire(self, key, seconds):
        self.commands.append(("expire", key, seconds))
        return self

    def execute(self):
        for command, *args in self.commands:
            getattr(self.client, command)(*args)


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.expirations = {}

    def pipeline(self, *, transaction):
        assert transaction is True
        return FakePipeline(self)

    def rpush(self, key, values):
        self.values.setdefault(key, []).extend(values)

    @staticmethod
    def _slice(values, start, end):
        length = len(values)
        start = max(0, length + start) if start < 0 else start
        end = length + end if end < 0 else end
        return values[start : end + 1]

    def ltrim(self, key, start, end):
        values = self.values.get(key, [])
        self.values[key] = self._slice(values, start, end)

    def expire(self, key, seconds):
        self.expirations[key] = seconds
        return True

    def lrange(self, key, start, end):
        values = self.values.get(key, [])
        return self._slice(values, start, end)


@pytest.fixture
def redis_store(monkeypatch):
    client = FakeRedis()
    monkeypatch.setattr(conversation_store, "_get_redis_client", lambda: client)
    monkeypatch.setattr(get_settings(), "conversation_history_limit", 20)
    monkeypatch.setattr(get_settings(), "conversation_ttl_seconds", 86400)
    return client


def test_conversation_store_persists_ordered_user_scoped_history(redis_store):
    conversation_store.save_conversation_turn(
        "conversation-1", "user-1", "First question", "First answer"
    )
    conversation_store.save_conversation_turn(
        "conversation-1", "user-1", "Second question", "Second answer"
    )
    conversation_store.save_conversation_turn(
        "conversation-1", "user-2", "Private question", "Private answer"
    )

    history = conversation_store.get_conversation_history(
        "conversation-1", "user-1"
    )

    assert history == [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Second question"},
        {"role": "assistant", "content": "Second answer"},
    ]


def test_conversation_store_limits_history_to_latest_messages(redis_store):
    conversation_store.save_conversation_turn(
        "conversation-1", "user-1", "First question", "First answer"
    )
    conversation_store.save_conversation_turn(
        "conversation-1", "user-1", "Second question", "Second answer"
    )

    history = conversation_store.get_conversation_history(
        "conversation-1", "user-1", limit=2
    )

    assert history == [
        {"role": "user", "content": "Second question"},
        {"role": "assistant", "content": "Second answer"},
    ]


def test_conversation_store_expires_history(redis_store):
    conversation_store.save_conversation_turn(
        "conversation-1", "user-1", "Question", "Answer"
    )

    key = conversation_store._conversation_key("conversation-1", "user-1")
    assert redis_store.expirations[key] == 86400


def test_conversation_store_rejects_non_positive_history_limit(redis_store):
    with pytest.raises(ValueError, match="History limit must be positive"):
        conversation_store.get_conversation_history(
            "conversation-1", "user-1", limit=0
        )


def test_conversation_store_reports_redis_connection_errors(monkeypatch):
    class UnavailableRedis:
        def lrange(self, *_):
            raise ConnectionError("connection refused")

    monkeypatch.setattr(
        conversation_store, "_get_redis_client", lambda: UnavailableRedis()
    )

    with pytest.raises(RuntimeError, match="Redis conversation memory is unavailable"):
        conversation_store.get_conversation_history("conversation-1", "user-1")
