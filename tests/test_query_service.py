from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from src.services import query_service


class FakeVectorStore:
    def __init__(self, documents):
        self.documents = documents
        self.search_args = None

    def similarity_search(self, query, *, k, filter):
        self.search_args = {"query": query, "k": k, "filter": filter}
        return self.documents


class FakeChatModel:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.messages = None
        self.__class__.instances.append(self)

    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(content="Answer supported by [1].")


@pytest.fixture
def fake_vector_store(monkeypatch):
    store = FakeVectorStore(
        [
            Document(
                page_content="PalmAI stores embeddings in Pinecone.",
                metadata={
                    "document_id": "doc-1",
                    "filename": "architecture.txt",
                    "chunk_index": 2,
                    "user_id": "user-1",
                },
            )
        ]
    )
    monkeypatch.setattr(query_service, "get_vectorstore", lambda: store)
    monkeypatch.setattr(query_service, "ChatOpenAI", FakeChatModel)
    FakeChatModel.instances.clear()
    monkeypatch.setattr(query_service.get_settings(), "groq_api_key", "test-key")
    return store


def test_answer_question_retrieves_user_scoped_context_and_returns_sources(
    fake_vector_store,
):
    result = query_service.answer_question("Where are embeddings stored?", "user-1")

    assert result == {
        "answer": "Answer supported by [1].",
        "sources": [
            {
                "citation": 1,
                "document_id": "doc-1",
                "filename": "architecture.txt",
                "chunk_index": 2,
                "content": "PalmAI stores embeddings in Pinecone.",
            }
        ],
    }
    assert fake_vector_store.search_args == {
        "query": "Where are embeddings stored?",
        "k": query_service.get_settings().top_k,
        "filter": {"user_id": "user-1"},
    }
    assert FakeChatModel.instances[0].kwargs["base_url"] == (
        "https://api.groq.com/openai/v1"
    )


def test_answer_question_without_matches_does_not_call_language_model(
    monkeypatch,
):
    store = FakeVectorStore([])
    monkeypatch.setattr(query_service, "get_vectorstore", lambda: store)
    monkeypatch.setattr(
        query_service,
        "ChatOpenAI",
        lambda **_: pytest.fail("model should not be called without context"),
    )

    result = query_service.answer_question("Unknown question", "user-1")

    assert result == {
        "answer": query_service.NO_CONTEXT_ANSWER,
        "sources": [],
    }


def test_answer_question_requires_groq_key(monkeypatch, fake_vector_store):
    monkeypatch.setattr(query_service.get_settings(), "groq_api_key", "")

    with pytest.raises(RuntimeError, match="GROQ_API_KEY is not configured"):
        query_service.answer_question("Question", "user-1")


def test_query_api_requires_admin_key(monkeypatch):
    from src import app as app_module

    monkeypatch.setattr(app_module.settings, "admin_api_key", "test-admin")
    client = TestClient(app_module.app)

    response = client.post(
        "/query",
        json={"question": "Question", "user_id": "user-1"},
    )

    assert response.status_code == 401


def test_query_api_returns_answer_and_sources(monkeypatch):
    from src import app as app_module

    monkeypatch.setattr(app_module.settings, "admin_api_key", "test-admin")
    monkeypatch.setattr(
        app_module,
        "answer_question",
        lambda question, user_id: {
            "answer": f"Answer for {question} ({user_id})",
            "sources": [],
        },
    )
    client = TestClient(app_module.app)

    response = client.post(
        "/query",
        headers={"x-admin-key": "test-admin"},
        json={"question": " What is PalmAI? ", "user_id": " user-1 "},
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Answer for What is PalmAI? (user-1)",
        "sources": [],
    }


def test_query_api_reports_missing_model_configuration(monkeypatch):
    from src import app as app_module

    monkeypatch.setattr(app_module.settings, "admin_api_key", "test-admin")

    def missing_configuration(**_):
        raise RuntimeError("GROQ_API_KEY is not configured.")

    monkeypatch.setattr(app_module, "answer_question", missing_configuration)
    client = TestClient(app_module.app)

    response = client.post(
        "/query",
        headers={"x-admin-key": "test-admin"},
        json={"question": "Question", "user_id": "user-1"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "GROQ_API_KEY is not configured."


def test_query_api_validates_non_whitespace_question(monkeypatch):
    from src import app as app_module

    monkeypatch.setattr(app_module.settings, "admin_api_key", "test-admin")
    client = TestClient(app_module.app)

    response = client.post(
        "/query",
        headers={"x-admin-key": "test-admin"},
        json={"question": "   ", "user_id": "user-1"},
    )

    assert response.status_code == 422
