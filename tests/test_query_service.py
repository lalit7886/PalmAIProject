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
    monkeypatch.setattr(query_service, "_extract_interview_booking", lambda *_: None)
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


def test_conversational_rag_uses_history_for_retrieval_and_prompt(
    monkeypatch,
    fake_vector_store,
):
    history = [
        {"role": "user", "content": "What is the first storage option?"},
        {"role": "assistant", "content": "The document mentions Pinecone."},
    ]
    saved_turns = []
    monkeypatch.setattr(
        query_service,
        "get_conversation_history",
        lambda conversation_id, user_id: history,
    )
    monkeypatch.setattr(
        query_service,
        "save_conversation_turn",
        lambda *args: saved_turns.append(args),
    )

    result = query_service.answer_conversational_question(
        "What about the second option?",
        "user-1",
        "conversation-1",
    )

    assert result["conversation_id"] == "conversation-1"
    assert result["answer"] == "Answer supported by [1]."
    assert "What is the first storage option?" in fake_vector_store.search_args["query"]
    assert "What about the second option?" in fake_vector_store.search_args["query"]
    model_messages = FakeChatModel.instances[0].messages
    assert model_messages[1].content == history[0]["content"]
    assert model_messages[2].content == history[1]["content"]
    assert saved_turns == [
        (
            "conversation-1",
            "user-1",
            "What about the second option?",
            "Answer supported by [1].",
        )
    ]


def test_regular_rag_turn_skips_booking_extraction(monkeypatch, fake_vector_store):
    monkeypatch.setattr(
        query_service,
        "get_conversation_history",
        lambda *_: [],
    )
    monkeypatch.setattr(
        query_service,
        "_extract_interview_booking",
        lambda *_: pytest.fail("booking extraction should be skipped"),
    )
    monkeypatch.setattr(query_service, "save_conversation_turn", lambda *_: None)

    result = query_service.answer_conversational_question(
        "Where are embeddings stored?",
        "user-1",
        "conversation-rag",
    )

    assert result["answer"] == "Answer supported by [1]."
    assert len(FakeChatModel.instances) == 1


def test_conversational_rag_saves_turn_when_no_documents_match(monkeypatch):
    store = FakeVectorStore([])
    saved_turns = []
    monkeypatch.setattr(query_service, "get_vectorstore", lambda: store)
    monkeypatch.setattr(query_service, "get_conversation_history", lambda *_: [])
    monkeypatch.setattr(
        query_service,
        "save_conversation_turn",
        lambda *args: saved_turns.append(args),
    )

    result = query_service.answer_conversational_question(
        "Unknown question",
        "user-1",
        "conversation-2",
    )

    assert result == {
        "answer": query_service.NO_CONTEXT_ANSWER,
        "sources": [],
        "conversation_id": "conversation-2",
    }
    assert saved_turns == [
        ("conversation-2", "user-1", "Unknown question", query_service.NO_CONTEXT_ANSWER)
    ]


def test_conversational_rag_collects_missing_interview_booking_details(monkeypatch):
    history = [
        {"role": "user", "content": "I want to book an interview."},
        {
            "role": "assistant",
            "content": (
                "To book the interview, please provide or correct: "
                "email, interview_time."
            ),
        },
    ]
    saved_turns = []
    monkeypatch.setattr(
        query_service,
        "get_conversation_history",
        lambda *_: history,
    )
    monkeypatch.setattr(query_service, "get_interview_booking", lambda **_: None)
    monkeypatch.setattr(
        query_service,
        "_extract_interview_booking",
        lambda *_: query_service.InterviewBookingExtraction(
            booking_requested=True,
            name="Alex Candidate",
            email=None,
            interview_date="2026-12-01",
            interview_time=None,
        ),
    )
    monkeypatch.setattr(
        query_service,
        "save_conversation_turn",
        lambda *args: saved_turns.append(args),
    )
    monkeypatch.setattr(
        query_service,
        "save_interview_booking",
        lambda **_: pytest.fail("incomplete booking must not be stored"),
    )

    result = query_service.answer_conversational_question(
        "My name is Alex Candidate and December 1 works.",
        "user-1",
        "conversation-booking",
    )

    assert result["booking_status"] == "collecting_details"
    assert "email" in result["answer"]
    assert "interview_time" in result["answer"]
    assert result["sources"] == []
    assert saved_turns[0][0:2] == ("conversation-booking", "user-1")


def test_booking_extraction_uses_llm_structured_output(monkeypatch):
    observed = {}
    expected = query_service.InterviewBookingExtraction(
        booking_requested=True,
        name="Alex Candidate",
        email="alex@example.com",
        interview_date="2026-12-01",
        interview_time="14:30",
    )

    class FakeStructuredModel:
        def invoke(self, messages):
            observed["messages"] = messages
            return expected

    class FakeModel:
        def __init__(self, **kwargs):
            observed["model_kwargs"] = kwargs

        def with_structured_output(self, schema):
            observed["schema"] = schema
            return FakeStructuredModel()

    monkeypatch.setattr(query_service, "ChatOpenAI", FakeModel)
    monkeypatch.setattr(query_service.get_settings(), "groq_api_key", "test-key")

    extraction = query_service._extract_interview_booking(
        "Book me for December 1 at 2:30 PM.",
        [{"role": "user", "content": "My name is Alex Candidate."}],
    )

    assert extraction == expected
    assert observed["schema"] is query_service.InterviewBookingExtraction
    assert observed["model_kwargs"]["base_url"] == (
        "https://api.groq.com/openai/v1"
    )
    assert "My name is Alex Candidate." in observed["messages"][1].content


def test_conversational_rag_saves_complete_booking(monkeypatch):
    from datetime import datetime

    from src.services.audit import InterviewBooking
    from src.services.calendar_service import CalendarEvent

    monkeypatch.setattr(query_service, "get_conversation_history", lambda *_: [])
    monkeypatch.setattr(query_service, "get_interview_booking", lambda **_: None)
    monkeypatch.setattr(
        query_service,
        "_extract_interview_booking",
        lambda *_: query_service.InterviewBookingExtraction(
            booking_requested=True,
            name="Alex Candidate",
            email="alex@example.com",
            interview_date="2026-12-01",
            interview_time="14:30",
        ),
    )
    saved_bookings = []
    monkeypatch.setattr(
        query_service,
        "create_interview_event",
        lambda **_: CalendarEvent(
            event_id="event-1",
            html_link="https://calendar.google.com/event?eid=event-1",
            start=datetime.fromisoformat("2026-12-01T14:30:00+05:45"),
            end=datetime.fromisoformat("2026-12-01T15:00:00+05:45"),
        ),
    )
    monkeypatch.setattr(
        query_service,
        "save_interview_booking",
        lambda **kwargs: (
            saved_bookings.append(kwargs)
            or (
                InterviewBooking(
                    booking_id="booking-1",
                    user_id=kwargs["user_id"],
                    conversation_id=kwargs["conversation_id"],
                    name=kwargs["name"],
                    email=kwargs["email"],
                    interview_date=kwargs["interview_date"],
                    interview_time=kwargs["interview_time"],
                    created_at="2026-11-13T00:00:00+00:00",
                    timezone=kwargs["timezone_name"],
                    calendar_event_id=kwargs["calendar_event_id"],
                    calendar_event_url=kwargs["calendar_event_url"],
                ),
                True,
            )
        ),
    )
    monkeypatch.setattr(query_service, "save_conversation_turn", lambda *_: None)

    result = query_service.answer_conversational_question(
        "Book me for December 1 at 2:30 PM. Alex Candidate, alex@example.com.",
        "user-1",
        "conversation-booking",
    )

    assert result["booking_status"] == "booked"
    assert result["booking"] == {
        "booking_id": "booking-1",
        "name": "Alex Candidate",
        "email": "alex@example.com",
        "interview_date": "2026-12-01",
        "interview_time": "14:30",
        "created_at": "2026-11-13T00:00:00+00:00",
        "timezone": "Asia/Kathmandu",
        "calendar_event_id": "event-1",
        "calendar_event_url": "https://calendar.google.com/event?eid=event-1",
    }
    assert saved_bookings == [
        {
            "user_id": "user-1",
            "conversation_id": "conversation-booking",
            "name": "Alex Candidate",
            "email": "alex@example.com",
            "interview_date": "2026-12-01",
            "interview_time": "14:30",
            "timezone_name": "Asia/Kathmandu",
            "calendar_event_id": "event-1",
            "calendar_event_url": "https://calendar.google.com/event?eid=event-1",
        }
    ]
    assert result["booking"]["calendar_event_id"] == "event-1"
    assert result["booking"]["timezone"] == "Asia/Kathmandu"


def test_conversational_rag_does_not_save_unavailable_calendar_slot(monkeypatch):
    monkeypatch.setattr(query_service, "get_interview_booking", lambda **_: None)
    monkeypatch.setattr(
        query_service,
        "create_interview_event",
        lambda **_: None,
    )
    monkeypatch.setattr(
        query_service,
        "save_interview_booking",
        lambda **_: pytest.fail("unavailable slot must not be persisted"),
    )
    monkeypatch.setattr(query_service, "save_conversation_turn", lambda *_: None)

    result = query_service._handle_interview_booking(
        query_service.InterviewBookingExtraction(
            booking_requested=True,
            name="Alex Candidate",
            email="alex@example.com",
            interview_date="2026-12-01",
            interview_time="14:30",
        ),
        user_id="user-1",
        conversation_id="conversation-booking",
    )

    assert result["booking_status"] == "unavailable"
    assert "unavailable" in result["answer"]


def test_conversational_rag_rejects_invalid_booking_fields(monkeypatch):
    monkeypatch.setattr(query_service, "get_conversation_history", lambda *_: [])
    monkeypatch.setattr(query_service, "get_interview_booking", lambda **_: None)
    monkeypatch.setattr(
        query_service,
        "_extract_interview_booking",
        lambda *_: query_service.InterviewBookingExtraction(
            booking_requested=True,
            name="Alex Candidate",
            email="not-an-email",
            interview_date="not-a-date",
            interview_time="25:90",
        ),
    )
    monkeypatch.setattr(
        query_service,
        "save_interview_booking",
        lambda **_: pytest.fail("invalid booking must not be stored"),
    )
    monkeypatch.setattr(query_service, "save_conversation_turn", lambda *_: None)

    result = query_service.answer_conversational_question(
        "Book an interview.",
        "user-1",
        "conversation-booking",
    )

    assert result["booking_status"] == "collecting_details"
    assert "email" in result["answer"]
    assert "YYYY-MM-DD" in result["answer"]
    assert "HH:MM" in result["answer"]


def test_query_api_does_not_require_admin_key(monkeypatch):
    from src import app as app_module

    monkeypatch.setattr(
        app_module,
        "answer_question",
        lambda question, user_id: {"answer": "Test answer", "sources": []},
    )
    client = TestClient(app_module.app)

    response = client.post(
        "/query",
        json={"question": "Question", "user_id": "user-1"},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Test answer"


def test_query_api_returns_answer_and_sources(monkeypatch):
    from src import app as app_module

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
        json={"question": " What is PalmAI? ", "user_id": " user-1 "},
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Answer for What is PalmAI? (user-1)",
        "sources": [],
    }


def test_query_api_reports_missing_model_configuration(monkeypatch):
    from src import app as app_module

    def missing_configuration(**_):
        raise RuntimeError("GROQ_API_KEY is not configured.")

    monkeypatch.setattr(app_module, "answer_question", missing_configuration)
    client = TestClient(app_module.app)

    response = client.post(
        "/query",
        json={"question": "Question", "user_id": "user-1"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "GROQ_API_KEY is not configured."


def test_query_api_validates_non_whitespace_question(monkeypatch):
    from src import app as app_module

    client = TestClient(app_module.app)

    response = client.post(
        "/query",
        json={"question": "   ", "user_id": "user-1"},
    )

    assert response.status_code == 422


def test_conversational_rag_is_a_separate_endpoint(monkeypatch):
    from src import app as app_module

    calls = []

    def fake_conversational_service(question, user_id, conversation_id):
        calls.append((question, user_id, conversation_id))
        return {
            "answer": "Follow-up answer.",
            "sources": [],
            "conversation_id": conversation_id or "generated-id",
        }

    monkeypatch.setattr(
        app_module,
        "answer_conversational_question",
        fake_conversational_service,
    )
    client = TestClient(app_module.app)

    response = client.post(
        "/conversational-rag",
        json={
            "question": " What about that? ",
            "user_id": " user-1 ",
            "conversation_id": " conversation-1 ",
        },
    )
    stateless_response = client.post(
        "/query",
        json={"question": "What about that?", "user_id": "user-1"},
    )

    assert response.status_code == 200
    assert response.json()["conversation_id"] == "conversation-1"
    assert calls == [("What about that?", "user-1", "conversation-1")]
    assert stateless_response.status_code == 200
    assert "conversation_id" not in stateless_response.json()
