import sqlite3

import pytest

from src.services import audit


@pytest.fixture
def audit_db(tmp_path, monkeypatch):
    db_path = tmp_path / "audit.db"
    monkeypatch.setattr(audit.settings, "audit_db_path", str(db_path))
    audit.init_document_table()
    return db_path


def make_metadata(document_id: str, user_id: str, *, content_hash: str = "hash-1") -> dict:
    return {
        "document_id": document_id,
        "user_id": user_id,
        "thread_id": "thread-1",
        "filename": "sample.txt",
        "file_type": ".txt",
        "source": "/tmp/sample.txt",
        "content_hash": content_hash,
        "version": 1,
        "total_chunks": 3,
        "chunking_strategy": "RecursiveCharacterTextSplitter",
        "status": "processing",
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T00:00:00+00:00",
    }


def test_init_document_table_creates_documents_table(audit_db):
    with sqlite3.connect(audit_db) as connection:
        row = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='documents'"
        ).fetchone()

    assert row is not None
    assert row[0] == "documents"


def test_save_document_metadata_and_fetch_by_hash(audit_db):
    metadata = make_metadata("doc-1", "user-1", content_hash="hash-1")

    audit.save_document_metadata(metadata)

    document = audit.get_document_by_hash("hash-1", "user-1")
    assert document is not None
    assert document["filename"] == "sample.txt"
    assert document["total_chunks"] == 3
    assert document["status"] == "processing"


def test_update_document_status_changes_status(audit_db):
    metadata = make_metadata("doc-2", "user-2", content_hash="hash-2")
    audit.save_document_metadata(metadata)

    audit.update_document_status("doc-2", "completed", "user-2")

    document = audit.get_document_by_id("doc-2", "user-2")
    assert document is not None
    assert document["status"] == "completed"


def test_update_document_status_rejects_invalid_status(audit_db):
    metadata = make_metadata("doc-3", "user-3", content_hash="hash-3")
    audit.save_document_metadata(metadata)

    with pytest.raises(ValueError, match="Invalid status"):
        audit.update_document_status("doc-3", "unknown-status", "user-3")


def test_save_document_metadata_requires_all_fields(audit_db):
    metadata = make_metadata("doc-4", "user-4", content_hash="hash-4")
    metadata.pop("total_chunks")

    with pytest.raises(ValueError, match="Missing metadata fields"):
        audit.save_document_metadata(metadata)


def test_save_document_metadata_is_idempotent_for_same_user_and_hash(audit_db):
    metadata = make_metadata("doc-5", "user-5", content_hash="hash-5")

    first_document_id = audit.save_document_metadata(metadata)
    second_document_id = audit.save_document_metadata({**metadata, "document_id": "doc-6"})

    assert first_document_id == "doc-5"
    assert second_document_id == "doc-5"
    assert audit.get_document_by_hash("hash-5", "user-5")["document_id"] == "doc-5"


def test_save_interview_booking_persists_booking(audit_db):
    booking, created = audit.save_interview_booking(
        user_id="user-1",
        conversation_id="conversation-1",
        name="Alex Candidate",
        email="alex@example.com",
        interview_date="2026-12-01",
        interview_time="14:30",
        timezone_name="Asia/Kathmandu",
        calendar_event_id="event-1",
        calendar_event_url="https://calendar.google.com/event?eid=event-1",
    )

    assert created is True
    assert booking.booking_id
    assert booking.name == "Alex Candidate"
    assert booking.email == "alex@example.com"
    assert booking.interview_date == "2026-12-01"
    assert booking.interview_time == "14:30"


def test_save_interview_booking_is_idempotent_per_conversation(audit_db):
    booking_values = {
        "user_id": "user-1",
        "conversation_id": "conversation-1",
        "name": "Alex Candidate",
        "email": "alex@example.com",
        "interview_date": "2026-12-01",
        "interview_time": "14:30",
        "timezone_name": "Asia/Kathmandu",
        "calendar_event_id": "event-1",
        "calendar_event_url": "https://calendar.google.com/event?eid=event-1",
    }

    first, first_created = audit.save_interview_booking(**booking_values)
    second, second_created = audit.save_interview_booking(
        **{**booking_values, "interview_date": "2026-12-02"}
    )

    assert first_created is True
    assert second_created is False
    assert second.booking_id == first.booking_id
    assert second.interview_date == "2026-12-01"


def test_get_interview_booking_migrates_existing_table(audit_db):
    with sqlite3.connect(audit_db) as connection:
        connection.execute(
            """
            CREATE TABLE interview_bookings (
                booking_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                interview_date TEXT NOT NULL,
                interview_time TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(user_id, conversation_id)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO interview_bookings (
                booking_id, user_id, conversation_id, name, email,
                interview_date, interview_time, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "booking-legacy",
                "user-legacy",
                "conversation-legacy",
                "Legacy Candidate",
                "legacy@example.com",
                "2026-12-01",
                "14:30",
                "2026-11-13T00:00:00+00:00",
            ),
        )

    booking = audit.get_interview_booking(
        user_id="user-legacy",
        conversation_id="conversation-legacy",
    )

    assert booking is not None
    assert booking.booking_id == "booking-legacy"
    assert booking.timezone == "Asia/Kathmandu"
    assert booking.calendar_event_id is None