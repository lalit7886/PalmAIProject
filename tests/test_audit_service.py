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