from fastapi.testclient import TestClient


def test_ingestion_api_accepts_text_upload(monkeypatch, tmp_path):
    from src import app as app_module

    monkeypatch.setattr(app_module.settings, "admin_api_key", "test-admin")
    monkeypatch.setattr(app_module.settings, "upload_dir", str(tmp_path))
    calls = {}

    def fake_ingestion_service(**kwargs):
        calls.update(kwargs)
        return {
            "document_id": "doc-1",
            "total_chunks": 1,
            "content_hash": "hash-1",
        }

    monkeypatch.setattr(app_module, "add_document_to_rag", fake_ingestion_service)
    client = TestClient(app_module.app)

    response = client.post(
        "/documents/ingest",
        headers={"x-admin-key": "test-admin"},
        data={"user_id": "user-1", "strategy": "RecursiveCharacterTextSplitter"},
        files={"file": ("sample.txt", b"sample content", "text/plain")},
    )

    assert response.status_code == 200
    assert response.json()["file"] == "sample.txt"
    assert response.json()["document_id"] == "doc-1"
    assert calls["user_id"] == "user-1"
    assert calls["strategy"] == "RecursiveCharacterTextSplitter"
    assert (tmp_path / "sample.txt").read_bytes() == b"sample content"


def test_ingestion_api_rejects_unsupported_file_type(monkeypatch, tmp_path):
    from src import app as app_module

    monkeypatch.setattr(app_module.settings, "admin_api_key", "test-admin")
    monkeypatch.setattr(app_module.settings, "upload_dir", str(tmp_path))
    client = TestClient(app_module.app)

    response = client.post(
        "/documents/ingest",
        headers={"x-admin-key": "test-admin"},
        data={"user_id": "user-1"},
        files={"file": ("sample.csv", b"sample content", "text/csv")},
    )

    assert response.status_code == 400
    assert "Supported:" in response.json()["detail"]
