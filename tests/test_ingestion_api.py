from pathlib import Path

from fastapi.testclient import TestClient


def test_home_returns_app_name():
    from src import app as app_module

    response = TestClient(app_module.app).get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the RAG API.",
        "app_name": app_module.settings.app_name,
    }


def test_ingestion_api_accepts_text_upload(monkeypatch, tmp_path):
    from src import app as app_module

    monkeypatch.setattr(app_module.settings, "upload_dir", str(tmp_path))
    calls = {}

    def fake_ingestion_service(**kwargs):
        calls.update(kwargs)
        assert Path(kwargs["file_path"]).exists()
        return {
            "document_id": "doc-1",
            "total_chunks": 1,
            "content_hash": "hash-1",
        }

    monkeypatch.setattr(app_module, "add_document_to_rag", fake_ingestion_service)
    client = TestClient(app_module.app)

    response = client.post(
        "/documents/ingest",
        data={"user_id": "user-1", "strategy": "RecursiveCharacterTextSplitter"},
        files={"file": ("sample.txt", b"sample content", "text/plain")},
    )

    assert response.status_code == 200
    assert response.json()["file"] == "sample.txt"
    assert response.json()["document_id"] == "doc-1"
    assert calls["user_id"] == "user-1"
    assert calls["strategy"] == "RecursiveCharacterTextSplitter"
    assert calls["original_filename"] == "sample.txt"
    assert not list(tmp_path.iterdir())


def test_ingestion_api_uses_unique_temp_path_for_same_filename(monkeypatch, tmp_path):
    from src import app as app_module

    monkeypatch.setattr(app_module.settings, "upload_dir", str(tmp_path))
    paths = []

    def fake_ingestion_service(**kwargs):
        paths.append(kwargs["file_path"])
        return {"document_id": f"doc-{len(paths)}", "total_chunks": 1}

    monkeypatch.setattr(app_module, "add_document_to_rag", fake_ingestion_service)
    client = TestClient(app_module.app)

    for content in (b"first upload", b"second upload"):
        response = client.post(
            "/documents/ingest",
            data={"user_id": "user-1"},
            files={"file": ("sample.txt", content, "text/plain")},
        )
        assert response.status_code == 200

    assert len(set(paths)) == 2
    assert not list(tmp_path.iterdir())


def test_ingestion_api_rejects_unsupported_file_type(monkeypatch, tmp_path):
    from src import app as app_module

    monkeypatch.setattr(app_module.settings, "upload_dir", str(tmp_path))
    client = TestClient(app_module.app)

    response = client.post(
        "/documents/ingest",
        data={"user_id": "user-1"},
        files={"file": ("sample.csv", b"sample content", "text/csv")},
    )

    assert response.status_code == 400
    assert "Supported:" in response.json()["detail"]


def test_ingestion_api_rejects_oversized_file(monkeypatch, tmp_path):
    from src import app as app_module

    monkeypatch.setattr(app_module.settings, "upload_dir", str(tmp_path))
    monkeypatch.setattr(app_module.settings, "max_upload_size_bytes", 4)
    client = TestClient(app_module.app)

    response = client.post(
        "/documents/ingest",
        data={"user_id": "user-1"},
        files={"file": ("sample.txt", b"12345", "text/plain")},
    )

    assert response.status_code == 413
    assert not list(tmp_path.iterdir())
