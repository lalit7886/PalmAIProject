import pytest

from langchain_core.documents import Document

from src.services import data_injestion_service as ingestion


class FakeSplitter:
    def __init__(self, chunks):
        self.chunks = chunks

    def split_text(self, text):
        return self.chunks


def run_ingestion_with_strategy(monkeypatch, tmp_path, strategy):
    path = tmp_path / "staged-upload.txt"
    path.write_text("A document body for chunking.", encoding="utf-8")
    saved_metadata = []
    added_documents = []
    updated_status = []

    monkeypatch.setattr(ingestion, "init_document_table", lambda: None)
    monkeypatch.setattr(ingestion, "get_document_by_hash", lambda *_: None)
    monkeypatch.setattr(
        ingestion,
        "save_document_metadata",
        lambda metadata: saved_metadata.append(metadata),
    )
    monkeypatch.setattr(
        ingestion,
        "update_document_status",
        lambda *args: updated_status.append(args),
    )

    class SplitterFactory:
        def __init__(self, **kwargs):
            assert kwargs == {"chunk_size": 1000, "chunk_overlap": 100}

        def split_text(self, text):
            return ["chunk one", "chunk two"]

    class SemanticSplitterFactory:
        def __init__(self, *, embeddings):
            assert embeddings == "test-embeddings"

        def split_text(self, text):
            return ["semantic chunk"]

    monkeypatch.setattr(ingestion, "RecursiveCharacterTextSplitter", SplitterFactory)
    monkeypatch.setattr(ingestion, "SemanticChunker", SemanticSplitterFactory)

    import src.services.veoctor_store as vector_store

    monkeypatch.setattr(
        vector_store,
        "add_documents",
        lambda docs: added_documents.extend(docs),
    )
    monkeypatch.setattr(vector_store, "get_embdeddings", lambda: "test-embeddings")

    result = ingestion.add_document_to_rag(
        file_path=str(path),
        strategy=strategy,
        thread_id="thread-1",
        user_id="user-1",
        original_filename="resume.txt",
    )
    return result, saved_metadata, added_documents, updated_status


def test_ingestion_supports_recursive_chunking(monkeypatch, tmp_path):
    result, metadata, documents, statuses = run_ingestion_with_strategy(
        monkeypatch,
        tmp_path,
        "RecursiveCharacterTextSplitter",
    )

    assert result["total_chunks"] == 2
    assert metadata[0]["filename"] == "resume.txt"
    assert metadata[0]["chunking_strategy"] == "RecursiveCharacterTextSplitter"
    assert [document.page_content for document in documents] == [
        "chunk one",
        "chunk two",
    ]
    assert all(isinstance(document, Document) for document in documents)
    assert statuses[0][1] == "completed"


def test_ingestion_supports_semantic_chunking(monkeypatch, tmp_path):
    result, metadata, documents, statuses = run_ingestion_with_strategy(
        monkeypatch,
        tmp_path,
        "SemanticChunker",
    )

    assert result["total_chunks"] == 1
    assert metadata[0]["filename"] == "resume.txt"
    assert metadata[0]["chunking_strategy"] == "SemanticChunker"
    assert [document.page_content for document in documents] == ["semantic chunk"]
    assert statuses[0][1] == "completed"


def test_ingestion_rejects_unknown_chunking_strategy(monkeypatch, tmp_path):
    path = tmp_path / "document.txt"
    path.write_text("document text", encoding="utf-8")
    monkeypatch.setattr(ingestion, "init_document_table", lambda: None)
    monkeypatch.setattr(ingestion, "get_document_by_hash", lambda *_: None)

    with pytest.raises(ValueError, match="Unsupported strategy"):
        ingestion.add_document_to_rag(
            file_path=str(path),
            strategy="unsupported",
            thread_id="thread-1",
            user_id="user-1",
        )
