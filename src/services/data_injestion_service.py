
from langchain_core.documents import Document
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone
import hashlib
from typing import Any, List

from src.services.audit import (
    get_document_by_hash,
    init_document_table,
    save_document_metadata,
    update_document_status,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from src.core.logging import logging
from pypdf import PdfReader


def read_file_text(file_path: str) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = []

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            pages.append(
                f"[PAGE {page_number}]\n{page_text}"
            )

        return "\n".join(pages)

    elif suffix == ".txt":
        return path.read_text(encoding="utf-8")

    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def generate_file_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def add_document_to_rag(
    file_path: str,
    strategy: str,
    thread_id: str,
    user_id: str,
) -> dict[str, Any]:
    path = Path(file_path)
    text = read_file_text(file_path)

    if not text.strip():
        raise ValueError("Document is empty.")

    content_hash = generate_file_hash(text)
    init_document_table()
    existing = get_document_by_hash(content_hash, user_id)
    if existing is not None:
        logging.info(
            f"Skipping duplicate document for user={user_id} and hash={content_hash}: {existing['document_id']}"
        )
        return {
            "document_id": existing["document_id"],
            "total_chunks": existing["total_chunks"],
            "content_hash": content_hash,
            "status": "already_exists",
        }

    if strategy == "RecursiveCharacterTextSplitter":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
    elif strategy == "SemanticChunker":
        from src.services.veoctor_store import get_embdeddings

        text_splitter = SemanticChunker(embeddings=get_embdeddings())
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    chunks = text_splitter.split_text(text)
    if not chunks:
        raise ValueError("Document did not produce any chunks.")

    document_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    docs: List[Document] = []
    for index, chunk in enumerate(chunks):
        chunk_metadata = {
            "document_id": document_id,
            "chunk_id": f"{document_id}_{index}",
            "chunk_index": index,
            "filename": path.name,
            "source": str(path),
            "file_type": path.suffix.lower(),
            "thread_id": thread_id,
            "user_id": user_id,
            "chunking_strategy": strategy,
            "content_hash": content_hash,
            "created_at": created_at,
        }
        docs.append(
            Document(
                page_content=chunk,
                metadata=chunk_metadata
            )
        )

    init_document_table()
    save_document_metadata(metadata={
        "document_id": document_id,
        "user_id": user_id,
        "thread_id": thread_id,
        "filename": path.name,
        "file_type": path.suffix.lower(),
        "source": str(path),
        "content_hash": content_hash,
        "version": 1,
        "total_chunks": len(docs),
        "chunking_strategy": strategy,
        "status": "processing",
        "created_at": created_at,
        "updated_at": created_at,
    })

    try:
        from src.services.veoctor_store import add_documents

        add_documents(docs)
        update_document_status(document_id, "completed", user_id)
    except Exception:
        update_document_status(document_id, "failed", user_id)
        raise

    logging.info(f"Successfully ingested {len(docs)} chunks from {path.name}")

    return {
        "document_id": document_id,
        "total_chunks": len(docs),
        "content_hash": content_hash
    }

if __name__ == "__main__":
    file_path = "/Users/lalitramanmishra/PalmAIProject/src/my.txt"
    strategy = "RecursiveCharacterTextSplitter"
    thread_id = str(uuid4())
    user_id = "local-user"

    result = add_document_to_rag(
        file_path=file_path,
        strategy=strategy,
        thread_id=thread_id,
        user_id=user_id,
    )

    print(result)