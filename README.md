# PalmAI backend

## Setup

Create and activate a virtual environment, then install the application and test
dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

Set these values in `.env` before using the APIs:

- `GROQ_API_KEY`
- `PINECONE_API_KEY`
- `ADMIN_API_KEY` (replace the development default)

Optional settings include `PINECONE_INDEX_NAME`, `PINECONE_NAMESPACE`,
`EMBEDDING_MODEL`, `LLM_MODEL`, `TOP_K`, `AUDIT_DB_PATH`, and `UPLOAD_DIR`.

Run the API server:

```bash
uvicorn src.app:app --reload
```

Interactive API documentation is available at `/docs`.

## APIs

Both APIs require the `x-admin-key` header.

### Upload and index a document

`POST /documents/ingest` accepts multipart form data with:

- `file`: a `.pdf` or `.txt` document
- `user_id`: the document owner
- `strategy`: `RecursiveCharacterTextSplitter` or `SemanticChunker`

### Ask a question

`POST /query` accepts JSON containing `user_id` and `question`. It searches
only that user's indexed chunks and returns an answer with cited source chunks.

```json
{
  "user_id": "user-123",
  "question": "What does the document say about deployment?"
}
```

Run the unit and API tests with:

```bash
python -m pytest -q
```
