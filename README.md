# PalmAI — Document Q&A and Interview Booking API

A local-development backend built with **FastAPI**. Upload PDF or text
documents, ask questions grounded in their contents, continue conversations
across turns, and collect interview booking details through chat.

> **Local development only:** the API has no authentication. Do not expose it
> to the public internet or use it with sensitive documents.

## What it does

| Capability | Implementation |
|---|---|
| Document ingestion | Accepts `.pdf` and `.txt`; extracts text and chunks it with a selectable strategy |
| Chunking | `RecursiveCharacterTextSplitter` or `SemanticChunker` |
| Embeddings and vector search | Hugging Face `BAAI/bge-m3` embeddings stored and searched in Pinecone |
| Document metadata | SQLite |
| Stateless document Q&A | Retrieves the user's relevant chunks and generates an answer with source citations |
| Conversational RAG | Custom retrieval and prompt flow (no `RetrievalQAChain`); Redis stores bounded, expiring chat history |
| Interview booking | LLM-assisted collection of name, email, date, and time; booking records are stored in SQLite |
| Calendar availability | Optional Google Calendar FreeBusy check and event creation for available slots |

This project does **not** use FAISS or Chroma and does not include a UI.
Scanned/image-only PDFs are not OCR-processed.

## Run locally with Docker

### Prerequisites

- Docker Desktop with Docker Compose
- API keys for Groq and Pinecone
- Google Calendar OAuth credentials only if you want calendar-backed bookings

### 1. Configure environment variables

Create a `.env` file in the project root. It is ignored by Git; never commit
secrets.

```dotenv
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key

# Optional: required for calendar-backed interview booking
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
GOOGLE_REFRESH_TOKEN=
```

The local Compose setup provides Redis automatically. The API connects to it
inside the Compose network; you do not need to install or expose Redis on your
host.

### 2. Build and start the API and Redis

From the project directory:

```bash
docker compose up --build
```

Wait until both containers are healthy. Open:

- API documentation: <http://localhost:8000/docs>
- API health/home response: <http://localhost:8000/>

Compose binds the API to `127.0.0.1:8000` by default. To use a different host
port, set `APP_PORT`, for example `APP_PORT=8080 docker compose up --build`.
Source changes under `src/` reload automatically.

Stop the containers with `Ctrl+C`, or from another terminal run:

```bash
docker compose down
```

SQLite data, Redis data, uploads, and the downloaded embedding model are stored
in named Docker volumes and remain after stopping the containers. To remove
those local data volumes as well, run `docker compose down --volumes`.

## API endpoints

### 1. Ingest a document

`POST /documents/ingest` accepts multipart form data:

| Field | Required | Description |
|---|---|---|
| `file` | Yes | A `.pdf` or `.txt` file |
| `user_id` | Yes | Owner identifier used to scope document retrieval |
| `strategy` | No | `RecursiveCharacterTextSplitter` (default) or `SemanticChunker` |

Example:

```bash
curl -X POST http://localhost:8000/documents/ingest \
  -F "file=@./my-document.txt" \
  -F "user_id=local-user" \
  -F "strategy=RecursiveCharacterTextSplitter"
```

On success, the response includes a `document_id`, `total_chunks`, and
`content_hash`. Uploads are size-limited (10 MiB by default), staged under a
unique temporary name, and deleted after processing.

### 2. Ask a question (stateless)

`POST /query` searches the calling `user_id`'s indexed document chunks. It
doesn't save conversation history.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "local-user",
    "question": "What does the document say about deployment?"
  }'
```

The response contains an `answer` and a `sources` array with citation numbers,
document IDs, filenames, chunk indexes, and source text. If no relevant chunks
are found, the API returns a no-context answer rather than calling the LLM.

### 3. Conversational RAG and interview booking

`POST /conversational-rag` uses the same `user_id` and `question` fields as
`/query`, plus an optional `conversation_id`. The first response returns a
conversation ID; send it on later turns to continue that conversation.

```bash
curl -X POST http://localhost:8000/conversational-rag \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "local-user",
    "question": "Summarize the main points"
  }'
```

Redis stores recent turns by user and conversation, with a 24-hour default
expiry. To continue, include the returned ID:

```json
{
  "user_id": "local-user",
  "conversation_id": "conversation-id-from-response",
  "question": "Can you explain the second point?"
}
```

The conversational endpoint can also collect the candidate's **name, email,
interview date (`YYYY-MM-DD`), and local time (`HH:MM`)** over multiple turns.
It asks for missing or invalid details and saves a complete booking to SQLite.
Each user/conversation pair can create one booking; repeated requests return
the existing booking.

When Google Calendar is configured, the API checks the requested slot against
calendar availability before creating an event. The configured timezone
defaults to `Asia/Kathmandu` and slot length to 30 minutes. Calendar events do
not add the candidate as an attendee or send an invitation email.

#### Configure Google Calendar (optional)

1. Enable the Google Calendar API in a Google Cloud project.
2. Configure OAuth consent and an OAuth client.
3. Generate a refresh token for the calendar owner with these scopes:
   - `https://www.googleapis.com/auth/calendar.events`
   - `https://www.googleapis.com/auth/calendar.freebusy`
4. Set `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, and
   `GOOGLE_REFRESH_TOKEN` in `.env`.

Set `GOOGLE_CALENDAR_ID` if the calendar is not `primary`. Never commit OAuth
client credentials or refresh tokens.

## Run tests

The test suite uses mocked external integrations; it does not require live
Pinecone, Groq, Redis, or Google Calendar credentials.

With the project's virtual environment:

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

## Configuration reference

Settings can be provided through environment variables or `.env`. Common
defaults:

| Setting | Default | Purpose |
|---|---|---|
| `PINECONE_INDEX_NAME` | `palmai` | Pinecone index |
| `PINECONE_NAMESPACE` | `RAG_` | Pinecone namespace |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Hugging Face embedding model |
| `LLM_MODEL` | `openai/gpt-oss-20b` | Model name sent to Groq's OpenAI-compatible API |
| `TOP_K` | `4` | Number of chunks retrieved per query |
| `CONVERSATION_HISTORY_LIMIT` | `20` | Maximum Redis history entries |
| `CONVERSATION_TTL_SECONDS` | `86400` | Redis conversation expiry |
| `MAX_UPLOAD_SIZE_BYTES` | `10485760` | Maximum upload size (10 MiB) |
| `GOOGLE_CALENDAR_ID` | `primary` | Calendar to check and book |
| `CALENDAR_TIMEZONE` | `Asia/Kathmandu` | Local timezone for interview slots |
| `INTERVIEW_SLOT_MINUTES` | `30` | Interview event length |
| `APP_PORT` | `8000` | Host port for Docker Compose |

## Project layout

```text
src/
  app.py                         FastAPI routes and request validation
  core/config.py                 Environment-based settings
  services/
    data_injestion_service.py    File parsing, chunking, and ingestion workflow
    veoctor_store.py             Embeddings and Pinecone integration
    query_service.py             Custom RAG, conversation, and booking flow
    conversation_store.py        Redis chat history
    audit.py                     SQLite document metadata and bookings
    calendar_service.py          Google Calendar availability and events
tests/                           Unit and API tests
Dockerfile                       Local API image
compose.yaml                     Local API and Redis services
```
