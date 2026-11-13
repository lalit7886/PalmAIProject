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
- `REDIS_URL` (defaults to `redis://localhost:6379/0`)
- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`
- `GOOGLE_REFRESH_TOKEN`

Start a local Redis server before using conversational RAG:

```bash
redis-server
```

The API currently has no authentication and is intended for local development
only. Do not expose it to an untrusted network.

Optional settings include `PINECONE_INDEX_NAME`, `PINECONE_NAMESPACE`,
`EMBEDDING_MODEL`, `LLM_MODEL`, `TOP_K`, `CONVERSATION_HISTORY_LIMIT`,
`CONVERSATION_TTL_SECONDS`, `GOOGLE_CALENDAR_ID` (defaults to `primary`),
`CALENDAR_TIMEZONE` (defaults to `Asia/Kathmandu`),
`INTERVIEW_SLOT_MINUTES` (defaults to `30`), `MAX_UPLOAD_SIZE_BYTES` (defaults
to 10 MiB), `AUDIT_DB_PATH`, and `UPLOAD_DIR`.

Run the API server:

```bash
.venv/bin/python -m uvicorn src.app:app --reload
```

Interactive API documentation is available at `/docs`.

## Run locally with Docker

Docker Compose runs the API and Redis for local development. Source changes are
picked up automatically; SQLite data, uploads, Redis data, and the Hugging
Face model cache are stored in named volumes.

Put `GROQ_API_KEY` and `PINECONE_API_KEY` in an ignored `.env` file in the
project root. Add `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, and
`GOOGLE_REFRESH_TOKEN` only if calendar booking is enabled. Do not commit this
file. Start the services with:

```bash
docker compose up --build
```

Open `http://localhost:8000/docs` for the API documentation. The API is bound
to `127.0.0.1:8000`; set `APP_PORT` to change the host port. Redis is only
available to the API container and is not exposed on the host. Stop the
services with `docker compose down`; named-volume data remains available the
next time you start them.

## APIs

### Upload and index a document

`POST /documents/ingest` accepts multipart form data with:

- `file`: a `.pdf` or `.txt` document
- `user_id`: the document owner
- `strategy`: `RecursiveCharacterTextSplitter` or `SemanticChunker`

Uploads are size-limited, staged under unique temporary names to avoid
same-filename collisions, and removed after ingestion.

### Ask a question

`POST /query` accepts JSON containing `user_id` and `question`. It searches
only that user's indexed chunks and returns an answer with cited source chunks.
This endpoint is stateless.

```json
{
  "user_id": "user-123",
  "question": "What does the document say about deployment?"
}
```

### Conversational RAG

`POST /conversational-rag` accepts the same fields as `/query` and an optional
`conversation_id`. The first response returns a conversation ID; include it in
subsequent requests to continue that user's conversation. Recent chat history is
stored in Redis, scoped to the user and conversation, and expires after the
configured TTL (24 hours by default). Redis must be running for this endpoint.
Retrieval and answer generation are implemented directly; no
`RetrievalQAChain` is used.

```json
{
  "user_id": "user-123",
  "conversation_id": "conversation-id-from-previous-response",
  "question": "What about the second option?"
}
```

The same conversational endpoint can collect interview booking details. The LLM
extracts the candidate's name, email, interview date (`YYYY-MM-DD`), and local
time (`HH:MM`) across turns. It asks for missing or invalid fields and saves the
booking to SQLite only when all four are valid. A conversation can create one
booking; repeat booking requests in that conversation return the existing
booking instead of creating duplicates.

Complete bookings are checked against the configured Google Calendar and added
as calendar events only if the requested slot is free. The local SQLite booking
is saved after the event is created; failed database persistence triggers an
attempt to remove the calendar event. The event includes the candidate name and
email in its description, but does not add the candidate as an attendee or send
an email invitation.

To configure Google Calendar:

1. Create a Google Cloud project and enable the Google Calendar API.
2. Configure OAuth consent and create an OAuth client ID and client secret.
3. Use OAuth 2.0 Playground with your own OAuth client credentials to authorize
   the calendar owner and generate a refresh token with these scopes:
   `https://www.googleapis.com/auth/calendar.events` and
   `https://www.googleapis.com/auth/calendar.freebusy`.
4. Put the values in `.env` as `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, and
   `GOOGLE_REFRESH_TOKEN`. Set `GOOGLE_CALENDAR_ID` if you do not use the
   primary calendar. Never commit OAuth credentials or refresh tokens.

Run the unit and API tests with:

```bash
python -m pytest -q
```
