
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from src.core.logging import logging
from src.core.config import get_settings

settings = get_settings()


@dataclass(frozen=True)
class InterviewBooking:
    booking_id: str
    user_id: str
    conversation_id: str
    name: str
    email: str
    interview_date: str
    interview_time: str
    created_at: str
    timezone: str = "Asia/Kathmandu"
    calendar_event_id: str | None = None
    calendar_event_url: str | None = None


@contextmanager
def get_db_connection():
    db_path = Path(settings.audit_db_path).resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"SQLite database location: {db_path}")

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row

    try:
        yield con
        con.commit()

    except Exception:
        con.rollback()
        raise

    finally:
        con.close()


def init_document_table():
    with get_db_connection() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                source TEXT,
                content_hash TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                total_chunks INTEGER DEFAULT 0,
                chunking_strategy TEXT,
                status TEXT DEFAULT 'processing',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(user_id, content_hash)
            )
        """)

    logging.info("Document audit table initialized")


def init_interview_bookings_table() -> None:
    with get_db_connection() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS interview_bookings (
                booking_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                interview_date TEXT NOT NULL,
                interview_time TEXT NOT NULL,
                timezone TEXT NOT NULL DEFAULT 'Asia/Kathmandu',
                calendar_event_id TEXT,
                calendar_event_url TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(user_id, conversation_id)
            )
            """
        )
        columns = {
            row["name"]
            for row in con.execute("PRAGMA table_info(interview_bookings)")
        }
        migrations = {
            "timezone": (
                "ALTER TABLE interview_bookings ADD COLUMN "
                "timezone TEXT NOT NULL DEFAULT 'Asia/Kathmandu'"
            ),
            "calendar_event_id": (
                "ALTER TABLE interview_bookings ADD COLUMN calendar_event_id TEXT"
            ),
            "calendar_event_url": (
                "ALTER TABLE interview_bookings ADD COLUMN calendar_event_url TEXT"
            ),
        }
        for column, statement in migrations.items():
            if column not in columns:
                con.execute(statement)
    logging.info("Interview booking table initialized")


def get_interview_booking(
    *,
    user_id: str,
    conversation_id: str,
) -> InterviewBooking | None:
    init_interview_bookings_table()
    with get_db_connection() as con:
        row = con.execute(
            """
            SELECT booking_id, user_id, conversation_id, name, email,
                   interview_date, interview_time, created_at, timezone,
                   calendar_event_id, calendar_event_url
            FROM interview_bookings
            WHERE user_id = ? AND conversation_id = ?
            """,
            (user_id, conversation_id),
        ).fetchone()
    return InterviewBooking(**dict(row)) if row is not None else None


def save_interview_booking(
    *,
    user_id: str,
    conversation_id: str,
    name: str,
    email: str,
    interview_date: str,
    interview_time: str,
    timezone_name: str,
    calendar_event_id: str,
    calendar_event_url: str | None,
) -> tuple[InterviewBooking, bool]:
    from uuid import uuid4

    required_values = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "name": name,
        "email": email,
        "interview_date": interview_date,
        "interview_time": interview_time,
    }
    if any(not value.strip() for value in required_values.values()):
        raise ValueError("Interview booking fields must not be empty.")

    init_interview_bookings_table()
    booking_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as con:
        cursor = con.execute(
            """
            INSERT OR IGNORE INTO interview_bookings (
                booking_id, user_id, conversation_id, name, email,
                interview_date, interview_time, timezone, calendar_event_id,
                calendar_event_url, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                booking_id,
                user_id,
                conversation_id,
                name,
                email,
                interview_date,
                interview_time,
                timezone_name,
                calendar_event_id,
                calendar_event_url,
                created_at,
            ),
        )
        created = cursor.rowcount == 1
        row = con.execute(
            """
            SELECT booking_id, user_id, conversation_id, name, email,
                   interview_date, interview_time, created_at, timezone,
                   calendar_event_id, calendar_event_url
            FROM interview_bookings
            WHERE user_id = ? AND conversation_id = ?
            """,
            (user_id, conversation_id),
        ).fetchone()
        if row is None:
            raise RuntimeError("Unable to retrieve the saved interview booking.")

    booking = InterviewBooking(**dict(row))
    logging.info(
        "Interview booking %s for user=%s conversation=%s",
        "created" if created else "already exists",
        user_id,
        conversation_id,
    )
    return booking, created


def save_document_metadata(metadata: dict[str, object]) -> str | None:
    required_fields = [
        "document_id",
        "user_id",
        "thread_id",
        "filename",
        "file_type",
        "source",
        "content_hash",
        "version",
        "total_chunks",
        "chunking_strategy",
        "status",
        "created_at",
        "updated_at",
    ]

    missing_fields = [field for field in required_fields if field not in metadata]

    if missing_fields:
        raise ValueError(f"Missing metadata fields: {missing_fields}")

    with get_db_connection() as con:
        cursor = con.execute("""
            INSERT OR IGNORE INTO documents (
                document_id,
                user_id,
                thread_id,
                filename,
                file_type,
                source,
                content_hash,
                version,
                total_chunks,
                chunking_strategy,
                status,
                created_at,
                updated_at
            )
            VALUES (
                :document_id,
                :user_id,
                :thread_id,
                :filename,
                :file_type,
                :source,
                :content_hash,
                :version,
                :total_chunks,
                :chunking_strategy,
                :status,
                :created_at,
                :updated_at
            )
        """, metadata)

        if cursor.rowcount == 0:
            existing = con.execute(
                """
                SELECT document_id
                FROM documents
                WHERE user_id = ?
                AND content_hash = ?
                """,
                (metadata["user_id"], metadata["content_hash"]),
            ).fetchone()
            if existing is not None:
                logging.info(
                    f"Document metadata already exists for user={metadata['user_id']} hash={metadata['content_hash']}"
                )
                return existing["document_id"]

    logging.info(f"Saved document metadata: {metadata['document_id']}")
    return metadata["document_id"]


def get_document_by_hash(content_hash: str, user_id: str) -> Optional[sqlite3.Row]:
    with get_db_connection() as con:
        return con.execute("""
            SELECT *
            FROM documents
            WHERE content_hash = ?
            AND user_id = ?
        """, (content_hash, user_id)).fetchone()


def get_document_by_id(document_id: str, user_id: str) -> Optional[sqlite3.Row]:
    with get_db_connection() as con:
        return con.execute("""
            SELECT *
            FROM documents
            WHERE document_id = ?
            AND user_id = ?
        """, (document_id, user_id)).fetchone()


def update_document_status(
    document_id: str,
    status: str,
    user_id: Optional[str] = None
) -> None:

    allowed_statuses = {
        "processing",
        "completed",
        "failed",
        "deleting",
        "deleted"
    }

    if status not in allowed_statuses:
        raise ValueError(f"Invalid status: {status}")

    updated_at = datetime.now(timezone.utc).isoformat()

    with get_db_connection() as con:
        if user_id is not None:
            cursor = con.execute("""
                UPDATE documents
                SET status = ?,
                    updated_at = ?
                WHERE document_id = ?
                AND user_id = ?
            """, (status, updated_at, document_id, user_id))

        else:
            cursor = con.execute("""
                UPDATE documents
                SET status = ?,
                    updated_at = ?
                WHERE document_id = ?
            """, (status, updated_at, document_id))

        if cursor.rowcount == 0:
            raise ValueError("Document not found")

    logging.info(f"Document {document_id} status updated to {status}")
