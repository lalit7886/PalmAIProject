from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator
from src.core.config import get_settings
from pathlib import Path
from src.constant import Valid_file_type
from src.services.data_injestion_service import add_document_to_rag
from src.services.query_service import (
    answer_conversational_question,
    answer_question,
)
from uuid import uuid4

app = FastAPI()
settings = get_settings()


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    user_id: str = Field(min_length=1, max_length=255)

    @field_validator("question", "user_id")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty or whitespace")
        return value


class ConversationalQueryRequest(QueryRequest):
    conversation_id: str | None = Field(default=None, min_length=1, max_length=255)

    @field_validator("conversation_id")
    @classmethod
    def strip_conversation_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("must not be empty or whitespace")
        return value


@app.get("/")
def home():
    return {
        "message": "Welcome to the RAG API.",
        "app_name": settings.app_name,
    }

@app.post("/documents/ingest")
@app.post("/data_injestion", include_in_schema=False)
async def ingest(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    strategy: str = Form("RecursiveCharacterTextSplitter"),
):
    original_filename = Path((file.filename or "").replace("\\", "/")).name
    suffix = Path(original_filename).suffix.lower()

    if not original_filename or suffix not in Valid_file_type:
        raise HTTPException(
            status_code=400,
            detail=f"Supported: {', '.join(sorted(Valid_file_type))}",
        )
    if not user_id.strip():
        raise HTTPException(status_code=400, detail="user_id must not be empty")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    content = await file.read(settings.max_upload_size_bytes + 1)
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(content) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                "Uploaded file exceeds the "
                f"{settings.max_upload_size_bytes}-byte limit"
            ),
        )

    dest = upload_dir / f"{uuid4().hex}{suffix}"
    try:
        dest.write_bytes(content)
        result = await run_in_threadpool(
            add_document_to_rag,
            file_path=str(dest),
            strategy=strategy,
            thread_id=str(uuid4()),
            user_id=user_id,
            original_filename=original_filename,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        dest.unlink(missing_ok=True)

    return {
        "message": "Document indexed",
        "file": original_filename,
        **result,
    }


@app.post("/query")
async def query(
    request: QueryRequest,
):
    try:
        return await run_in_threadpool(
            answer_question,
            question=request.question,
            user_id=request.user_id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/conversational-rag")
async def conversational_rag(
    request: ConversationalQueryRequest,
):
    try:
        return await run_in_threadpool(
            answer_conversational_question,
            question=request.question,
            user_id=request.user_id,
            conversation_id=request.conversation_id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc