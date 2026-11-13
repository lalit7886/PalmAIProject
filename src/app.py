from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator
from src.core.config import get_settings
from pathlib import Path
from src.constant import Valid_file_type
from src.services.data_injestion_service import add_document_to_rag
from src.services.query_service import answer_question
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


@app.post("/documents/ingest")
@app.post("/data_injestion", include_in_schema=False)
async def ingest(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    strategy: str = Form("RecursiveCharacterTextSplitter"),
    x_admin_key: str = Header(default=""),
):
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="invalid admin key")

    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in Valid_file_type:
        raise HTTPException(
            status_code=400,
            detail=f"Supported: {', '.join(sorted(Valid_file_type))}",
        )

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    dest = upload_dir / Path(file.filename).name

    # Save the uploaded file before loading it
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    dest.write_bytes(content)
    try:
        result = await run_in_threadpool(
            add_document_to_rag,
            file_path=str(dest),
            strategy=strategy,
            thread_id=str(uuid4()),
            user_id=user_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "message": "Document indexed",
        "file": dest.name,
        **result,
    }


@app.post("/query")
async def query(
    request: QueryRequest,
    x_admin_key: str = Header(default=""),
):
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="invalid admin key")

    try:
        return await run_in_threadpool(
            answer_question,
            question=request.question,
            user_id=request.user_id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc