from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from src.core.config import get_settings
from pathlib import Path
from src.constant import Valid_file_type
from src.services.data_injestion_service import add_document_to_rag
from uuid import uuid4

app = FastAPI()
settings = get_settings()


@app.post("/data_injestion")
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
    result = add_document_to_rag(
        file_path=str(dest),
        strategy=strategy,
        thread_id=str(uuid4()),
        user_id=user_id,
    )

    return {
        "message": "Document indexed",
        "file": dest.name,
        **result,
    }