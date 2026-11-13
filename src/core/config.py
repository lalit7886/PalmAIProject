from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    app_name: str = "palm-ai"
    app_env: str = "development"
    groq_api_key : str = ""
    pinecone_api_key : str = ""
    pinecone_index_name: str = "palmai"
    pinecone_namespace: str = "RAG_"
    redis_url: str = "redis://localhost:6379/0"
    conversation_history_limit: int = 20
    conversation_ttl_seconds: int = 86400
    google_client_id: str = ""
    google_client_secret: str = ""
    google_refresh_token: str = ""
    google_calendar_id: str = "primary"
    calendar_timezone: str = "Asia/Kathmandu"
    interview_slot_minutes: int = 30
    max_upload_size_bytes: int = 10 * 1024 * 1024
    embedding_model : str = "BAAI/bge-m3"
    llm_model: str = "openai/gpt-oss-20b"
    top_k: int = 4
    max_retries : int =1
    audit_db_path : str = str(BASE_DIR / "data" / "audit.db")
    upload_dir : str = str(BASE_DIR / "uploads")
    data_dir : str = str(BASE_DIR / "data" / "sample_kb")
    
    
    
    model_config = SettingsConfigDict(env_file= str(BASE_DIR / ".env"), extra = "ignore")
    
@lru_cache
def get_settings()-> Settings:
    return Settings()
    