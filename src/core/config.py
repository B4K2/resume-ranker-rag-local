from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Resume Ranker AI"
    
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "temp_uploads"
    
    MAX_UPLOAD_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB limit for Zip
    MAX_EXTRACTED_SIZE_BYTES: int = 500 * 1024 * 1024 # 500 MB limit extracted
    MAX_FILE_COUNT: int = 500 # Max files inside zip
    

    EMBEDDING_MODEL_ID: str = "Qwen/Qwen3-Embedding-0.6B" 
    RERANKER_MODEL_ID: str = "Qwen/Qwen3-Reranker-0.6B"
    LLM_MODEL_ID: str = "Qwen/Qwen3-0.6B"
    

    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()

settings.UPLOAD_DIR.mkdir(exist_ok=True)