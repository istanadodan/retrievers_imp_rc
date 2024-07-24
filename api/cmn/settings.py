from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):

    class Config:
        env_file = Path(__file__).parents[1] / ".env"
        env_file_encoding = "utf-8"        
        case_sensitive = True
    
    LOCAL_LLM_MODEL: str
    LOCAL_EMBEDDING_MODEL: str
    LOCAL_API_KEY: str
    LOCAL_BASE_URL: str

settings = Settings()