import os
from typing import Dict, Any, Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings."""
    APP_NAME: str = "vllm-judge"
    DEBUG: bool = False
    
    # vLLM Server configuration
    VLLM_API_BASE: str = os.getenv("VLLM_API_BASE", "http://localhost:8080/v1")
    VLLM_API_KEY: Optional[str] = os.getenv("VLLM_API_KEY", "")
    
    # Adapter configuration
    DEFAULT_TIMEOUT: int = 60
    MAX_RETRY_ATTEMPTS: int = 3
    TEMPLATE_STORAGE_PATH: str = os.getenv(
        "TEMPLATE_STORAGE_PATH", 
        os.path.join(os.path.dirname(__file__), "../templates/default_templates.json")
    )
    
    # Async task configuration
    TASK_EXPIRY_SECONDS: int = 3600  # 1 hour

settings = Settings()
