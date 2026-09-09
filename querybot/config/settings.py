"""
Configuration settings for QueryBot Political Intelligence AI
"""
from pydantic_settings import BaseSettings
from typing import Optional, List


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Application
    APP_NAME: str = "QueryBot - Political Intelligence AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Database
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/querybot"
    PGVECTOR_DIMENSION: int = 768  # Match your embedding model dimension
    
    # Embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE: int = 32
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    
    # LLM Configuration
    LLM_PROVIDER: str = "openrouter"  # or "local", "openai", "anthropic"
    OPENAI_API_KEY: Optional[str] = None
    OPENROUTER_API_KEY: Optional[str] = None
    LLM_MODEL: str = "meta-llama/llama-3-70b-instruct"
    LLM_TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 2048
    
    # LangGraph Configuration
    GRAPH_MEMORY_TYPE: str = "conversation_buffer"  # or "vectorstore", "entity"
    GRAPH_MAX_ITERATIONS: int = 10
    
    # MCP Servers
    MCP_SERVERS: List[str] = ["database", "pgvector", "external_apis"]
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    
    # Data Ingestion
    DATA_SOURCE_DIRS: List[str] = ["data/elections", "data/surveys", "data/social_media", "data/manifestos"]
    SUPPORTED_FORMATS: List[str] = [".json", ".csv", ".txt", ".pdf"]
    
    # AWS (if deployed on EC2)
    AWS_REGION: Optional[str] = None
    AWS_ACCESS_KEY: Optional[str] = None
    AWS_SECRET_KEY: Optional[str] = None
    
    # Observability
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_HOST: Optional[str] = "https://cloud.langfuse.com"
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_PROJECT: str = "querybot"
    
    # Guardrails
    GUARDRAILS_STRICT_MODE: bool = True
    GUARDRAILS_AUTO_CONFIRM_LOW_RISK: bool = True
    
    # Redis Cache
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL_DEFAULT: int = 3600
    CACHE_SIMILARITY_THRESHOLD: float = 0.85
    CACHE_ENABLED: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
