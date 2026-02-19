from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """
    Central configuration for DepShield.
    All values loaded from environment variables / .env file.
    Type-safe, validated at startup — app fails fast if misconfigured.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # RAG Chunking Configuration
    chunk_max_length: int = Field(
        default=1000,
        description="Maximum characters per chunk"
    )
    chunk_min_length: int = Field(
        default=50,
        description="Minimum characters for a chunk to be kept"
    )
    chunk_overlap: int = Field(
        default=100,
        description="Character overlap between chunks for context preservation"
    )

    # LLM
    ollama_base_url: str = Field(default="http://localhost:11434")
    llm_model: str = Field(default="llama3.1:8b")
    embedding_model: str = Field(default="nomic-embed-text")

    # External APIs
    github_token: str = Field(default="")
   

    # NVD API
    nvd_api_base_url: str = Field(
        default="https://services.nvd.nist.gov/rest/json/cves/2.0"
    )
    nvd_api_key: str = Field(default="")

    # OSV API
    osv_api_base_url: str = Field(
        default="https://api.osv.dev/v1"
    )

    # Vector Store
    chroma_persist_dir: str = Field(default="./data/vector_store")
    chroma_collection_name: str = Field(default="depshield_knowledge")

    # MLflow
    mlflow_tracking_uri: str = Field(default="http://localhost:5001")

    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")

    # Environment
    environment: str = Field(default="development")

    @property
    def is_development(self) -> bool:
        return self.environment == "development"


# Singleton — import this everywhere, never instantiate Settings directly
settings = Settings()