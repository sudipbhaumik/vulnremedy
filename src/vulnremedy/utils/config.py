from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # Embedding Configuration
    embedding_model: str = Field(
        default="nomic-embed-text",
        description="Ollama embedding model name"
    )
    embedding_batch_size: int = Field(
        default=10,
        description="Number of chunks to embed per batch"
    )
    embedding_cache_dir: str = Field(
        default="./data/embeddings_cache",
        description="Directory to cache embeddings"
    )
    embedding_max_length: int = Field(
        default=8000,
        description="Maximum characters per text for embedding (truncate longer texts)"
    )

    # Retrieval Configuration
    retrieval_top_k: int = Field(
        default=10,
        description="Number of chunks to retrieve from vector search"
    )
    retrieval_batch_size: int = Field(
        default=100,
        description="Batch size for adding chunks to vector store"
    )
    retrieval_metadata_limit: int = Field(
        default=100,
        description="Maximum results for metadata-only searches"
    )
    retrieval_stats_sample_size: int = Field(
        default=100,
        description="Number of documents to sample for stats"
    )
    
    # Hybrid Retrieval Weights
    retrieval_vector_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for vector search in hybrid retrieval (0.0-1.0)"
    )
    retrieval_keyword_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for keyword search in hybrid retrieval (0.0-1.0)"
    )

    # Reranker Configuration
    reranker_relevance_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score to keep results (0.0-1.0)"
    )

    # GitHub Configuration
    github_api_base_url: str = Field(
        default="https://api.github.com",
        description="GitHub API base URL"
    )
    github_token: str = Field(
        default="",
        description="GitHub personal access token for API access"
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
    chroma_collection_name: str = Field(default="vulremedy_knowledge")

    # MLflow
    mlflow_tracking_uri: str = Field(default="http://localhost:5001")

    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")

    # Prompt templates
    prompt_templates_dir: str = Field(
        default="prompts",
        description="Directory containing prompt template files"
    )

    # Impact Assessor Guardrails
    guardrails_max_reasoning_chars: int = Field(
        default=1000,
        description="Maximum characters for LLM reasoning field before truncation"
    )
    guardrails_max_business_impact_chars: int = Field(
        default=500,
        description="Maximum characters for LLM business_impact field before truncation"
    )
    guardrails_max_chunk_chars: int = Field(
        default=2000,
        description="Maximum characters per RAG chunk before truncation"
    )
    guardrails_circuit_breaker_max_failures: int = Field(
        default=3,
        description="Consecutive LLM failures before circuit breaker opens"
    )
    guardrails_rate_limit_interval_seconds: float = Field(
        default=0.5,
        ge=0.0,
        description="Minimum seconds between consecutive LLM calls"
    )

    # Remediation Planner Guardrails
    remediation_max_field_chars: int = Field(
        default=2000,
        description="Maximum characters for LLM strategy text fields before truncation"
    )
    remediation_max_list_items: int = Field(
        default=10,
        description="Maximum items in LLM-generated lists (breaking_changes, testing_plan, etc.)"
    )
    remediation_circuit_breaker_max_failures: int = Field(
        default=3,
        description="Consecutive LLM failures before remediation circuit breaker opens"
    )
    remediation_rate_limit_interval_seconds: float = Field(
        default=0.5,
        ge=0.0,
        description="Minimum seconds between consecutive remediation LLM calls"
    )

    # PR Creator Guardrails
    pr_creator_max_title_chars: int = Field(
        default=200,
        description="Maximum characters for LLM-generated PR title before truncation"
    )
    pr_creator_max_body_chars: int = Field(
        default=5000,
        description="Maximum characters for LLM-generated PR body before truncation"
    )
    pr_creator_circuit_breaker_max_failures: int = Field(
        default=3,
        description="Consecutive LLM failures before PR creator circuit breaker opens"
    )
    pr_creator_rate_limit_interval_seconds: float = Field(
        default=0.5,
        ge=0.0,
        description="Minimum seconds between consecutive PR description LLM calls"
    )

    # Environment
    environment: str = Field(default="development")

    @property
    def is_development(self) -> bool:
        return self.environment == "development"

    def load_prompt(self, filename: str) -> str:
        """Load a prompt template from the prompt_templates_dir.

        Args:
            filename: File name within the prompts directory (e.g. "impact_assessor_exploitability.txt").

        Returns:
            Raw template string with {placeholder} variables.

        Raises:
            FileNotFoundError: If the template file does not exist.
        """
        path = Path(self.prompt_templates_dir) / filename
        return path.read_text(encoding="utf-8")


# Singleton — import this everywhere, never instantiate Settings directly
settings = Settings()