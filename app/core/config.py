from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App
    APP_NAME: str = "SHL RAG Agent"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # LLM
    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.1-flash-lite-preview"
    LLM_TEMPERATURE: float = 0.0

    # Embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_BATCH_SIZE: int = 32

    # Vector Store
    CHROMA_DIR: str = "./chroma_db"
    CHROMA_COLLECTION: str = "shl_products"

    # Data
    SHL_CATALOG_URL: str = (
        "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
    )
    CATALOG_CACHE_PATH: str = "./shl_catalog.json"

    # Retrieval
    FINAL_K: int = 20       # Increased from 10 to 20 to give the LLM more options
    DENSE_K: int = 40
    BM25_K: int = 40
    DENSE_WEIGHT: float = 0.50 # Lowered from 0.65
    BM25_WEIGHT: float = 0.50  # Raised from 0.35 to catch exact acronyms better

    class Config:
        env_file = ".env"
        case_sensitive = False


# ─── lru_cache-free settings loader ──────────────────────────────────────────

_settings_instance = None

def get_settings() -> Settings:
    """Returns a cached instance of the settings using a simple global variable."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance