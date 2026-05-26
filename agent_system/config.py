import os

class Config:
    # --- LLM Provider Settings ---
    LLM_MODEL = os.getenv("LLM_MODEL", "groq/llama-3.1-8b-instant")
    
    # --- RAG Settings ---
    CHUNK_SIZE = 500
    CHUNK_OVERLAP_DEFAULT = 50
    CHUNK_OVERLAP_BOUNDARY = 10  # Reduced overlap near semantic boundaries
    RAG_TOP_K = 3
    
    # --- Memory Settings ---
    MAX_LONG_TERM_CONTEXT_LIMIT = 5
    MAX_TOKEN_SUMMARIZATION_THRESHOLD = 2000
    
    # --- Tool Execution & Queue Settings ---
    MAX_RETRIES = 2
    TOOL_TIMEOUT_SECONDS = 3600  # 1 hour
    
    # --- Database Settings ---
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/agent_db")

    # --- Telemetry & Hardware Thresholds ---
    MIN_RAM_REQUIRED_GB = 1.0
    MIN_DISK_REQUIRED_GB = 2.0
    MIN_VRAM_REQUIRED_GB = 2.0
    
    # --- Security Settings ---
    MAX_UNCOMPRESSED_SIZE = 5 * 1024 * 1024 * 1024  # 5GB
    MAX_COMPRESSION_RATIO = 100
    ALLOWED_EXTENSIONS = {
        '.csv', '.json', '.yaml', '.yml', '.txt',
        '.wav', '.mp3', '.pdf', '.png', '.jpg', '.jpeg',
        '.zip', '.tar', '.gz'
    }

    # --- Epoch Constraints (Strict User Requirements) ---
    YOLO_EPOCHS = {
        "small_mb": 50,    # < 50MB
        "small_range": (80, 100),
        "medium_mb": 500,  # 50 - 500MB
        "medium_range": (50, 80),
        "large_range": (30, 50), # > 500MB
        "absolute_min": 10,
        "absolute_max": 300
    }
    
    WHISPER_EPOCHS = {
        "small_mb": 100,
        "small_range": (50, 100),
        "medium_mb": 1024,
        "medium_range": (20, 70),
        "large_range": (10, 25),
        "absolute_min": 1,
        "absolute_max": 100
    }

    VISION_EPOCHS = {
        "small_mb": 100,
        "small_range": (50, 100),
        "medium_mb": 1024,
        "medium_range": (20, 70),
        "large_range": (10, 25),
        "absolute_min": 1,
        "absolute_max": 150
    }

    LLM_EPOCHS = {
        "small_mb": 100,
        "small_range": (50, 100),
        "medium_mb": 1024,
        "medium_range": (20, 70),
        "large_range": (10, 25),
        "absolute_min": 1,
        "absolute_max": 100
    }
