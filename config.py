"""
Configuration settings for the multi-agent AI system.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"
TESTS_DIR = BASE_DIR / "tests"

# Ensure directories exist
UPLOADS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
TESTS_DIR.mkdir(exist_ok=True)

# Database configuration
DATABASE_URL = DATA_DIR / "database.db"

# Runtime mode configuration
RUNTIME_CONFIG = {
    "mode": "auto",  # "cli", "ui", or "auto"
    "cli_output_format": "pretty",  # "pretty", "json", "minimal"
    "ui_auto_refresh": True,
    "enable_streamlit_warnings": False,  # Suppress Streamlit warnings in CLI mode
}

# LLM Configuration
LLM_CONFIG = {
    "model_name": os.getenv("LLM_MODEL_NAME", "meta-llama/llama-4-maverick:free"),
    "api_base": os.getenv("LLM_API_BASE", "https://openrouter.ai/api/v1"),  # OpenRouter API
    # "api_key": os.getenv("LLM_API_KEY", "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"),
    "api_key": os.getenv("LLM_API_KEY", "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"),
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.1")),
    "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "2048")),
    "timeout": int(os.getenv("LLM_TIMEOUT", "30")),
}

# Agent configuration
AGENT_CONFIG = {
    "classifier": {
        "confidence_threshold": 0.7,
        "max_retries": 3,
        "fallback_enabled": True,  # Enable rule-based fallback in CLI mode
    },
    "email": {
        "sender_extraction_patterns": [
            r"From:\s*(.+)",
            r"from:\s*(.+)",
            r"Sender:\s*(.+)",
        ],
        "urgency_keywords": ["urgent", "asap", "immediate", "critical", "emergency"],
        "cli_verbose": True,  # Show detailed extraction in CLI mode
    },
    "json": {
        "max_size_mb": 10,
        "validation_strict": True,
        "cli_show_schema": True,  # Show schema validation details in CLI
    },
    "pdf": {
        "max_pages": 50,
        "text_extraction_method": "pdfplumber",  # or "pdfminer"
        "cli_page_preview": 3,  # Show first N pages in CLI mode
    },
}

# File handling configuration
FILE_CONFIG = {
    "max_file_size_mb": 25,
    "allowed_extensions": {
        "pdf": [".pdf"],
        "json": [".json"],
        "email": [".txt", ".eml", ".msg"],
    },
    "upload_timeout": 30,
    "cli_file_validation": True,  # Strict validation in CLI mode
}

# UI configuration
UI_CONFIG = {
    "page_title": "Multi-Agent AI Document Processor",
    "page_icon": "🤖",
    "layout": "wide",
    "sidebar_width": 300,
    "disable_in_cli": True,  # Don't load UI components in CLI mode
}

# CLI-specific configuration
CLI_CONFIG = {
    "colors": {
        "success": "\033[92m",
        "error": "\033[91m",
        "warning": "\033[93m",
        "info": "\033[94m",
        "reset": "\033[0m",
    },
    "progress_bar": True,
    "verbose_output": False,  # Can be overridden with --verbose flag
    "output_file": None,  # Optional output file for results
}

# Intent classification categories
INTENT_CATEGORIES = [
    "RFQ",           # Request for Quote
    "Invoice",       # Invoice processing
    "Complaint",     # Customer complaint
    "Regulation",    # Regulatory document
    "Contract",      # Contract analysis
    "Email",         # General email
    "Technical",     # Technical documentation
    "Financial",     # Financial document
    "Legal",         # Legal document
    "Other",         # Fallback category
]

# Logging configuration
LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": BASE_DIR / "app.log",
    "cli_level": "WARNING",  # Less verbose logging in CLI mode
    "ui_level": "INFO",      # More detailed logging in UI mode
}

# Development settings
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
MOCK_LLM = os.getenv("MOCK_LLM", "False").lower() == "true"  # For testing without LLM

# SQLite settings
SQLITE_CONFIG = {
    "check_same_thread": False,
    "timeout": 20,
    "isolation_level": None,  # Autocommit mode
}

def setup_logging():
    """Setup logging with mode-specific configuration"""
    import logging
    
    # Determine log level based on mode
    log_level = LOGGING_CONFIG["level"]
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=LOGGING_CONFIG["format"],
        handlers=[
            logging.FileHandler(LOGGING_CONFIG["file"]),
            logging.StreamHandler()
        ]
    )
    
    # Suppress Streamlit warnings if not in UI mode
    if not RUNTIME_CONFIG.get("enable_streamlit_warnings", True):
        logging.getLogger("streamlit").setLevel(logging.ERROR)
        logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

class Config:
    BASE_DIR = BASE_DIR
    UPLOADS_DIR = UPLOADS_DIR
    DATA_DIR = DATA_DIR
    TESTS_DIR = TESTS_DIR
    DATABASE_URL = DATABASE_URL
    RUNTIME_CONFIG = RUNTIME_CONFIG
    LLM_CONFIG = LLM_CONFIG
    AGENT_CONFIG = AGENT_CONFIG
    FILE_CONFIG = FILE_CONFIG
    UI_CONFIG = UI_CONFIG
    CLI_CONFIG = CLI_CONFIG
    INTENT_CATEGORIES = INTENT_CATEGORIES
    LOGGING_CONFIG = LOGGING_CONFIG
    DEBUG = DEBUG
    MOCK_LLM = MOCK_LLM
    SQLITE_CONFIG = SQLITE_CONFIG