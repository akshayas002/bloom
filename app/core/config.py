"""Application configuration via environment variables."""
import os
import sys
from functools import lru_cache

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from pydantic_settings import BaseSettings as _Base

    class Settings(_Base):
        SECRET_KEY:   str  = "bloom-change-me-in-production"
        APP_NAME:     str  = "Bloom"
        DEBUG:        bool = False
        DATABASE_URL: str  = "sqlite+aiosqlite:///./instance/bloom.db"
        GROQ_API_KEY: str  = ""
        GROQ_MODEL:   str  = "llama3-8b-8192"
        SESSION_COOKIE_HTTPONLY: bool = True
        SESSION_COOKIE_SAMESITE: str  = "lax"

        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            extra = "ignore"

except ImportError:
    class Settings:  # type: ignore
        SECRET_KEY   = os.environ.get("SECRET_KEY",   "bloom-change-me-in-production")
        APP_NAME     = os.environ.get("APP_NAME",     "Bloom")
        DEBUG        = os.environ.get("DEBUG",        "false").lower() == "true"
        DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./instance/bloom.db")
        GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
        GROQ_MODEL   = os.environ.get("GROQ_MODEL",   "llama3-8b-8192")
        SESSION_COOKIE_HTTPONLY = True
        SESSION_COOKIE_SAMESITE = "lax"


@lru_cache()
def get_settings() -> Settings:
    s = Settings()
    # Refuse to start in production with the default key
    if not s.DEBUG and s.SECRET_KEY == "bloom-change-me-in-production":
        print(
            "\n🚨  FATAL: SECRET_KEY is still the default value.\n"
            "    Set a strong random key in your .env or Render environment variables:\n"
            "      python -c \"import secrets; print(secrets.token_hex(32))\"\n",
            file=sys.stderr,
        )
        sys.exit(1)
    return s
