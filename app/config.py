from functools import lru_cache
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    # Base configuration
    APP_NAME: str = "FastAPI Project"
    DEBUG: bool = False
    
    # JWT configuration
    JWT_SECRET_KEY: str = "123456"  # Change this in production!
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Supabase settings
    SUPABASE_URL: str = "https://liitzahdobuegvokysqo.supabase.co"
    SUPABASE_ANON_KEY: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImxpaXR6YWhkb2J1ZWd2b2t5c3FvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDg0MTYwNjksImV4cCI6MjA2Mzk5MjA2OX0.xDrBdxvSQJ0r4X479L-pXRnE8ZfvKhjtCFFTOvHaTAw"
    SUPABASE_KEY: str = ""  # Add this field that was in the error

    # Google API settings
    GOOGLE_API_KEY: str = ""

    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Logging configuration
    LOG_LEVEL: str = "INFO"

    # CORS configuration
    CORS_ORIGINS: str = ""  # Empty by default, set to comma-separated list of domains or "*" for all
    CORS_METHODS: list = ["*"]
    CORS_HEADERS: list = ["*"]

    # Environment-specific settings (for dynamic behavior)
    ENVIRONMENT: str = "development"  # Default to development
    
    # Additional fields that might be in .env file
    environment: str = "development"  # Lowercase version
    host: str = "0.0.0.0"  # Lowercase version
    port: str = "8000"  # String version
    supabase_anon_key: str = ""  # Lowercase version
    log_level: str = "DEBUG"  # Lowercase version
    cors_origins: str = "*"  # Lowercase version
    
    class Config:
        env_file = ".env"  # Single .env file for all environments
        case_sensitive = False  # Allow case insensitive matching
        extra = "allow"  # Allow extra fields

@lru_cache()
def get_settings():
    """
    Function to load settings based on the environment from the `.env` file.
    """
    settings = Settings()  # Load the settings from the .env file
    
    # Adjust settings dynamically based on the environment
    env = (settings.ENVIRONMENT or settings.environment).lower()
    if env == "production":
        settings.DEBUG = False
        settings.LOG_LEVEL = "INFO"
        # Parse CORS origins from the environment string
        cors_origins = settings.CORS_ORIGINS or settings.cors_origins
        if cors_origins:
            # Remove quotes if present
            cors_str = cors_origins.strip('"').strip("'")
            if cors_str == "*":
                settings.CORS_ORIGINS = []  # Disallow "*" in production for security
            else:
                settings.CORS_ORIGINS = [origin.strip() for origin in cors_str.split(",") if origin.strip()]
        else:
            settings.CORS_ORIGINS = []  # No CORS origins allowed if not specified
        settings.CORS_HEADERS: list = [ # type: ignore
            "Authorization",  # For JWT tokens
            "Content-Type",   # For application/json and other content types
            "Accept",         # For content negotiation
            "Origin",        # Required for CORS
            "X-Requested-With"  # For AJAX requests
        ]
    else:
        settings.DEBUG = True
        settings.LOG_LEVEL = "DEBUG"
        settings.CORS_ORIGINS = ["*"]  # Allow all origins for development
    
    return settings

# Create a settings instance
settings = get_settings()