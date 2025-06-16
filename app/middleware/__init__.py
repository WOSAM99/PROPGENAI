from app.config import settings
from fastapi import FastAPI # type: ignore  
from .logging import log_requests_middleware
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from .jwt_auth import JWTAuthMiddleware # type: ignore  

def setup_middlewares(app: FastAPI):
    """Apply all middlewares to the FastAPI app"""
    
    # Add CORS middleware first
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,  # Configure this appropriately for production
        allow_credentials=True,
        allow_methods=settings.CORS_METHODS,
        allow_headers=settings.CORS_HEADERS,
    )
    
    # Add JWT authentication middleware after CORS
    # Temporarily disabled authentication
    # app.add_middleware(JWTAuthMiddleware)
    
    # Add logging middleware last
    app.middleware("http")(log_requests_middleware)