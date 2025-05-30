from app.utils.response import error_response, success_response
from fastapi import APIRouter, HTTPException
from datetime import timedelta
from app.utils.token import JWTAuth
from app.model.auth_model import TokenRequest, TokenResponse, SupabaseTokenExchangeRequest
from loguru import logger
from app.config import settings
from supabase import create_client, Client

router = APIRouter()

# Helper to get Supabase client
_supabase_client: Client = None

def get_supabase_client():
    global _supabase_client
    if _supabase_client is None:
        if settings.SUPABASE_URL and settings.SUPABASE_ANON_KEY:
            try:
                _supabase_client = create_client(settings.SUPABASE_URL, settings.SUPABASE_ANON_KEY)
                logger.info("Supabase client initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {e}", exc_info=True)
                return None
        else:
            logger.warning("Supabase URL or Anon Key not configured. Supabase token exchange will fail.")
            return None
    return _supabase_client

# Helper to create backend JWT
def create_backend_jwt(user_id: str, expires_minutes: int = 60):
    payload = {"user_id": user_id}
    return JWTAuth.create_token(
        data=payload,
        expires_delta=timedelta(minutes=expires_minutes)
    )

@router.post("/exchange-supabase-token", response_model=TokenResponse, summary="Exchange Supabase Token for Backend JWT")
async def exchange_supabase_token(request_data: SupabaseTokenExchangeRequest):
    """
    Exchanges a Supabase access token for a backend-specific JWT.
    The backend JWT will contain the Supabase user ID.
    """
    supabase_client = get_supabase_client()
    if not supabase_client:
        logger.error("Supabase client not initialized due to missing configuration or init error.", exc_info=True)
        return error_response("Authentication service temporarily unavailable.", 503)

    try:
        logger.info("Attempting to validate Supabase token via supabase_client.auth.get_user().")
        user_response = supabase_client.auth.get_user(jwt=request_data.supabase_token)
        if not user_response or not user_response.user or not user_response.user.id:
            logger.warning(f"Supabase token validation failed or user not found. Full response: {user_response}")
            return error_response("Invalid or expired Supabase token.", 401)
        supabase_user_id = str(user_response.user.id)
        logger.info(f"Supabase token validated successfully for Supabase user_id: {supabase_user_id}")
        expires_delta_minutes = getattr(settings, 'ACCESS_TOKEN_EXPIRE_MINUTES', 60)
        backend_access_token = create_backend_jwt(supabase_user_id, expires_delta_minutes)
        logger.info(f"Backend JWT created for Supabase user_id: {supabase_user_id}")
        return success_response(TokenResponse(access_token=backend_access_token, token_type="bearer"))
    except HTTPException as e:
        logger.error(f"HTTPException during Supabase token exchange: {e.detail}", exc_info=True)
        return error_response(str(e.detail), e.status_code)
    except Exception as e:
        logger.error(f"Unexpected error during Supabase token exchange: {str(e)}", exc_info=True)
        return error_response("An unexpected error occurred during token exchange.", 500)

@router.post("/token", response_model=TokenResponse, deprecated=True, summary="[DEPRECATED] Create Backend JWT from user_id")
async def create_token_deprecated(request: TokenRequest):
    """
    [DEPRECATED - Prefer /exchange-supabase-token]
    Creates a new JWT token directly from a user_id. 
    Primarily for development/testing if Supabase flow is bypassed.
    """
    if settings.ENVIRONMENT.lower() == "production":
        logger.warning("Attempt to use deprecated /token endpoint in production.")
        return error_response("This token generation method is not allowed in production.", 403)
    logger.info(f"Attempting to create a direct JWT for user_id: {request.user_id} (deprecated /token endpoint)")
    try:
        token = create_backend_jwt(request.user_id, getattr(settings, 'ACCESS_TOKEN_EXPIRE_MINUTES', 30))
        return success_response(TokenResponse(access_token=token, token_type="bearer"))
    except Exception as e:
        logger.error(f"Error creating direct JWT via deprecated /token: {str(e)}", exc_info=True)
        return error_response("Error creating direct JWT.", 500)
