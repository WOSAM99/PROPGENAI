from fastapi import HTTPException, Request, FastAPI
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from loguru import logger
from app.utils.token import JWTAuth
from app.utils.response import error_response

# Paths that don't require authentication
# Paths that don't require authentication
PUBLIC_PATHS = [
    "/docs",  # Swagger UI
    "/redoc",  # ReDoc UI
    "/openapi.json",  # OpenAPI schema
    "/auth/exchange-supabase-token", # New endpoint for Supabase token exchange
    "/auth/token",  # Old deprecated token endpoint (still public for now)
    "/health",  # Health check endpoint if you have one
]

class JWTAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, exclude_paths=None):
        super().__init__(app)
        self.exclude_paths = exclude_paths or PUBLIC_PATHS
        self.valid_routes = None  # Initialize as None

    def get_valid_routes(self, app: FastAPI):
        """ Extract all (path, method) pairs after app is fully initialized. """
        if self.valid_routes is None:
            self.valid_routes = set()
            for route in app.router.routes:
                if hasattr(route, "path") and hasattr(route, "methods"):
                    for method in route.methods:
                        self.valid_routes.add((route.path, method))
        return self.valid_routes

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        request_path = request.url.path
        request_method = request.method

        logger.info(f"🛡️  JWT MIDDLEWARE: Processing {request_method} {request_path}")

        # Ensure routes are extracted after app initialization
        valid_routes = self.get_valid_routes(request.app)

        # Check if the route exists with the correct method
        if (request_path, request_method) not in valid_routes:
            # Check if the path exists but method is incorrect → 405
            if any(request_path == path for path, _ in valid_routes):
                logger.info(f"🔄 JWT MIDDLEWARE: Route exists but method mismatch, proceeding...")
                return await call_next(request)
            # Otherwise, the route does not exist → 404
            logger.info(f"🔄 JWT MIDDLEWARE: Route not found, proceeding...")
            return await call_next(request)

        # Allow public paths without authentication
        is_public = any(request.url.path.startswith(path) for path in self.exclude_paths)
        if is_public:
            logger.info(f"🟢 JWT MIDDLEWARE: Public path '{request_path}' - skipping authentication")
            return await call_next(request)

        logger.info(f"🔒 JWT MIDDLEWARE: Protected path '{request_path}' - authentication required")

        # Get Authorization header
        auth_header = request.headers.get("Authorization")
        logger.info(f"🔑 JWT MIDDLEWARE: Authorization header: {auth_header}")
        
        # Validate Bearer token format
        try:
            jwt_payload = JWTAuth.verify_token(auth_header)
            # Store the jwt_payload in the request state
            request.state.jwt_payload = jwt_payload
            logger.info(f"✅ JWT MIDDLEWARE: Authentication successful, proceeding to endpoint")
            return await call_next(request)  # Continue processing
        except HTTPException as e:
            logger.error(f"❌ JWT MIDDLEWARE: Authentication failed - {str(e.detail)}")
            return error_response(str(e.detail), e.status_code)
        except Exception as e:
            logger.error(f"❌ JWT MIDDLEWARE: Unexpected error - {str(e)}")
            return error_response("Internal server error", 500)