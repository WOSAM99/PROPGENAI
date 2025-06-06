from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from loguru import logger
from starlette.exceptions import HTTPException
from app.controller.test_controller import router as test_router
from app.controller.auth_controller import router as auth_router
from app.controller.document_controller import router as document_controller
from app.controller.chat_controller import router as chat_controller
from app.utils.response import error_response

def setup_routes(app: FastAPI):
    """Setup all routes for the application"""

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        if exc.status_code == 404:
            # Now distinguish between route not found and explicit raise
            if exc.detail == "Not Found":
                # This is likely a route-missing 404 from FastAPI internals
                return error_response("The requested resource was not found", 404)
            else:
                # This is an explicit raise HTTPException(404) with custom detail
                logger.error(f"HTTP error: {exc.detail}")
                return error_response(error=exc.detail, status_code=404)
        else:
            logger.error(f"HTTP error: {exc.detail}")
            return error_response(error=exc.detail, status_code=exc.status_code)

    @app.exception_handler(405)
    async def method_not_allowed_handler(request: Request, exc: HTTPException):
        return error_response(f"Method {request.method} not allowed for this endpoint", 405)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.error(f"Validation error: {exc.errors()}")
        
        if exc.errors():
            error = exc.errors()[0]
            
            # Get the field name from the error location
            field_path = " -> ".join(str(loc) for loc in error.get("loc", []))
            
            # Get the error message
            if "ctx" in error and "error" in error["ctx"]:
                error_message = str(error["ctx"]["error"])
            else:
                error_type = error.get("type", "")
                error_msg = error.get("msg", "")
                error_message = f"{field_path}: {error_msg}"
                
                # Make the message more user-friendly
                if error_type == "missing":
                    error_message = f"Missing required field: {field_path}"
                elif error_type == "type_error":
                    error_message = f"Invalid type for field {field_path}: {error_msg}"
            
            logger.error(f"Validation error details: {error_message}")
            return error_response(error_message, 422)
        else:
            return error_response("Request validation failed", 422)

    # Include all routes from the app
    app.include_router(test_router, prefix="/test", tags=["Test"])
    app.include_router(auth_router, prefix="/auth", tags=["Auth"])
    app.include_router(document_controller, prefix="/doc", tags=["Document"])
    app.include_router(chat_controller, prefix="/chat", tags=["Chat"])
    