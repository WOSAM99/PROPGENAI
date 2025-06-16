import time
import json
from loguru import logger # type: ignore
from fastapi import Request # type: ignore  
from starlette.responses import Response # type: ignore

async def log_requests_middleware(request: Request, call_next):
    """Enhanced middleware to log API requests with detailed debugging info"""

    start_time = time.time()  # Start the timer
    
    # Log request details
    headers = dict(request.headers)
    auth_header = headers.get("authorization", "NO AUTH HEADER")
    
    logger.info(f"🔵 INCOMING REQUEST: {request.method} {request.url.path}")
    logger.info(f"📋 Headers: {json.dumps(dict(headers), indent=2)}")
    logger.info(f"🔑 Authorization Header: {auth_header}")
    
    # Log request body for POST/PUT/PATCH
    request_body = None
    if request.method in ("POST", "PUT", "PATCH"):
        try:
            request_body = await request.body()
            if request_body:
                # Try to decode as JSON for better logging
                try:
                    body_json = json.loads(request_body.decode())
                    logger.info(f"📄 Request Body (JSON): {json.dumps(body_json, indent=2)}")
                except:
                    logger.info(f"📄 Request Body (Raw): {request_body.decode()[:500]}...")
        except Exception as e:
            logger.error(f"❌ Error reading request body: {e}")
    
    # Process the request
    response = await call_next(request)
    
    duration = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    # Log response details
    status_emoji = "✅" if response.status_code < 400 else "❌"
    logger.info(f"{status_emoji} RESPONSE: {request.method} {request.url.path} | Status: {response.status_code} | Time: {duration:.2f}ms")
    
    return response