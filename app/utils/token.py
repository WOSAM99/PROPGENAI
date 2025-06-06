from fastapi import HTTPException
from jose import jwt, JWTError
from datetime import datetime, timedelta, timezone
from typing import Optional
from app.config import settings
from loguru import logger

# You should store this securely in environment variables
JWT_SECRET_KEY = settings.JWT_SECRET_KEY
JWT_ALGORITHM = settings.JWT_ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES

class JWTAuth:
    @staticmethod
    def create_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """
        Create a new JWT token
        """
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def decrypt_token(token: str) -> dict:
        """
        Decrypt and validate the JWT token
        """
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            return payload
        except JWTError as e:
            logger.error(f"JWT validation error: {str(e)}")
            raise HTTPException(
                status_code=403,
                detail="Invalid token or expired token"
            )

    @staticmethod
    def verify_token(auth_header: str | None) -> dict:
        """Verify JWT token and return user_id
        
        Args:
            auth_header: Authorization header containing the Bearer token
            
        Returns:
            dict: The user_id from the token payload
            
        Raises:
            HTTPException: If token is invalid or missing required data
        """
        logger.info(f"🔍 JWT VALIDATION STARTED")
        logger.info(f"🔑 Received auth_header: {auth_header}")
        
        if not auth_header:
            logger.error("❌ JWT VALIDATION FAILED: No authorization header found")
            raise HTTPException(status_code=403, detail="No authorization header found")
        
        try:
            scheme, token = auth_header.split()
            logger.info(f"🔧 Parsed scheme: '{scheme}', token length: {len(token) if token else 0}")
        except ValueError:
            logger.error("❌ JWT VALIDATION FAILED: Invalid authorization header format")
            raise HTTPException(status_code=403, detail="Invalid authorization header format")

        if scheme.lower() != "bearer":
            logger.error(f"❌ JWT VALIDATION FAILED: Invalid authentication scheme: '{scheme}'")
            raise HTTPException(status_code=403, detail="Invalid authentication scheme")

        logger.info(f"🔐 JWT Token (first 20 chars): {token[:20]}...")
        logger.info(f"🔧 Using JWT_SECRET_KEY (first 10 chars): {JWT_SECRET_KEY[:10]}...")
        logger.info(f"🔧 Using JWT_ALGORITHM: {JWT_ALGORITHM}")

        try:
            payload = JWTAuth.decrypt_token(token)
            logger.info(f"✅ JWT VALIDATION SUCCESS!")
            logger.info(f"📋 Decoded payload: {payload}")
            
            # TODO: Add validation for payload for e.g. user_id or domain
                
            return payload
            
        except HTTPException:
            logger.error("❌ JWT VALIDATION FAILED: HTTPException during decryption")
            raise
        except Exception as e:
            logger.error(f"❌ JWT VALIDATION FAILED: Unexpected error during validation: {str(e)}")
            raise HTTPException(status_code=403, detail="Invalid token")