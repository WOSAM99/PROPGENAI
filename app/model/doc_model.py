from pydantic import BaseModel, validator, UUID4
from fastapi import UploadFile, HTTPException
from typing import Optional, List
import os
import uuid
import re

class FileUploadResponse(BaseModel):
    document_id: str
    collection_id: str
    status: str = "processing"
    filename: str
    detail: str
    chunks_created: Optional[int] = None

    @validator('document_id')
    def validate_uuid(cls, v):
        # Validate that the string is a valid UUID format
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        if not re.match(uuid_pattern, v.lower()):
            raise ValueError('document_id must be a valid UUID string')
        return v

class ErrorDetail(BaseModel):
    code: int
    message: str
    # further_info: Optional[dict] = None # For more detailed, structured errors if needed

class ErrorResponse(BaseModel):
    error: ErrorDetail

# Validation Utilities / Models

def validate_uploaded_pdf(file: UploadFile):
    """
    Validates if the uploaded file is a PDF and is not empty.
    Raises HTTPException if validation fails.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, 
            detail=ErrorResponse(error=ErrorDetail(code=400, message="Invalid file type. Only PDF files are supported.")).model_dump()
        )
    
    # To check if the file is empty, we need to read its content or check its size.
    # Reading content here might consume the stream before the controller can use it.
    # FastAPI's UploadFile.size is available if the server supports it (depends on SpooledTemporaryFile threshold).
    # A common way is to check after initial read in the controller or by passing the read bytes.
    # For now, this function primarily handles the file type check.
    # Emptiness check will be done after reading bytes in the controller.
    return True

async def validate_pdf_bytes(file_bytes: bytes, filename: str):
    """
    Validates if the provided bytes (from a PDF) are not empty.
    Raises HTTPException if validation fails.
    """
    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(error=ErrorDetail(code=400, message=f"Uploaded file {filename} is empty.")).model_dump()
        )
    return True 

# Models for Delete Collection functionality
class DeleteCollectionRequest(BaseModel):
    collection_name: str

class DeleteCollectionResponse(BaseModel):
    collection_name: str
    detail: str

class DeleteFileRequest(BaseModel):
    collection_name: str
    file_id: str  # This will be the document_id from Supabase

    @validator('collection_name')
    def validate_collection_name(cls, v):
        if not v:
            raise ValueError('collection_name cannot be empty')
        return v

    @validator('file_id')
    def validate_file_id(cls, v):
        if not v:
            raise ValueError('file_id cannot be empty')
        return v

def validate_collection_name(collection_name: str) -> bool:
    """
    Validates that a collection name meets ChromaDB requirements:
    - Must be 3-63 characters
    - Must start and end with alphanumeric character
    - Can only contain alphanumeric characters, underscores, or hyphens
    - Cannot contain two consecutive periods
    - Cannot be a valid IPv4 address
    """
    # Check length
    if not 3 <= len(collection_name) <= 63:
        raise ValueError("Collection name must be between 3 and 63 characters")

    # Check start and end characters
    if not (collection_name[0].isalnum() and collection_name[-1].isalnum()):
        raise ValueError("Collection name must start and end with alphanumeric characters")

    # Check for valid characters
    if not re.match('^[a-zA-Z0-9][a-zA-Z0-9_-]*[a-zA-Z0-9]$', collection_name):
        raise ValueError("Collection name can only contain alphanumeric characters, underscores, or hyphens")

    # Check for consecutive periods
    if '..' in collection_name:
        raise ValueError("Collection name cannot contain consecutive periods")

    # Check for IPv4 address format
    ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    if re.match(ip_pattern, collection_name):
        try:
            # Check if all numbers are valid for IPv4
            if all(0 <= int(i) <= 255 for i in collection_name.split('.')):
                raise ValueError("Collection name cannot be a valid IPv4 address")
        except ValueError:
            pass  # Not a valid IPv4, which is good

    return True 