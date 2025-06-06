import os
import shutil
import logging
import tempfile
import io
import uuid
import hashlib
import chromadb
# import pymupdf # No longer directly used here, but indirectly by read_pdf
from fastapi import APIRouter, UploadFile, HTTPException, File, Depends, Body, Form # Added Form
from typing import List, Optional # Added Optional

# Imports from your RAG embedding script
from app.RAG.embed import (
    chunk_by_headings, 
    store_chunks_in_chromadb, 
    read_pdf,
    delete_collection_from_chromadb # Added delete_collection_from_chromadb
)
# import pymupdf4llm # No longer directly used here, but indirectly by read_pdf

# Imports from model
from app.model.doc_model import (
    FileUploadResponse,
    ErrorResponse, 
    ErrorDetail, # Kept for context, though not directly used for raising errors here
    validate_uploaded_pdf, 
    validate_pdf_bytes,
    validate_collection_name,
    DeleteCollectionRequest, # Added DeleteCollectionRequest
    DeleteCollectionResponse, # Added DeleteCollectionResponse
    DeleteFileRequest  # Add this new model
)
# Import for success_response
from app.utils.response import success_response, error_response


# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

router = APIRouter()

@router.post("/upload",
            summary="Upload a document and create embeddings",
            response_model=FileUploadResponse
)
async def upload_document(
    file: UploadFile = File(...),
    profileID: str = Form(...),  # Changed from Body to Form
):
    try:
        # Validate the uploaded file
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        # Generate a unique document ID
        document_id = str(uuid.uuid4())
        
        # Read and validate PDF content
        pdf_bytes = await file.read()
        if not validate_pdf_bytes(pdf_bytes, file.filename):
            raise HTTPException(status_code=400, detail="Invalid PDF file")

        # Create a temporary file to process the PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_file.flush()
            
            try:
                # Extract text from PDF
                chunks = read_pdf(tmp_file.name, source=file.filename)
                
                # Store chunks in ChromaDB
                collection = store_chunks_in_chromadb(chunks, profileID, document_id)
                
                return success_response(FileUploadResponse(
                    document_id=document_id,
                    collection_id=profileID,
                    status="completed",
                    filename=file.filename,
                    detail="Document processed successfully",
                    chunks_created=len(chunks)
                ))
            finally:
                # Clean up temporary file
                os.unlink(tmp_file.name)
                
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@router.post("/delete",
            summary="Delete a ChromaDB collection by name",
            response_model=DeleteCollectionResponse,
            responses={
                404: {"model": ErrorResponse, "description": "Collection not found"},
                422: {"model": ErrorResponse, "description": "Validation Error (e.g. invalid collection name format if validation added)"},
                500: {"model": ErrorResponse, "description": "Internal Server Error"}
            }
)
async def delete_collection_endpoint(request: DeleteCollectionRequest = Body(...)):
    try:
        logger.info(f"Attempting to delete collection: {request.collection_name}")
        delete_collection_from_chromadb(request.collection_name)
        logger.info(f"Successfully initiated deletion for collection: {request.collection_name}")
        return success_response(DeleteCollectionResponse(
            collection_name=request.collection_name,
            detail=f"Collection '{request.collection_name}' has been successfully deleted."
        ))
    except HTTPException as e:
        logger.error(f"HTTPException during collection deletion: {e.detail}", exc_info=True)
        return error_response(str(e.detail), e.status_code)
    except Exception as e:
        logger.error(f"Error deleting collection {getattr(request, 'collection_name', 'unknown')}: {str(e)}", exc_info=True)
        return error_response(f"Error deleting collection: {str(e)}", 500)

@router.post("/delete-file",
            summary="Delete a specific file from a ChromaDB collection",
            response_model=DeleteCollectionResponse,
            responses={
                404: {"model": ErrorResponse, "description": "Collection or file not found"},
                422: {"model": ErrorResponse, "description": "Validation Error"},
                500: {"model": ErrorResponse, "description": "Internal Server Error"}
            }
)
async def delete_file_from_collection_endpoint(request: DeleteFileRequest = Body(...)):
    try:
        logger.info(f"Attempting to delete file {request.file_id} from collection: {request.collection_name}")
        
        chroma_client = chromadb.PersistentClient(path="chromadb_store")
        collection = chroma_client.get_collection(name=request.collection_name)
        
        # Delete all chunks associated with this file_id
        collection.delete(
            where={"file_id": request.file_id}
        )
        
        logger.info(f"Successfully deleted file {request.file_id} from collection: {request.collection_name}")
        return success_response(DeleteCollectionResponse(
            collection_name=request.collection_name,
            detail=f"File '{request.file_id}' has been successfully deleted from collection '{request.collection_name}'."
        ))
    except ValueError as ve:
        logger.error(f"Collection or file not found: {str(ve)}", exc_info=True)
        return error_response("Collection or file not found", 404)
    except Exception as e:
        logger.error(f"Error deleting file from collection: {str(e)}", exc_info=True)
        return error_response(f"Error deleting file: {str(e)}", 500)

