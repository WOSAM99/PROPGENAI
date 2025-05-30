import os
import shutil
import logging
import tempfile
import io
import uuid
import hashlib
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
    DeleteCollectionResponse # Added DeleteCollectionResponse
)
# Import for success_response
from app.utils.response import success_response, error_response


# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

router = APIRouter()

@router.post("/upload", 
            summary="Upload a document (PDF expected) and process it for embedding",
            response_model=FileUploadResponse,
            responses={
                400: {"model": ErrorResponse, "description": "Bad Request (e.g., invalid file type, empty file)"},
                422: {"model": ErrorResponse, "description": "Unprocessable Entity (e.g., PDF has no text, no chunks created)"},
                500: {"model": ErrorResponse, "description": "Internal Server Error processing the file"}
            }
)
async def upload_and_embed_document(file: UploadFile = File(...), profileID: str = Form(...)):
    try:
        validate_uploaded_pdf(file)
        if not profileID:
            logger.error("profileID is required for document upload.")
            return error_response("profileID is required.", 400)
        document_id = str(uuid.uuid4())
        collection_name = profileID
        try:
            validate_collection_name(collection_name)
        except ValueError as e:
            logger.error(f"Invalid collection name generated: {collection_name}", exc_info=True)
            return error_response(f"Error generating valid collection name: {str(e)}", 500)
        logger.info(f"Reading uploaded file into memory: {file.filename} for profileID: {profileID} (collection: {collection_name})")
        pdf_bytes = await file.read()
        await file.close()
        await validate_pdf_bytes(pdf_bytes, file.filename)
        logger.info(f"Processing PDF bytes for: {file.filename}")
        markdown_text = read_pdf(pdf_bytes)
        logger.info(f"Markdown extracted for {file.filename}. Length: {len(markdown_text)}")
        markdown_docs = [{"filename": file.filename, "markdown": markdown_text}]
        logger.info(f"Successfully converted {file.filename} to markdown from bytes.")
        chunks = chunk_by_headings(markdown_docs)
        logger.info(f"Successfully chunked content from {file.filename} into {len(chunks)} chunks.")
        collection = store_chunks_in_chromadb(chunks=chunks, collection_name=collection_name)
        logger.info(f"Successfully embedded and stored document: {file.filename} in collection: {collection_name}")
        response = FileUploadResponse(
            document_id=document_id,
            collection_id=collection_name,
            status="processing",
            filename=file.filename,
            detail="Document uploaded and processing started",
            chunks_created=len(chunks)
        )
        return success_response(response)
    except HTTPException as e:
        logger.error(f"HTTPException during document upload: {e.detail}", exc_info=True)
        return error_response(str(e.detail), e.status_code)
    except Exception as e:
        logger.error(f"Error processing document {getattr(file, 'filename', 'unknown')}: {str(e)}", exc_info=True)
        return error_response(f"Error processing document: {str(e)}", 500)

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

