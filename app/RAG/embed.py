import os
import pymupdf4llm
import pathlib
import re
import chromadb
import google.generativeai as genai
from chromadb.config import Settings
from tqdm import tqdm
import uuid
from typing import List, Dict
from dotenv import load_dotenv
import pymupdf
from fastapi import HTTPException
from datetime import datetime
import logging

# Configure logging
logger = logging.getLogger(__name__)

load_dotenv()

# Handle Google API key gracefully
google_api_key = os.environ.get("GOOGLE_API_KEY", "")
if google_api_key:
    genai.configure(api_key=google_api_key)
else:
    print("Warning: GOOGLE_API_KEY not set. Google AI features will be disabled.")

def parse_pdfs_to_markdown(folder_path, save_output=True):
    all_markdown = []
    
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder_path, file)
            
            md_text = pymupdf4llm.to_markdown(file_path)
            all_markdown.append({
                "filename": file,
                "markdown": md_text
            })
            
            if save_output:
                output_path = os.path.join(folder_path, file.replace(".pdf", ".md"))
                pathlib.Path(output_path).write_bytes(md_text.encode("utf-8"))
    
    return all_markdown

def chunk_by_headings(markdown_docs):
    chunks = []

    for doc in markdown_docs:
        sections = re.split(r"(#+ .+)", doc["markdown"])  # capture headings
        section_text = ""

        for i in range(1, len(sections), 2):
            heading = sections[i].strip()
            body = sections[i + 1].strip()
            section_text = f"{heading}\n\n{body}"

            chunks.append({
                "text": section_text,
                "source": doc["filename"]
            })
    
    if not chunks:
        # If, after processing all markdown_docs, no chunks were created
        # (e.g., because no headings were found or markdown_docs was empty)
        # raise an exception.
        # Check if markdown_docs was empty to provide a more specific message, though
        # this case should ideally be caught earlier if markdown_docs comes from read_pdf output.
        if not markdown_docs:
            # This case might be redundant if read_pdf already ensures content.
            raise HTTPException(status_code=422, detail="No markdown documents provided for chunking.")
        else:
            # Assumes markdown_docs was not empty, but no headings were found to create chunks.
            raise HTTPException(status_code=422, detail="No actionable chunks created from the document (e.g., no headings found for splitting).")

    print(f"Total chunks: {len(chunks)}")
    return chunks

def create_google_embeddings(texts, model="text-embedding-004", dim=768):
    # Check if Google API key is configured
    if not google_api_key:
        raise HTTPException(
            status_code=500, 
            detail="Google API key not configured. Cannot create embeddings."
        )
    
    embeddings = []
    for i, text in enumerate(texts):
        try:
            result = genai.embed_content(
                model=f"models/{model}",
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result['embedding'])
        except Exception as e:
            print(f"Error at chunk {i}: {e}")
            embeddings.append([0.0] * dim)
    return embeddings

def store_chunks_in_chromadb(chunks: List[Dict], collection_name: str, document_id: str):
    """
    Store document chunks in ChromaDB.
    Args:
        chunks: List of dictionaries containing text chunks and metadata
        collection_name: Name of the collection (profile_id)
        document_id: Unique identifier for the document from Supabase
    """
    if not collection_name:
        raise HTTPException(status_code=400, detail="profileID (collection_name) is required to store chunks.")

    if not document_id:
        raise HTTPException(status_code=400, detail="document_id is required to store chunks.")

    texts = [chunk["text"] for chunk in chunks]
    metadatas = [{
        "source": chunk.get("source", "unknown"),
        "file_id": document_id,
        "upload_timestamp": datetime.utcnow().isoformat()
    } for chunk in chunks]
    ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
    embeddings = create_google_embeddings(texts)

    chroma_client = chromadb.PersistentClient(path="chromadb_store")

    collection = chroma_client.get_or_create_collection(name=collection_name)
    collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)
    print(f"Stored {len(texts)} chunks in ChromaDB collection: {collection_name} for document: {document_id}")
    return collection

def read_pdf(file_path: str, source: str = None) -> List[Dict]:
    """
    Read a PDF file and convert it to markdown chunks.
    Args:
        file_path: Path to the PDF file
        source: Original filename or source identifier
    Returns:
        List of dictionaries containing text chunks and metadata
    """
    try:
        # Convert PDF to markdown
        markdown_text = pymupdf4llm.to_markdown(file_path)
        if not markdown_text.strip():
            raise HTTPException(status_code=422, detail="No text content could be extracted from the PDF.")
        
        # Create markdown document with source information
        markdown_docs = [{
            "filename": source or os.path.basename(file_path),
            "markdown": markdown_text
        }]
        
        # Chunk the markdown text
        chunks = chunk_by_headings(markdown_docs)
        
        if not chunks:
            raise HTTPException(status_code=422, detail="No chunks could be created from the PDF content.")
        
        return chunks
    except Exception as e:
        logger.error(f"Error reading PDF {file_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

def delete_collection_from_chromadb(collection_name: str):
    """
    Deletes a collection from ChromaDB by its name.
    Raises HTTPException if the collection is not found or other error occurs.
    """
    try:
        chroma_client = chromadb.PersistentClient(path="chromadb_store")
        chroma_client.delete_collection(name=collection_name)
        print(f"Successfully deleted collection: {collection_name}")
    except ValueError as ve: # ChromaDB typically raises ValueError for non-existent collections
        # We can check the error message to be more specific if needed, but
        # for now, assume ValueError on delete_collection means it didn't exist.
        print(f"Attempted to delete non-existent collection {collection_name}: {ve}")
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found.")
    except Exception as e:
        # Catch any other unexpected errors during ChromaDB operation
        print(f"Error deleting collection {collection_name} from ChromaDB: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete collection '{collection_name}' due to a server error: {str(e)}")