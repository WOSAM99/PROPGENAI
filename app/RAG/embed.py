import os
import re
import logging
from typing import List, Dict
from datetime import datetime

import pymupdf4llm # type: ignore
import pymupdf # type: ignore
import chromadb # type: ignore
import google.generativeai as genai # type: ignore
from dotenv import load_dotenv # type: ignore
from fastapi import HTTPException # type: ignore

load_dotenv()
logger = logging.getLogger(__name__)

# Configure Google API
google_api_key = os.environ.get("GOOGLE_API_KEY", "")
if google_api_key:
    genai.configure(api_key=google_api_key)
else:
    logger.warning("GOOGLE_API_KEY not set. Google AI features will be disabled.")

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
        
        logger.info(f"Successfully processed PDF: {len(chunks)} chunks created")
        return chunks
        
    except Exception as e:
        logger.error(f"Error reading PDF {file_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

# def chunk_by_headings(markdown_docs: List[Dict]) -> List[Dict]:
#     markdown_docs: List[Dict], fallback_chunk_size: int = 300) -> List[Dict]
#     """
#     Split markdown documents into chunks based on headings.
    
#     Args:
#         markdown_docs: List of markdown documents with filename and content
        
#     Returns:
#         List of text chunks with source metadata
#     """
#     chunks = []

#     for doc in markdown_docs:
#         sections = re.split(r"(#+ .+)", doc["markdown"])  # capture headings
        
#         for i in range(1, len(sections), 2):
#             heading = sections[i].strip()
#             body = sections[i + 1].strip()
#             section_text = f"{heading}\n\n{body}"

#             chunks.append({
#                 "text": section_text,
#                 "source": doc["filename"]
#             })
    
#     if not chunks:
#         if not markdown_docs:
#             raise HTTPException(status_code=422, detail="No markdown documents provided for chunking.")
#         else:
#             raise HTTPException(status_code=422, detail="No actionable chunks created from the document (no headings found).")

#     return chunks



def chunk_by_headings(markdown_docs: List[Dict], fallback_chunk_size: int = 300) -> List[Dict]:
    """
    Chunk markdown documents by headings if available, else by paragraphs.

    Args:
        markdown_docs: List of markdown documents with filename and markdown.
        fallback_chunk_size: Approximate number of words per fallback chunk.

    Returns:
        List of chunks with source metadata.
    """
    chunks = []

    for doc in markdown_docs:
        source = doc["filename"]
        markdown = doc["markdown"].strip()

        # Try heading-based split
        sections = re.split(r"(#+ .+)", markdown)
        if len(sections) > 1:
            for i in range(1, len(sections), 2):
                heading = sections[i].strip()
                body = sections[i + 1].strip() if i + 1 < len(sections) else ""
                if len(body.split()) < 10:
                    continue  # skip very short sections
                section_text = f"{heading}\n\n{body}"
                chunks.append({"text": section_text, "source": source})
        else:
            # Fallback: paragraph chunking
            words = markdown.split()
            chunk = []
            for word in words:
                chunk.append(word)
                if len(chunk) >= fallback_chunk_size:
                    chunk_text = " ".join(chunk).strip()
                    if chunk_text:
                        chunks.append({"text": chunk_text, "source": source})
                    chunk = []
            # Add remaining chunk
            if chunk:
                chunk_text = " ".join(chunk).strip()
                if chunk_text:
                    chunks.append({"text": chunk_text, "source": source})

    if not chunks:
        raise HTTPException(status_code=422, detail="Unable to chunk the document meaningfully.")

    return chunks


def create_google_embeddings(texts: List[str], model: str = "text-embedding-004") -> List[List[float]]:
    """
    Create embeddings for text chunks using Google's embedding model.
    
    Args:
        texts: List of text strings to embed
        model: Google embedding model to use
        
    Returns:
        List of embedding vectors
    """
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
            logger.error(f"Error embedding chunk {i}: {e}")
            # Use zero vector as fallback
            embeddings.append([0.0] * 768)
    
    logger.info(f"Created embeddings for {len(texts)} text chunks")
    return embeddings

def store_chunks_in_chromadb(chunks: List[Dict], collection_name: str, document_id: str):
    """
    Store document chunks in ChromaDB with embeddings.
    
    Args:
        chunks: List of dictionaries containing text chunks and metadata
        collection_name: Name of the collection (profile_id)
        document_id: Unique identifier for the document from Supabase
        
    Returns:
        ChromaDB collection object
    """
    if not collection_name:
        raise HTTPException(status_code=400, detail="Collection name (profile_id) is required.")

    if not document_id:
        raise HTTPException(status_code=400, detail="Document ID is required.")

    # Prepare data for ChromaDB
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [{
        "source": chunk.get("source", "unknown"),
        "file_id": document_id,
        "upload_timestamp": datetime.utcnow().isoformat()
    } for chunk in chunks]
    ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
    embeddings = create_google_embeddings(texts)

    # Store in ChromaDB
    chroma_client = chromadb.PersistentClient(path="chromadb_store")
    collection = chroma_client.get_or_create_collection(name=collection_name)
    collection.add(
        documents=texts, 
        embeddings=embeddings, 
        metadatas=metadatas, 
        ids=ids
    )
    
    logger.info(f"Stored {len(texts)} chunks in ChromaDB collection '{collection_name}' for document '{document_id}'")
    return collection

def delete_file_from_collection(collection_name: str, file_id: str):
    """
    Delete specific file chunks from a ChromaDB collection.
    
    Args:
        collection_name: Name of the ChromaDB collection
        file_id: Document ID to delete chunks for
    """
    try:
        chroma_client = chromadb.PersistentClient(path="chromadb_store")
        collection = chroma_client.get_collection(name=collection_name)
        
        # Delete all chunks for this file_id
        collection.delete(where={"file_id": file_id})
        logger.info(f"Deleted chunks for file '{file_id}' from collection '{collection_name}'")
        
    except ValueError as ve:
        logger.error(f"Collection '{collection_name}' not found: {ve}")
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found.")
    except Exception as e:
        logger.error(f"Error deleting file chunks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete file chunks: {str(e)}")

def delete_collection_from_chromadb(collection_name: str):
    """
    Delete an entire collection from ChromaDB.
    
    Args:
        collection_name: Name of the collection to delete
    """
    try:
        chroma_client = chromadb.PersistentClient(path="chromadb_store")
        chroma_client.delete_collection(name=collection_name)
        logger.info(f"Successfully deleted collection: {collection_name}")
        
    except ValueError as ve:
        logger.error(f"Collection '{collection_name}' not found: {ve}")
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found.")
    except Exception as e:
        logger.error(f"Error deleting collection: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete collection '{collection_name}': {str(e)}")

# Legacy functions for backward compatibility
def parse_pdfs_to_markdown(folder_path: str, save_output: bool = True) -> List[Dict]:
    """
    Parse multiple PDFs in a folder to markdown format.
    
    Args:
        folder_path: Path to folder containing PDF files
        save_output: Whether to save markdown files to disk
        
    Returns:
        List of markdown documents
    """
    all_markdown = []
    
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder_path, file)
            
            try:
                md_text = pymupdf4llm.to_markdown(file_path)
                all_markdown.append({
                    "filename": file,
                    "markdown": md_text
                })
                
                if save_output:
                    output_path = os.path.join(folder_path, file.replace(".pdf", ".md"))
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(md_text)
                        
            except Exception as e:
                logger.error(f"Error processing {file}: {e}")
                continue
    
    logger.info(f"Processed {len(all_markdown)} PDF files from {folder_path}")
    return all_markdown