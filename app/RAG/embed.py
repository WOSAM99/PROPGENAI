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

load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

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

def store_chunks_in_chromadb(chunks: List[Dict], collection_name: str):
    if not collection_name:
        # Log this error as well for server-side tracking
        # logger.error("Attempted to store chunks without a collection_name (profileID).") 
        # Assuming logger is configured in this file or globally
        raise HTTPException(status_code=400, detail="profileID (collection_name) is required to store chunks.")

    texts = [chunk["text"] for chunk in chunks]
    metadatas = [{"source": chunk.get("source", "unknown")} for chunk in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    embeddings = create_google_embeddings(texts)

    chroma_client = chromadb.PersistentClient(path="chromadb_store")

    collection = chroma_client.get_or_create_collection(name=collection_name)
    collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)
    print(f"Stored {len(texts)} chunks in ChromaDB collection: {collection_name}")
    return collection

def read_pdf(pdf_bytes):
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    markdown_text = pymupdf4llm.to_markdown(doc, write_images=False)
    if not markdown_text.strip():
        raise HTTPException(status_code=422, detail="No text content could be extracted from the PDF.")
    return markdown_text

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