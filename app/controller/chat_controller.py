import logging
from fastapi import APIRouter, Depends, HTTPException, Body

# Models
from app.model.chat_model import ChatRequest, ChatResponse, SourceDocument
from app.model.doc_model import ErrorResponse # For OpenAPI responses

# RAG Pipeline
from app.RAG.rag import get_rag_response

# Response Utilities
from app.utils.response import success_response, error_response # Assuming you have this

# Configure logging
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO) # Configure at application level if preferred

router = APIRouter()

@router.post("/query",
            summary="Query the RAG pipeline to get an answer with optional filters and prompts.",
            response_model=ChatResponse,
            responses={
                422: {"model": ErrorResponse, "description": "Validation Error (e.g., invalid filter, bad request parameters)"},
                500: {"model": ErrorResponse, "description": "Internal Server Error (e.g., LLM error, unexpected RAG pipeline error)"},
                503: {"model": ErrorResponse, "description": "Service Unavailable (e.g., cannot connect to Vector DB)"}
                # Add 404 if collection not found becomes a distinct error raised from rag.py, 
                # currently handled by returning no docs or specific 503 for DB connection issues.
            }
)
async def query_rag_pipeline(request: ChatRequest = Body(...)):
    """
    Receives a user query and other optional parameters, then queries the RAG pipeline.

    - **query**: The question to ask the RAG system.
    - **profileID**: Specifies the ChromaDB collection to use for context retrieval.
    - **k_retrieval**: Number of documents to retrieve from the vector store.
    - **retriever_filter**: A dictionary to filter documents in ChromaDB (e.g., `{"source": "my_document.pdf"}`).
    - **system_prompt**: An alternative system prompt to guide the LLM. If not provided, a default is used.
    """
    try:
        logger.info(f"Received chat query for collection '{request.profile_id}': '{request.query}'")
        
        rag_result = get_rag_response(
            query=request.query,
            collection_name=request.profile_id,
            k_retrieval=request.k_retrieval,
            retriever_filter=request.retriever_filter,
            custom_system_prompt=request.system_prompt
        )

        # Transform source_documents dicts to SourceDocument model instances if they exist
        source_docs_models = []
        if rag_result.get("source_documents"):
            for doc_data in rag_result["source_documents"]:
                source_docs_models.append(SourceDocument(**doc_data))

        return success_response(
            ChatResponse(
                answer=rag_result["answer"],
                source_documents=source_docs_models if source_docs_models else None, # Ensure None if empty
                profile_id=rag_result["collection_used"]
            )
        )
    except HTTPException as e:
        logger.error(f"HTTPException in chat controller: {e.detail}", exc_info=True)
        return error_response(str(e.detail), e.status_code)
    except Exception as e:
        logger.error(f"Unexpected error in chat controller while querying RAG: {str(e)}", exc_info=True)
        return error_response("An unexpected error occurred while processing your chat request.", 500) 