from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class ChatRequest(BaseModel):
    query: str = Field(..., description="The user's query for the RAG pipeline.")
    profile_id: str = Field( description="The ChromaDB collection to query.")
    k_retrieval: int = Field(default=3, ge=1, le=10, description="Number of documents to retrieve.")
    retriever_filter: Optional[Dict[str, Any]] = Field(default=None, description="Optional filter for the ChromaDB retriever, e.g., {\"source\": \"filename.pdf\"}.")
    system_prompt: Optional[str] = Field(default=None, description="Optional custom system prompt to override the default.")

class SourceDocument(BaseModel):
    page_content: str
    metadata: Dict[str, Any]

class ChatResponse(BaseModel):
    answer: str = Field(..., description="The LLM's answer to the query.")
    source_documents: Optional[List[SourceDocument]] = Field(default=None, description="List of source documents used to generate the answer.")
    profile_id: str = Field(..., description="The ChromaDB collection that was queried.") 