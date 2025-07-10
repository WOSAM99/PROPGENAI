import os
import logging
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict, Annotated

import google.generativeai as genai # type: ignore
from dotenv import load_dotenv # type: ignore
from fastapi import HTTPException # type: ignore

from langchain_core.embeddings import Embeddings # type: ignore
from langchain_core.documents import Document # type: ignore
from langchain_core.prompts import ChatPromptTemplate # type: ignore
from langchain_chroma import Chroma # type: ignore
from langchain_google_genai import ChatGoogleGenerativeAI # type: ignore
from langgraph.graph import START, StateGraph # type: ignore
from langgraph.graph.message import add_messages # type: ignore

# Import conversation history functions
try:
    from .conv import fetch_conversation_as_context_string
except ImportError:
    def fetch_conversation_as_context_string(profile_id: str, limit: int = 10) -> str:
        return "No conversation history available."

load_dotenv()
logger = logging.getLogger(__name__)

# Configure Google API
google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key:
    genai.configure(api_key=google_api_key)
else:
    logger.warning("GOOGLE_API_KEY not found in environment variables")

class GoogleEmbeddings(Embeddings):
    def embed_documents(self, texts):
        try:
            return [genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )['embedding'] for text in texts]
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            raise HTTPException(status_code=500, detail=f"Error embedding documents: {str(e)}")

    def embed_query(self, text):
        try:
            return genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_query"
            )['embedding']
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise HTTPException(status_code=500, detail=f"Error embedding query: {str(e)}")

# Global embedding model instance
embedding_model = GoogleEmbeddings()

# System prompt with conversation history support
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant engaging in a conversation with the user.

Previous Conversation History:
{conversation_history}

Document Context:
{context}



Example tone: Think of how an experienced freelancer would respond on Upwork or LinkedIn DM—not too long, not too short, no fluff.
."




"""
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    retriever: Any
    prompt_template: ChatPromptTemplate
    # messages: Annotated[list, add_messages]
    profile_id: str
    conversation_history: str

def retrieve_documents(state: State):
    """Retrieve relevant documents from the vector store."""
    try:
        retrieved_docs = state["retriever"].invoke(state["question"])
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        
        return {
            "context": retrieved_docs,
            "question": state["question"],
            "retriever": state["retriever"],
            "prompt_template": state["prompt_template"],
            # "messages": state["messages"],
            "profile_id": state["profile_id"],
            "conversation_history": state["conversation_history"]
        }
    except Exception as e:
        logger.error(f"Error in retrieve_documents: {e}")
        raise

def generate_answer(state: State):
    """Generate answer using retrieved documents and conversation history."""
    try:
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        conversation_history = state.get("conversation_history", "No previous conversation.")
        
        # Create messages with context and conversation history
        messages = state["prompt_template"].invoke({
            "question": state["question"],
            "context": docs_content,
            "conversation_history": conversation_history
        })

        # Generate response
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20")
        response = llm.invoke(messages)
        
        logger.info(f"Generated response: {len(response.content)} characters")

        return {
            "answer": response.content,
            "context": state["context"],
            "question": state["question"],
            # "messages": state["messages"],
            "profile_id": state["profile_id"],
            "conversation_history": state["conversation_history"]
        }
    except Exception as e:
        logger.error(f"Error in generate_answer: {e}")
        raise

# Build the RAG graph
graph_builder = StateGraph(State)
graph_builder.add_node("retrieve_documents", retrieve_documents)
graph_builder.add_node("generate_answer", generate_answer)
graph_builder.add_edge(START, "retrieve_documents")
graph_builder.add_edge("retrieve_documents", "generate_answer")
compiled_rag_graph = graph_builder.compile()

def get_rag_response(
    query: str,
    collection_name: str,
    profile_id: str,
    k_retrieval: int = 6,
    retriever_filter: Optional[Dict] = None,
    custom_system_prompt: Optional[str] = None,
    conversation_limit: int = 10
) -> Dict:
    """
    Generate a response using RAG with document context and conversation history.
    
    Args:
        query: User's question
        collection_name: ChromaDB collection name
        profile_id: Profile ID to fetch conversation history
        k_retrieval: Number of documents to retrieve
        retriever_filter: Optional filter for document retrieval
        custom_system_prompt: Optional custom system prompt
        conversation_limit: Number of previous messages to include
    """
    logger.info(f"RAG request - Query: {query[:50]}..., Collection: {collection_name}, Profile: {profile_id}")
    
    try:
        # Fetch conversation history
        conversation_history = fetch_conversation_as_context_string(
            profile_id=profile_id, 
            limit=conversation_limit
        )   
        
        # Set up vector store retriever
        vectordb = Chroma(
            collection_name=collection_name,
            persist_directory="chromadb_store",
            embedding_function=embedding_model
        )

        retriever = vectordb.as_retriever(
            search_kwargs={
                "k": k_retrieval,
                **({"filter": retriever_filter} if retriever_filter else {})
            }
        )

        # Create system prompt
        system_prompt = custom_system_prompt + "\n\n" + DEFAULT_SYSTEM_PROMPT if custom_system_prompt else DEFAULT_SYSTEM_PROMPT
        
        # Ensure required placeholders exist
        if "{context}" not in system_prompt:
            system_prompt += "\n\nDocument Context:\n{context}"
        if "{conversation_history}" not in system_prompt:
            system_prompt = "Previous Conversation History:\n{conversation_history}\n\n" + system_prompt

        # Create chat prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])

        # Set up initial state
        initial_state = {
            "question": query,
            "retriever": retriever,
            "prompt_template": prompt_template,
            "context": [],
            "answer": "",
            # "messages": [],
            "profile_id": profile_id,
            "conversation_history": conversation_history
        }

        # Execute RAG pipeline
        result_state = compiled_rag_graph.invoke(initial_state)

        # Prepare response
        source_documents = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in result_state.get("context", [])
        ]

        response = {
            "answer": result_state.get("answer", "No answer generated."),
            "source_documents": source_documents,
            "collection_used": collection_name,
            "num_source_documents": len(source_documents),
            "profile_id": profile_id,
            "conversation_history_used": len(conversation_history) > 0,
            "conversation_history_length": len(conversation_history)
        }

        logger.info(f"RAG response generated - Documents: {len(source_documents)}, History used: {response['conversation_history_used']}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in RAG response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"RAG processing error: {str(e)}")

if __name__ == "__main__":
    # Example usage
    logger.info("Running RAG example...")
    example_collection = "google_embed_chunks"
    
    try:
        vectordb = Chroma(collection_name=example_collection, persist_directory="chromadb_store", embedding_function=embedding_model)
        vectordb.get()
        
        print(f"Testing collection '{example_collection}'...")
        test_query = "who is rajeev menon"
        response = get_rag_response(
            query=test_query, 
            collection_name=example_collection, 
            profile_id="test-profile", 
            conversation_limit=5
        )
        
        print(f"Answer: {response['answer']}")
        print(f"Documents found: {response['num_source_documents']}")
        print(f"Conversation history used: {response['conversation_history_used']}")
        
    except Exception as e:
        logger.warning(f"Test skipped - collection may not exist: {e}") 