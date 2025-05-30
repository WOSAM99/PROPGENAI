import os
import google.generativeai as genai
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict, Optional
from typing import Any
from dotenv import load_dotenv
from fastapi import HTTPException
import logging

# Configure logging
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO) # Configure at application level if preferred

load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

class GoogleEmbeddings(Embeddings):
    def embed_documents(self, texts):
        try:
            return [genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )['embedding'] for text in texts]
        except Exception as e:
            logger.error(f"Error embedding documents: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error embedding documents: {str(e)}")

    def embed_query(self, text):
        try:
            return genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_query"
            )['embedding']
        except Exception as e:
            logger.error(f"Error embedding query: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error embedding query: {str(e)}")

# Global embedding model instance (can be reused)
embedding_model = GoogleEmbeddings()

# Default system prompt (can be overridden)
DEFAULT_SYSTEM_PROMPT = ( # Renamed for clarity
    "You are a helpful assistant. Use the following context to answer the question. "
    "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n"
    "---"
    "Context: {context}"
)

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    # Add other necessary components that need to be passed through the graph state
    retriever: Any # To pass the dynamically created retriever
    prompt_template: Any # To pass the dynamically created prompt template

# Graph node functions will now accept retriever and prompt_template from the state

def retrieve_documents(state: State):
    try:
        retrieved_docs = state["retriever"].invoke(state["question"])
        logger.info(f"Retrieved {len(retrieved_docs)} documents for question: {state['question']}")
        if not retrieved_docs:
            logger.warning(f"No documents retrieved for question: {state['question']}. Filter might be too restrictive or collection empty.")
        return {"context": retrieved_docs, "question": state["question"], "retriever": state["retriever"], "prompt_template": state["prompt_template"]}
    except ValueError as ve: # Catch issues with filter format from ChromaDB
        logger.error(f"ValueError during document retrieval (likely filter issue): {ve}", exc_info=True)
        raise HTTPException(status_code=422, detail=f"Invalid retriever filter: {str(ve)}")
    except Exception as e:
        logger.error(f"Error during document retrieval: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")

def generate_answer(state: State):
    try:
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = state["prompt_template"].invoke({"question": state["question"], "context": docs_content})
        
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest") # Consider making model configurable
        response = llm.invoke(messages)
        logger.info(f"Generated answer for question: {state['question']}")
        logger.info(f"The response is: {state["context"]}")
        return {"answer": response.content, "context": state["context"], "question": state["question"], "retriever": state["retriever"], "prompt_template": state["prompt_template"]}
    except Exception as e:
        logger.error(f"Error during answer generation with LLM: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating answer with LLM: {str(e)}")

# Graph Definition (remains structurally similar, but nodes get retriever/prompt from state)
graph_builder = StateGraph(State)
graph_builder.add_node("retrieve_documents", retrieve_documents) # Renamed node for clarity
graph_builder.add_node("generate_answer", generate_answer) # Renamed node for clarity
graph_builder.add_edge(START, "retrieve_documents")
graph_builder.add_edge("retrieve_documents", "generate_answer")
compiled_rag_graph = graph_builder.compile() # Keep compiled graph

def get_rag_response(query: str, collection_name: str, k_retrieval: int = 3, retriever_filter: Optional[dict] = None, custom_system_prompt: Optional[str] = None) -> dict:
    logger.info(f"RAG query: '{query}' for collection: '{collection_name}', k: {k_retrieval}, filter: {retriever_filter}, system_prompt: {'Custom' if custom_system_prompt else 'Default'}")
    try:
        vectordb = Chroma(
            collection_name=collection_name,
            persist_directory="chromadb_store", 
            embedding_function=embedding_model
        )
        # Check if collection exists / is not empty implicitly by trying to get it or checking count.
        # Chroma's get_collection doesn't fail if empty, count() can be used.
        # For now, retrieval will return empty if collection missing/empty, which is handled in retrieve_documents node.

    except Exception as e: # Catch ChromaDB client/connection errors
        logger.error(f"Failed to connect to or initialize ChromaDB collection '{collection_name}': {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Could not connect to vector database for collection '{collection_name}'. {str(e)}")

    dynamic_retriever = vectordb.as_retriever(
        search_kwargs={
            "k": k_retrieval,
            **({"filter": retriever_filter} if retriever_filter else {})
        }
    )

    current_system_prompt_template = custom_system_prompt + "\n\n" + DEFAULT_SYSTEM_PROMPT if custom_system_prompt else DEFAULT_SYSTEM_PROMPT
    
    # Ensure context is part of the system prompt if it's custom
    if custom_system_prompt and "{context}" not in custom_system_prompt:
        logger.warning("Custom system prompt does not contain '{context}'. Context from retriever will not be used in prompt.")
        # Or, alternatively, enforce/add {context} - for now, just log.

    final_prompt_template = ChatPromptTemplate.from_messages([
        ("system", current_system_prompt_template),
        ("human", "{question}"),
    ])

    initial_state = {
        "question": query,
        "retriever": dynamic_retriever,
        "prompt_template": final_prompt_template,
        "context": [], # Initialize context as empty list
        "answer": "" # Initialize answer as empty string
    }

    try:
        result_state = compiled_rag_graph.invoke(initial_state)
    except HTTPException: # Re-raise HTTPExceptions from graph nodes
        raise
    except Exception as e:
        logger.error(f"Unexpected error invoking RAG graph: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error processing RAG query: {str(e)}")

    # Prepare source documents for response
    source_documents_data = [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in result_state.get("context", [])
    ]

    return {
        "answer": result_state.get("answer", "No answer generated."),
        "source_documents": source_documents_data,
        "collection_used": collection_name
    }


if __name__ == "__main__":
    # Example usage (assuming chromadb_store and a collection exist)
    try:
        # Setup basic logging for the __main__ example
        logging.basicConfig(level=logging.INFO)
        logger.info("Running RAG example...")
        
        # Ensure a collection exists for the example, e.g., by running embed.py first
        # or by creating a dummy one here if needed for testing rag.py independently.
        example_collection = "google_embed_chunks" # Make sure this collection exists from previous steps
        
        # Simple query
        # response = get_rag_response(query="What is the capital of France?", collection_name=example_collection)
        # print(f"Answer: {response['answer']}")
        # print(f"Sources: {response['source_documents']}")

        # Query with a filter (ensure 'source' metadata exists and 'example.pdf' is a valid source)
        # This requires 'example.pdf' to have been processed and its chunks stored with this metadata.
        # print("\nQuerying with filter...")
        # response_filtered = get_rag_response(
        #     query="Tell me about topic X from example.pdf", 
        #     collection_name=example_collection, 
        #     retriever_filter={"source": "example.pdf"} 
        # )
        # print(f"Answer (filtered): {response_filtered['answer']}")
        # print(f"Sources (filtered): {response_filtered['source_documents']}")

        # Query specific to the hardcoded example if collection not empty
        try:
            # Attempt to get the collection to see if it exists, otherwise skip this query
            Chroma(collection_name=example_collection, persist_directory="chromadb_store", embedding_function=embedding_model).get()
            print(f"\nQuerying specific content (ensure {example_collection} has relevant data)...")
            specific_query = "who is rajeev menon"
            response_specific = get_rag_response(query=specific_query, collection_name=example_collection)
            print(f"Answer for '{specific_query}': {response_specific['answer']}")
            if response_specific['source_documents']:
                print(f"Sources: {[doc['metadata'].get('source', 'Unknown source') for doc in response_specific['source_documents']]}")
            else:
                print("No source documents found.")

        except Exception as e:
            # This catch is for Chroma().get() if the collection doesn't exist or other issues.
            logger.warning(f"Skipping specific query for '{example_collection}' as it might not exist or be queryable: {e}")

    except Exception as e:
        logger.error(f"Error in RAG example: {e}", exc_info=True) 