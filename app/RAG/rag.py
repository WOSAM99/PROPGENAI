import os
import google.generativeai as genai
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict, Optional
from typing import Any, Dict
from dotenv import load_dotenv
from fastapi import HTTPException
import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

load_dotenv()

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

# Global embedding model instance
embedding_model = GoogleEmbeddings()

# Default system prompt
DEFAULT_SYSTEM_PROMPT = """
You are a helpful AI assistant. 

Document Context:
{context}

Guidelines:
- Use the document context to provide accurate and relevant answers
- If you can't find relevant information in the context, say so clearly
- Don't make up information that isn't in the provided documents
- Be concise and helpful in your responses
"""

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    retriever: Any
    prompt_template: Any

def retrieve_documents(state: State):
    """Node function to retrieve relevant documents."""
    try:
        logger.info(f"Retrieving documents for question: {state['question']}")
        retrieved_docs = state["retriever"].invoke(state["question"])
        
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        for i, doc in enumerate(retrieved_docs):
            logger.info(f"Document {i+1} metadata: {doc.metadata}")
            logger.debug(f"Document {i+1} content preview: {doc.page_content[:100]}...")

        return {
            "context": retrieved_docs,
            "question": state["question"],
            "retriever": state["retriever"],
            "prompt_template": state["prompt_template"]
        }
    except Exception as e:
        logger.error(f"Error in retrieve_documents: {e}")
        raise

def generate_answer(state: State):
    """Node function to generate answer using retrieved documents."""
    try:
        # Prepare document context
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        
        logger.info("Preparing LLM input:")
        logger.info(f"- Document Context Length: {len(docs_content)} characters")
        logger.info(f"- Number of Documents: {len(state['context'])}")
        logger.info(f"- Question: {state['question']}")

        # Create messages with context
        messages = state["prompt_template"].invoke({
            "question": state["question"],
            "context": docs_content
        })

        # Generate response
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
        response = llm.invoke(messages)
        
        logger.info(f"Generated response length: {len(response.content)} characters")
        logger.debug(f"Response preview: {response.content[:200]}...")

        return {
            "answer": response.content,
            "context": state["context"],
            "question": state["question"]
        }
    except Exception as e:
        logger.error(f"Error in generate_answer: {e}", exc_info=True)
        raise

# Define the graph
graph_builder = StateGraph(State)
graph_builder.add_node("retrieve_documents", retrieve_documents)
graph_builder.add_node("generate_answer", generate_answer)
graph_builder.add_edge(START, "retrieve_documents")
graph_builder.add_edge("retrieve_documents", "generate_answer")
compiled_rag_graph = graph_builder.compile()

def get_rag_response(
    query: str,
    collection_name: str,
    k_retrieval: int = 6,
    retriever_filter: Optional[Dict] = None,
    custom_system_prompt: Optional[str] = None
) -> Dict:
    """
    Generate a response using RAG with document context.
    """
    logger.info(f"Starting RAG response generation:")
    logger.info(f"- Query: {query}")
    logger.info(f"- Collection: {collection_name}")
    logger.info(f"- K retrieval: {k_retrieval}")
    logger.info(f"- Filter: {retriever_filter}")
    
    try:
        # Set up the vector store retriever
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

        # Create the system prompt
        system_prompt = custom_system_prompt + "\n\n" + DEFAULT_SYSTEM_PROMPT if custom_system_prompt else DEFAULT_SYSTEM_PROMPT
        
        if "{context}" not in system_prompt:
            logger.warning("System prompt missing {context} placeholder. Adding default context section.")
            system_prompt += "\n\nDocument Context:\n{context}"

        # Create the chat prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])

        # Set up the initial state
        initial_state = {
            "question": query,
            "retriever": retriever,
            "prompt_template": prompt_template,
            "context": [],
            "answer": ""
        }

        # Execute the RAG pipeline
        logger.info("Executing RAG pipeline...")
        result_state = compiled_rag_graph.invoke(initial_state)

        # Prepare the response
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
            "num_source_documents": len(source_documents)
        }

        logger.info("RAG response generated successfully:")
        logger.info(f"- Answer length: {len(response['answer'])} characters")
        logger.info(f"- Source documents used: {len(source_documents)}")

        return response

    except HTTPException as he:
        logger.error(f"HTTP Exception in RAG response: {he}")
        raise he
    except Exception as e:
        logger.error(f"Error in RAG response: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG processing error: {str(e)}")

if __name__ == "__main__":
    # Example usage
    try:
        logging.basicConfig(level=logging.INFO)
        logger.info("Running RAG example...")
        
        example_collection = "google_embed_chunks"
        
        try:
            # Test if collection exists
            vectordb = Chroma(collection_name=example_collection, persist_directory="chromadb_store", embedding_function=embedding_model)
            vectordb.get()
            
            print(f"\nQuerying collection '{example_collection}'...")
            specific_query = "who is rajeev menon"
            response_specific = get_rag_response(query=specific_query, collection_name=example_collection)
            print(f"Answer for '{specific_query}': {response_specific['answer']}")
            if response_specific['source_documents']:
                print(f"Sources: {[doc['metadata'].get('source', 'Unknown source') for doc in response_specific['source_documents']]}")
            else:
                print("No source documents found.")

        except Exception as e:
            logger.warning(f"Skipping query for '{example_collection}' as it might not exist: {e}")

    except Exception as e:
        logger.error(f"Error in RAG example: {e}", exc_info=True) 