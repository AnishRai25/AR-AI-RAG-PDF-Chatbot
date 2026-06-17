import os
import shutil
import requests
from typing import List, Tuple, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".temp_uploads")

def get_local_ollama_models(base_url: str = "http://localhost:11434") -> List[str]:
    """Queries the local Ollama API to get a list of pulled models."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=2)
        if response.status_code == 200:
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            # Filter out embedding models from LLM list if we want to, or keep them
            return models if models else ["llama3:latest"]
    except Exception:
        pass
    return ["llama3:latest", "llama3", "mistral", "phi3"]


def clean_temp_directory():
    """Removes the temp upload directory if it exists."""
    if os.path.exists(TEMP_DIR):
        try:
            shutil.rmtree(TEMP_DIR)
        except Exception as e:
            print(f"Error cleaning temp directory: {e}")
    os.makedirs(TEMP_DIR, exist_ok=True)

def save_uploaded_file(file_name: str, file_bytes: bytes) -> str:
    """Saves uploaded file bytes to the temp directory and returns the absolute path."""
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR, exist_ok=True)
    
    file_path = os.path.join(TEMP_DIR, file_name)
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    return file_path

def process_pdfs(file_paths: List[str], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Any]:
    """Loads PDFs and splits them into text chunks."""
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    for path in file_paths:
        try:
            loader = PyPDFLoader(path)
            docs = loader.load()
            chunks = text_splitter.split_documents(docs)
            all_chunks.extend(chunks)
        except Exception as e:
            raise RuntimeError(f"Failed to process PDF '{os.path.basename(path)}': {str(e)}")
            
    return all_chunks

def get_embeddings(provider: str, api_key: str = None, model: str = None, base_url: str = None):
    """Factory to build Embeddings object based on provider."""
    provider_clean = provider.lower()
    if "openai" in provider_clean:
        if not api_key:
            raise ValueError("OpenAI API Key is required for OpenAI Embeddings.")
        model_name = model if model else "text-embedding-3-small"
        return OpenAIEmbeddings(openai_api_key=api_key, model=model_name)
    elif "ollama" in provider_clean:
        url = base_url if base_url else "http://localhost:11434"
        model_name = model if model else "nomic-embed-text"
        return OllamaEmbeddings(base_url=url, model=model_name)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")

def get_llm(provider: str, api_key: str = None, model: str = None, base_url: str = None, temperature: float = 0.7):
    """Factory to build Chat LLM based on provider."""
    provider_clean = provider.lower()
    if "openai" in provider_clean:
        if not api_key:
            raise ValueError("OpenAI API Key is required for OpenAI Chat.")
        model_name = model if model else "gpt-4o-mini"
        return ChatOpenAI(openai_api_key=api_key, model=model_name, temperature=temperature, streaming=True)
    elif "ollama" in provider_clean:
        url = base_url if base_url else "http://localhost:11434"
        model_name = model if model else "llama3"
        return ChatOllama(base_url=url, model=model_name, temperature=temperature)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def create_vector_store(chunks: List[Any], embeddings: Any) -> FAISS:
    """Creates a FAISS vector store in-memory from chunks and embeddings."""
    if not chunks:
        raise ValueError("No text chunks provided to index.")
    return FAISS.from_documents(chunks, embeddings)

def query_rag(vector_store: FAISS, query: str, llm: Any, k: int = 4) -> Tuple[str, List[Dict[str, Any]]]:
    """Queries the RAG chatbot and returns the generated answer along with retrieved citation contexts."""
    # 1. Retrieve most similar documents
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    
    if not docs:
        return "I could not find any relevant information in the uploaded documents.", []
        
    # 2. Format the context from retrieved docs
    context_parts = []
    for i, doc in enumerate(docs):
        src = os.path.basename(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page", 0) + 1
        content = doc.page_content.strip()
        context_parts.append(f"--- Document: {src} | Page: {page} ---\n{content}")
        
    context = "\n\n".join(context_parts)
    
    # 3. Create RAG prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a professional RAG (Retrieval-Augmented Generation) chatbot. "
            "You are answering questions based ONLY on the provided document context. "
            "If the retrieved context does not contain the answer, say "
            "'I cannot find the answer to that in the uploaded documents.' "
            "Do not make up facts or use external training knowledge that contradicts or is unsupported by the context. "
            "Always state which document and page you found the information in if appropriate.\n\n"
            "Retrieved Document Context:\n"
            "{context}"
        )),
        ("human", "{question}")
    ])
    
    # 4. Construct chain and run
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query})
    
    # 5. Build citations payload
    citations = []
    for doc in docs:
        citations.append({
            "source": os.path.basename(doc.metadata.get("source", "unknown")),
            "page": doc.metadata.get("page", 0) + 1,
            "content": doc.page_content
        })
        
    return answer, citations
