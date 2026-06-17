import streamlit as st
import os
import time
from typing import List, Dict, Any
import rag_backend

# Set page configuration
st.set_page_config(
    page_title="AR AI | RAG PDF Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply premium custom CSS styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@300;400;500;600&display=swap');

/* Main UI Fonts */
html, body, [class*="css"], .stMarkdown {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
}

/* Title Styling */
.title-container {
    padding: 1.5rem 0rem;
}

.title-gradient {
    background: linear-gradient(135deg, #FF4B4B 0%, #8B5CF6 50%, #3B82F6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -1px;
}

.subtitle {
    color: #6B7280;
    font-size: 1.1rem;
    margin-top: 0.25rem;
}

/* Glassmorphism sidebar info card */
.sidebar-info-card {
    background-color: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 1rem;
    margin-top: 1rem;
    font-size: 0.85rem;
}

/* Metrics and status cards */
.stat-container {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.stat-card {
    flex: 1;
    background-color: rgba(139, 92, 246, 0.07);
    border: 1px solid rgba(139, 92, 246, 0.15);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}

.stat-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #8B5CF6;
}

.stat-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #6B7280;
}

/* Citations & Sources list styling */
.citation-box {
    background-color: rgba(59, 130, 246, 0.06);
    border-left: 3px solid #3B82F6;
    border-radius: 4px;
    padding: 0.75rem 1rem;
    margin-top: 0.75rem;
    margin-bottom: 0.75rem;
}

.citation-source {
    font-size: 0.85rem;
    font-weight: 600;
    color: #3B82F6;
    margin-bottom: 0.25rem;
}

.citation-snippet {
    font-size: 0.85rem;
    font-style: italic;
    color: #4B5563;
    line-height: 1.4;
}

/* Dark mode overrides for citation text */
@media (prefers-color-scheme: dark) {
    .citation-snippet {
        color: #D1D5DB;
    }
}
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "chunks_count" not in st.session_state:
    st.session_state.chunks_count = 0
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar Controls
st.sidebar.image("https://img.icons8.com/clouds/100/brain.png", width=80)
st.sidebar.markdown("<h2 style='margin-top:0;'>Control Center</h2>", unsafe_allow_html=True)

# 1. Choose Provider
provider = st.sidebar.selectbox(
    "AI Provider",
    ["OpenAI", "Ollama (Local)"],
    help="Select OpenAI for high accuracy (needs API key) or Ollama for free local generation."
)

# Provider Settings
api_key = None
base_url = None
llm_model = ""
emb_model = ""

if provider == "OpenAI":
    api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key (starts with sk-)")
    llm_model = st.sidebar.selectbox(
        "LLM Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0
    )
    emb_model = st.sidebar.selectbox(
        "Embedding Model",
        ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
        index=0
    )
else:
    base_url = st.sidebar.text_input("Ollama Base URL", value="http://localhost:11434", help="Default URL for Ollama local service")
    local_models = rag_backend.get_local_ollama_models(base_url)
    llm_model = st.sidebar.selectbox("LLM Model", local_models, index=local_models.index("llama3:latest") if "llama3:latest" in local_models else 0)
    emb_model = st.sidebar.text_input("Embedding Model", value="nomic-embed-text", help="Make sure this embedding model is pulled locally via 'ollama pull <name>'")


# Advanced Hyperparameters
with st.sidebar.expander("Advanced Settings"):
    chunk_size = st.slider("Chunk Size (characters)", min_value=200, max_value=2000, value=1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
    top_k = st.slider("Top K Source Chunks", min_value=1, max_value=10, value=4)

# Document Upload Section in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📄 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    # Button to process files
    if st.sidebar.button("⚙️ Process & Index Documents", use_container_width=True):
        with st.spinner("Processing and indexing PDFs..."):
            rag_backend.clean_temp_directory()
            saved_paths = []
            
            for f in uploaded_files:
                path = rag_backend.save_uploaded_file(f.name, f.read())
                saved_paths.append(path)
            
            try:
                # 1. Process files into chunks
                chunks = rag_backend.process_pdfs(saved_paths, chunk_size, chunk_overlap)
                
                # 2. Get embeddings
                embeddings = rag_backend.get_embeddings(
                    provider=provider,
                    api_key=api_key,
                    model=emb_model,
                    base_url=base_url
                )
                
                # 3. Create FAISS vector store
                vector_store = rag_backend.create_vector_store(chunks, embeddings)
                
                # 4. Save to state
                st.session_state.vector_store = vector_store
                st.session_state.processed_files = [f.name for f in uploaded_files]
                st.session_state.chunks_count = len(chunks)
                
                st.sidebar.success(f"Successfully indexed {len(uploaded_files)} PDF(s) into {len(chunks)} chunks!")
            except Exception as e:
                st.sidebar.error(f"Error indexing files: {str(e)}")

# Clear DB action
if st.sidebar.button("🗑️ Clear Vector Index", use_container_width=True):
    rag_backend.clean_temp_directory()
    st.session_state.vector_store = None
    st.session_state.processed_files = []
    st.session_state.chunks_count = 0
    st.session_state.chat_history = []
    st.sidebar.info("Vector store index and chat history cleared.")

# Main Page Design
st.markdown("""
<div class="title-container">
    <span class="title-gradient">AR AI</span>
    <div class="subtitle">An advanced RAG Chatbot powered by LangChain, FAISS, and Streamlit</div>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["💬 Chatbot", "🔍 Semantic Search", "📄 Document Manager"])

# Tab 1: Chatbot Interface
with tab1:
    if not st.session_state.vector_store:
        st.info("👈 Please upload and index some PDFs in the sidebar to start asking questions!")
    else:
        # Show mini statistics bar
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{len(st.session_state.processed_files)}</div>
                <div class="stat-label">Active Documents</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{st.session_state.chunks_count}</div>
                <div class="stat-label">Text Chunks Indexed</div>
            </div>
            """, unsafe_allow_html=True)

        # Clear Chat Button
        if st.button("🧹 Clear Chat History", key="clear_chat_tab1"):
            st.session_state.chat_history = []
            st.rerun()

        # Display Chat History
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                
                # Show citations if present and role is assistant
                if msg["role"] == "assistant" and msg.get("citations"):
                    with st.expander("📚 View Source Citations"):
                        for idx, cit in enumerate(msg["citations"]):
                            st.markdown(f"""
                            <div class="citation-box">
                                <div class="citation-source">[{idx+1}] File: {cit['source']} | Page: {cit['page']}</div>
                                <div class="citation-snippet">"{cit['content']}"</div>
                            </div>
                            """, unsafe_allow_html=True)

        # Chat Input
        query = st.chat_input("Ask a question about your documents...")
        
        if query:
            # Display user message
            with st.chat_message("user"):
                st.write(query)
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            # Retrieve LLM and vector store
            try:
                llm = rag_backend.get_llm(
                    provider=provider,
                    api_key=api_key,
                    model=llm_model,
                    base_url=base_url,
                    temperature=temperature
                )
                
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing documents to find answer..."):
                        answer, citations = rag_backend.query_rag(
                            vector_store=st.session_state.vector_store,
                            query=query,
                            llm=llm,
                            k=top_k
                        )
                        st.write(answer)
                        
                        # Show citation expander
                        if citations:
                            with st.expander("📚 View Source Citations"):
                                for idx, cit in enumerate(citations):
                                    st.markdown(f"""
                                    <div class="citation-box">
                                        <div class="citation-source">[{idx+1}] File: {cit['source']} | Page: {cit['page']}</div>
                                        <div class="citation-snippet">"{cit['content']}"</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations
                })
            except Exception as e:
                st.error(f"Error querying chatbot: {str(e)}")
                if provider == "OpenAI" and not api_key:
                    st.warning("Hint: Did you forget to enter your OpenAI API key in the sidebar?")
                elif provider == "Ollama (Local)":
                    st.warning("Hint: Make sure your Ollama app is running locally, and the model is downloaded via `ollama run <model>`.")

# Tab 2: Semantic Search Tab
with tab2:
    st.markdown("### 🔍 Standalone Semantic Search")
    st.markdown("Find the exact matching text segments from your files directly by comparing vector embeddings (no LLM synthesis).")
    
    if not st.session_state.vector_store:
        st.info("Please index your documents in the sidebar first to run a semantic search.")
    else:
        search_query = st.text_input("Enter search query or keywords:")
        num_results = st.slider("Number of matches to return", min_value=1, max_value=15, value=5)
        
        if search_query:
            with st.spinner("Searching vector index..."):
                try:
                    # Do direct similarity search
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": num_results})
                    matched_docs = retriever.invoke(search_query)
                    
                    if not matched_docs:
                        st.write("No matching documents found.")
                    else:
                        st.success(f"Found {len(matched_docs)} matching snippets:")
                        for idx, doc in enumerate(matched_docs):
                            src = os.path.basename(doc.metadata.get("source", "unknown"))
                            page = doc.metadata.get("page", 0) + 1
                            content = doc.page_content.strip()
                            
                            st.markdown(f"""
                            <div style="border: 1px solid rgba(139, 92, 246, 0.2); border-radius: 8px; padding: 1rem; margin-bottom: 1rem; background-color: rgba(139, 92, 246, 0.02);">
                                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(0, 0, 0, 0.05); padding-bottom: 0.5rem; margin-bottom: 0.5rem;">
                                    <span style="font-weight: 600; color: #8B5CF6;">Match #{idx+1}</span>
                                    <span style="font-size: 0.85rem; color: #6B7280;">📄 {src} (Page {page})</span>
                                </div>
                                <p style="font-size: 0.9rem; font-style: italic; line-height: 1.5; margin: 0;">"{content}"</p>
                            </div>
                            """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error performing semantic search: {str(e)}")

# Tab 3: Document Manager Tab
with tab3:
    st.markdown("### 📄 Managed Documents")
    if not st.session_state.processed_files:
        st.info("No documents uploaded yet.")
    else:
        st.markdown("The following documents are currently indexed in FAISS:")
        for idx, filename in enumerate(st.session_state.processed_files):
            st.markdown(f"**{idx+1}. {filename}**")
            
        st.markdown("---")
        st.markdown("### 💡 Tips for RAG Optimization")
        st.markdown("""
        * **Chunk Size**: Smaller chunk sizes (e.g. 500 characters) capture specific details better but lose context. Larger chunk sizes (e.g. 1500 characters) preserve context but can dilute specific facts.
        * **Chunk Overlap**: Ensure you have some overlap (100-200 characters) so that information split across chunk boundaries isn't lost.
        * **Top K**: Set Top K higher if you have a complex query that needs information compiled from multiple places in the document.
        """)
