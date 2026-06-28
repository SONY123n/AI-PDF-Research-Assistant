import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from rag.loader import PDFLoader
from rag.splitter import DocumentSplitter
from rag.embeddings import GeminiEmbeddings
from rag.vector_store import VectorStoreManager
from rag.chain import RAGChain

# Load environment variables (like GEMINI_API_KEY) from .env file
load_dotenv()

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Research Assistant")
st.markdown(
    "Upload one or more PDFs and ask questions about them using Gemini + FAISS."
)

# -------------------------
# Session State Initialization
# -------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# -------------------------
# Sidebar Layout
# -------------------------

with st.sidebar:
    st.header("Upload PDFs")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    process_button = st.button(
        "Process Documents",
        use_container_width=True
    )

# -------------------------
# Document Processing Backend
# -------------------------

if process_button:
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
        with st.spinner("Reading PDFs..."):
            temp_paths = []
            for uploaded_file in uploaded_files:
                temp = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                )
                temp.write(uploaded_file.read())
                temp.close()
                temp_paths.append(temp.name)

            loader = PDFLoader()
            documents = loader.load_multiple_pdfs(temp_paths)

        with st.spinner("Splitting Documents..."):
            splitter = DocumentSplitter()
            chunks = splitter.split_documents(documents)

        with st.spinner("Creating Embeddings..."):
            embedding_model = GeminiEmbeddings()

        with st.spinner("Building Vector Store..."):
            vector_manager = VectorStoreManager(embedding_model)
            vector_manager.create_vector_store(chunks)

            retriever = vector_manager.get_retriever()
            rag_chain = RAGChain(retriever)

            # Store instances in session state so they persist across reruns
            st.session_state.vector_store = vector_manager
            st.session_state.retriever = retriever
            st.session_state.rag_chain = rag_chain
            st.session_state.documents_loaded = True

        # Clean up temporary files on local disk
        for file in temp_paths:
            if os.path.exists(file):
                os.remove(file)

        st.success("Documents processed successfully!")

# -------------------------
# Display Past Chat History
# -------------------------

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if (
            message["role"] == "assistant"
            and "sources" in message
            and message["sources"]
        ):
            st.caption(
                "Source Pages: "
                + ", ".join(str(page) for page in message["sources"])
            )

# -------------------------
# Chat Input & RAG Execution
# -------------------------

question = st.chat_input("Ask a question about your uploaded PDFs...")

if question:
    if not st.session_state.documents_loaded:
        st.warning("Please upload and process your PDFs before asking questions.")
    else:
        # 1. Instantly display user question
        with st.chat_message("user"):
            st.markdown(question)
        
        st.session_state.chat_history.append({"role": "user", "content": question})

        # 2. Query the LLM chain and display response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Uses the correct .ask() method for your backend architecture
                    response = st.session_state.rag_chain.ask(question) 
                    
                    # Verify if response contains source chunks/metadata or just text
                    if isinstance(response, dict):
                        answer = response.get("answer", "No answer generated.")
                        sources = response.get("sources", [])
                    else:
                        answer = str(response)
                        sources = []

                    st.markdown(answer)
                    if sources:
                        st.caption("Source Pages: " + ", ".join(str(p) for p in sources))

                    # 3. Cache response to chat history state
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                except Exception as e:
                    st.error(f"An error occurred while generating an answer: {e}")