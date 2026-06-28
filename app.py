import streamlit as st
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables (for local testing, Streamlit Cloud will use Secrets)
load_dotenv()

st.set_page_config(page_title="AI PDF Research Assistant", layout="wide")
st.title("📚 AI PDF Research Assistant")
st.write("Upload your long-form PDF documents and get instant, grounded answers with page references.")

# Initialize session states
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# Sidebar for document uploads
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    process_button = st.button("Process Documents")

# Document Processing Block
if process_button and uploaded_files:
    with st.spinner("Processing documents... Splitting text and building vector database..."):
        try:
            temp_paths = []
            # Save uploaded files temporarily to read them
            for uploaded_file in uploaded_files:
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                temp_paths.append(temp_path)

            # Load and parse PDFs
            docs = []
            for path in temp_paths:
                loader = PyPDFLoader(path)
                docs.extend(loader.load())

            # Text Chunking Strategy
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_documents = text_splitter.split_documents(docs)

            # Generate Embeddings and create FAISS Vector Index
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.from_documents(final_documents, embeddings)
            retriever = vector_store.as_retriever(search_kwargs={"k": 4})

            # Initialize LLM and Prompt Engineering Templates
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
            
            system_prompt = (
                "You are an expert research assistant. Answer the user's question using only the provided context below. "
                "If the context does not contain the answer, say 'I cannot find the answer in the uploaded documents.' "
                "Do not make things up. Always keep your answers accurate and truthful.\n\n"
                "Context:\n{context}"
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])

            # Build the Final RAG Chain
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)

            # Save to Session State
            st.session_state.rag_chain = rag_chain
            st.session_state.documents_loaded = True
            
            # CRUCIAL FIX: Clear old chat history when a completely new PDF dataset is processed
            st.session_state.chat_history = []

            # Clean up temporary files
            for path in temp_paths:
                if os.path.exists(path):
                    os.remove(path)

            st.success("Documents processed successfully! Chat history has been reset for the new files.")

        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")

# Chat Interface Display
if st.session_state.documents_loaded:
    # Render historical conversation from cache
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new user input
    if user_query := st.chat_input("Ask a question about your documents:"):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # Generate response using the RAG chain
        with st.chat_message("assistant"):
            with st.spinner("Searching document index..."):
                try:
                    response = st.session_state.rag_chain.invoke({"input": user_query})
                    answer = response["answer"]
                    
                    # Extract page reference citations from context documents
                    source_documents = response.get("context", [])
                    citations = []
                    for doc in source_documents:
                        page_num = doc.metadata.get("page", 0) + 1  # 0-indexed to 1-indexed
                        file_name = os.path.basename(doc.metadata.get("source", "Document"))
                        citations.append(f"Page {page_num} of {file_name}")
                    
                    # Deduplicate and format citation string
                    if citations:
                        unique_citations = sorted(list(set(citations)))
                        answer += "\n\n**Sources Verified:**\n" + "\n".join([f"- {c}" for c in unique_citations])

                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})

                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
else:
    st.info("👈 Please upload your PDF documents in the sidebar and click 'Process Documents' to begin chatting.")