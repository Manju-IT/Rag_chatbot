# app.py - Enhanced with Memory and Context
import os
import tempfile
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
from langchain_groq import ChatGroq
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
# from langchain.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key='answer'
    )
    
    

# Configure the app
st.set_page_config(page_title="Document Chatbot", page_icon="📄", layout="wide")

# Sidebar for file upload and settings
with st.sidebar:
    st.title("📄 Document Chatbot")
    st.markdown("Upload documents and ask questions about their content")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload documents", 
        type=["pdf", "txt", "docx", "pptx"], 
        accept_multiple_files=True
    )
    
    # Model selection
    model_name = st.selectbox(
        "Select Groq model",
        ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"]
    )
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        k_slider = st.slider("Number of document chunks to retrieve", 2, 10, 4)
        memory_window = st.slider("Conversation memory window", 5, 20, 10)
        temperature = st.slider("Temperature (creativity)", 0.0, 1.0, 0.1)
    
    # Process button
    process_clicked = st.button("Process Documents")
    
    # Clear conversation button
    clear_chat = st.button("Clear Conversation")
    
    # Display processed files
    if st.session_state.processed_files:
        st.subheader("Processed Files")
        for file in st.session_state.processed_files:
            st.write(f"✅ {file}")

# Clear conversation history
if clear_chat:
    st.session_state.chat_history = []
    st.session_state.memory.clear()
    st.rerun()

# Main content area
st.title("💬 Document Chat")
st.caption("Ask questions about your uploaded documents - I'll remember our conversation")

# Initialize LLM
@st.cache_resource
def load_llm(model_name, temperature=0.1):
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name=model_name,
        temperature=temperature
    )

# Initialize embeddings
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )



def process_documents(files):
    documents = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(tmp_file_path)
            else:
                loader = UnstructuredFileLoader(tmp_file_path)
            
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["source"] = file.name
            documents.extend(loaded_docs)
            
            if file.name not in st.session_state.processed_files:
                st.session_state.processed_files.append(file.name)
                
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
        finally:
            os.unlink(tmp_file_path)
    
    if documents:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        
        embeddings = load_embeddings()
        
        st.session_state.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        
        st.success(f"Processed {len(documents)} documents with {len(chunks)} chunks!")
    return True


# Process documents when button is clicked
if process_clicked and uploaded_files:
    with st.spinner("Processing documents..."):
        process_documents(uploaded_files)

# Initialize Conversational QA chain with memory
def get_qa_chain():
    if st.session_state.vectorstore is None:
        return None
    
    # Create custom prompt with context about conversation history
    # prompt_template = """You are a helpful assistant that answers questions based on provided documents.
    # Use the following pieces of context to answer the question at the end. 
    # If you don't know the answer, just say that you don't know, don't try to make up an answer.
    # Always cite your sources by including the document name in your response.
    prompt_template = """You are a helpful assistant that answers questions based on provided documents.
    Use the following pieces of context to answer the question at the end. 
    If you dont know the answer, just fetch answer from web or internet and show source as from the Web.
    Always cite your sources by including the document name in your response.

    Previous conversation:
    {chat_history}

    Context: {context}

    Question: {question}
    Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question", "chat_history"]
    )
    
    llm = load_llm(model_name, temperature)
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": k_slider})
    
    # Create conversational chain with memory
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True,
        get_chat_history=lambda h: h,
        verbose=False
    )
    return chain

# Display chat messages
for i, msg in enumerate(st.session_state.chat_history):
    if msg["role"] == "user":
        message(msg["content"], is_user=True, key=f"user_{i}")
    else:
        message(msg["content"], key=f"ai_{i}")
        
        # Show sources if available
        if "source_documents" in msg and msg["source_documents"]:
            with st.expander("Source Documents"):
                for j, doc in enumerate(msg["source_documents"]):
                    st.write(f"**Source {j+1}:** {doc.metadata.get('source', 'Unknown')}")
                    st.write(f"**Content:** {doc.page_content[:200]}...")
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask a question about your documents"):
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    message(prompt, is_user=True)
    
    # Get response
    qa_chain = get_qa_chain()
    if qa_chain:
        with st.spinner("Thinking..."):
            try:
                # Format the conversation history for the chain
                formatted_history = []
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        formatted_history.append(f"Human: {msg['content']}")
                    else:
                        formatted_history.append(f"Assistant: {msg['content']}")
                
                # Get response from the chain
                result = qa_chain({
                    "question": prompt, 
                    "chat_history": "\n".join(formatted_history[-memory_window:])
                })
                
                response = result["answer"]
                source_docs = result.get("source_documents", [])
                
                # Add AI response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": response,
                    "source_documents": source_docs
                })
                
                # Display response
                message(response)
                
                # Display sources
                if source_docs:
                    with st.expander("Source Documents"):
                        for j, doc in enumerate(source_docs):
                            st.write(f"**Source {j+1}:** {doc.metadata.get('source', 'Unknown')}")
                            st.write(f"**Content:** {doc.page_content[:200]}...")
                            st.divider()
                        
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
    else:
        st.warning("Please process some documents first before asking questions.")

# Display memory status in sidebar
with st.sidebar:
    st.divider()
    st.write(f"**Conversation memory:** {len(st.session_state.chat_history)} messages")
    if st.button("View Memory Details"):
        st.write(st.session_state.memory.load_memory_variables({}))


