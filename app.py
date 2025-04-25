import os
import PyPDF2
import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools import ArxivQueryRun
from dotenv import load_dotenv
from huggingface_hub import login
import uuid

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

# Configure Google Generative AI with your API key
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)

# Login to Hugging Face if token is available
if huggingface_token:
    login(token=huggingface_token)

# Initialize ChromaDB and other tools
client = chromadb.PersistentClient(path="chroma_db")
text_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
arxiv_tool = ArxivQueryRun()

def extract_text_from_pdfs(uploaded_files):
    all_text = ""
    for uploaded_file in uploaded_files:
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            all_text += page.extract_text() or ""
    return all_text 

def process_text_and_store(text, collection_name=None):
    """Process text into chunks, embed, and store in ChromaDB"""
    if collection_name is None:
        collection_name = f"collection_{uuid.uuid4().hex[:8]}"
    
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    
    # Split text into chunks
    chunks = text_splitter.split_text(text)
    
    # Get or create collection
    try:
        collection = client.get_collection(name=collection_name)
    except:
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    # Embed and store chunks
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    embeddings = [text_embedding_model.encode(chunk).tolist() for chunk in chunks]
    
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks
    )
    
    return collection

def semantic_search(query, collection, top_k=2):
    query_embedding = text_embedding_model.encode(query)
    results = collection.query(
        query_embeddings=[query_embedding.tolist()], n_results=top_k
    )
    return results

def generate_response(query, context):
    prompt = f"""
You are a research assistant helping to answer questions about academic papers.
Use the provided context to answer the query accurately and concisely.
If the information isn't available in the context, say so clearly.

Query: {query}

Context: {context}

Answer:
"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

def main():
    st.title("RAG-powered Research Paper Assistant")
    
    # Option to choose between PDF upload and arXiv search
    option = st.radio("Choose an option:", ("Upload PDFs", "Search arXiv"))
    
    if option == "Upload PDFs":
        uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])
        
        if uploaded_files:
            if st.button("Process PDFs"):
                st.write("Processing uploaded files...")
                all_text = extract_text_from_pdfs(uploaded_files)
                
                if all_text:
                    collection_name = f"pdf_collection_{uuid.uuid4().hex[:8]}"
                    collection = process_text_and_store(all_text, collection_name)
                    st.session_state["current_collection"] = collection
                    st.session_state["collection_name"] = collection_name
                    st.success("PDF content processed and stored successfully!")
                else:
                    st.error("Could not extract text from the uploaded PDFs")
            
            if "current_collection" in st.session_state:
                query = st.text_input("Enter your query about the papers:")
                if st.button("Execute Query") and query:
                    results = semantic_search(query, st.session_state["current_collection"])
                    if results and results['documents'] and results['documents'][0]:
                        context = "\n\n".join(results['documents'][0])
                        response = generate_response(query, context)
                        st.subheader("Generated Response:")
                        st.write(response)
                    else:
                        st.warning("No relevant context found for your query")
    
    elif option == "Search arXiv":
        query = st.text_input("Enter your search query for arXiv:")
        
        if st.button("Search arXiv") and query:
            with st.spinner("Searching arXiv..."):
                try:
                    arxiv_results = arxiv_tool.invoke(query)
                    st.session_state["arxiv_results"] = arxiv_results
                    st.subheader("Search Results:")
                    st.write(arxiv_results)
                    
                    # Process and store the arxiv results
                    collection_name = f"arxiv_collection_{uuid.uuid4().hex[:8]}"
                    collection = process_text_and_store(arxiv_results, collection_name)
                    st.session_state["current_collection"] = collection
                    st.session_state["collection_name"] = collection_name
                    st.success("arXiv paper content processed and stored successfully!")
                except Exception as e:
                    st.error(f"Error searching arXiv: {e}")
        
        # Only allow querying if search has been performed
        if "current_collection" in st.session_state and "arxiv_results" in st.session_state:
            paper_query = st.text_input("Ask a question about the paper:")
            if st.button("Execute Query on Paper") and paper_query:
                results = semantic_search(paper_query, st.session_state["current_collection"])
                if results and results['documents'] and results['documents'][0]:
                    context = "\n\n".join(results['documents'][0])
                    response = generate_response(paper_query, context)
                    st.subheader("Generated Response:")
                    st.write(response)
                else:
                    st.warning("No relevant context found for your query")

if __name__ == "__main__":
    main()