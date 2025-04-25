# RAG Research Paper Assistant

## Overview

RAG Research Paper Assistant is an interactive tool that uses Retrieval-Augmented Generation (RAG) to help researchers query and extract insights from academic papers. The application enables users to upload PDF papers or search for papers on arXiv, and then ask questions about the content using natural language. The system provides concise, relevant answers based on the paper's content.

## Features

- **PDF Upload**: Process and analyze local PDF research papers
- **arXiv Search**: Find and analyze papers directly from the arXiv repository
- **Natural Language Querying**: Ask questions about papers in plain English
- **AI-Powered Responses**: Get precise answers based on the paper's content

## Technology Stack

- **Frontend**: Streamlit for the interactive web interface
- **Embeddings**: Sentence Transformers for semantic text encoding
- **Vector Database**: ChromaDB for efficient storage and retrieval of text embeddings
- **Large Language Model**: Google's Gemini 1.5 Flash for generating responses
- **Document Processing**: PyPDF2 for PDF text extraction
- **Research Tools**: LangChain's ArxivQueryRun for searching academic papers

## Concepts

### Retrieval-Augmented Generation (RAG)

RAG combines retrieval-based and generation-based approaches to provide accurate and contextually relevant answers:

1. **Retrieval**: The system retrieves relevant passages from the document based on semantic similarity to the query
2. **Augmentation**: These passages are used to augment the prompt sent to the LLM
3. **Generation**: The LLM generates a response grounded in the retrieved information

This approach helps overcome the limitations of standard LLMs by:

- Providing up-to-date information from the source documents
- Reducing hallucinations by grounding responses in retrieved context
- Enabling precise citations and references to specific parts of documents

### Vector Similarity Search

1. **Text Chunking**: Documents are split into manageable chunks
2. **Embedding**: Each chunk is converted into a high-dimensional vector using a neural network
3. **Indexing**: Vectors are stored in ChromaDB for efficient retrieval
4. **Semantic Search**: User queries are converted to vectors and compared against stored vectors to find semantically similar content

## Installation and Setup

### Prerequisites

- Python 3.10+
- Conda

### Environment Setup

```bash
# Create a new conda environment
conda create -n rag_env python=3.10 -y
conda activate rag_env

# Install required packages
pip install streamlit
pip install sentence-transformers
pip install chromadb
pip install google-generativeai
pip install langchain
pip install langchain-community
pip install pypdf2
pip install python-dotenv
pip install huggingface_hub
```

### API Keys

Create a `.env` file in the project root with the following:

```
GEMINI_API_KEY=your_gemini_api_key_here
HUGGINGFACE_TOKEN=your_huggingface_token_here
```

## Usage

1. Navigate to your project directory:
   ```bash
   d:
   cd D:\Code\PROJECTS\RAG Research
   ```

2. Activate the conda environment:
   ```bash
   conda activate rag_env
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

4. Use the application:
   - Select "Upload PDFs" to process local files or "Search arXiv" to find papers online
   - Process the documents
   - Ask questions in natural language about the paper content

## How It Works

### PDF Processing Pipeline

1. **Text Extraction**: Extract raw text from uploaded PDF files
2. **Text Chunking**: Split text into semantically meaningful chunks
3. **Embedding Generation**: Convert text chunks into vector embeddings
4. **Vector Storage**: Store embeddings in ChromaDB for retrieval

### Query Processing Pipeline

1. **Query Embedding**: Convert user query into a vector embedding
2. **Semantic Search**: Find the most relevant text chunks using vector similarity
3. **Context Assembly**: Combine relevant chunks to create context for the LLM
4. **Response Generation**: Send query and context to Gemini to generate a human-readable response

## Future Enhancements

- Multi-document comparison and analysis
- Citation generation for answers
- Support for more file formats (DOCX, TXT, etc.)
- User authentication and session management
- Enhanced visualization of paper insights
- Batch processing of multiple papers

## License

This project is provided for educational and research purposes.
