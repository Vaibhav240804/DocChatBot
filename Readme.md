# RAG Documentation Chatbot 📚

A comprehensive RAG (Retrieval-Augmented Generation) chatbot for technical documentation with advanced features including:

- **Intelligent Metadata Extraction**: Automatically extracts title, author, department, topic, prerequisites, and redirects
- **LangGraph-powered Workflow**: Sophisticated query enhancement and document retrieval
- **Knowledge Graph Integration**: Maps relationships between documents and prerequisites
- **Memory-enabled Conversations**: Maintains context across chat sessions
- **Beautiful Streamlit UI**: Separate interfaces for document processing and chatting
- **Advanced Chunking**: Smart section-based document splitting
- **Metadata Filtering**: Filter search results by extracted metadata

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the files
# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

```bash
# Copy environment template
cp .env.template .env

# Edit .env and add your Groq API key
# Get your API key from: https://console.groq.com/keys
GROQ_API_KEY=your_actual_api_key_here
```

### 3. Run the Application

```bash
streamlit run streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

## 📋 Features Overview

### Document Processing

- **One-click URL processing**: Just paste URLs and let the system handle the rest
- **Intelligent metadata extraction**: Uses LLM to extract structured metadata
- **Section-based chunking**: Splits documents by sections with proper metadata
- **Knowledge graph building**: Creates relationships between documents

### Chat Interface

- **Enhanced query processing**: Automatically improves user queries
- **Metadata-based filtering**: Uses document metadata to improve search
- **Prerequisite detection**: Shows required knowledge before topics
- **Related documents**: Discovers connected content through knowledge graph
- **Source attribution**: Shows exactly where information comes from
- **Conversational memory**: Remembers context throughout the chat session

### Advanced Retrieval

- **Dual query approach**: Searches with both original and enhanced queries
- **Vector similarity search**: Uses semantic embeddings for relevant results
- **Graph-based relationships**: Leverages document connections for better context
- **Confidence scoring**: Shows how relevant each source is

## 🏗️ Architecture

### Core Components

1. **MetadataExtractor**: Intelligently extracts metadata using both traditional parsing and LLM analysis
2. **VectorStoreManager**: Handles document chunking, embedding, and storage in ChromaDB
3. **RAGAgent**: LangGraph-powered agent for query processing and response generation
4. **StreamlitApp**: Beautiful UI with separate interfaces for processing and chatting

### Data Flow

```
URLs → Scraping → Metadata Extraction → Chunking → Embedding → Vector Store
                                    ↓
User Query → Query Enhancement → Metadata Filtering → Retrieval → Answer Generation
```

### Knowledge Graph

The system builds a NetworkX graph with:

- **Document nodes**: Main documentation pages
- **Section nodes**: Individual sections within documents
- **Prerequisite nodes**: Required knowledge or dependencies
- **Relationships**: "contains", "prerequisite_for", "redirects_to"

## 📁 File Structure

```
rag-chatbot/
├── config.py                 # Configuration settings
├── metadata_extractor.py     # Intelligent metadata extraction
├── vector_store_manager.py   # Vector store and graph management
├── rag_agent.py              # LangGraph RAG agent with memory
├── streamlit_app.py          # Beautiful Streamlit UI
├── requirements.txt          # Python dependencies
├── .env.template             # Environment variables template
└── README.md                 # This file
```

## 🔧 Configuration

All configuration is handled in `config.py`. Key settings:

- **Models**: Groq LLM model and embedding model
- **Chunking**: Chunk size and overlap for document splitting
- **Retrieval**: Number of documents to retrieve and similarity threshold
- **Memory**: Chat history management settings

## 📖 Usage Guide

### 1. Document Processing

1. Navigate to the "Document Processing" page
2. Enter URLs (one per line) in the text area
3. Click "Process URLs"
4. Wait for processing to complete
5. Review the results and statistics

**Supported URL Types:**

- Documentation websites
- API references
- Technical guides
- Knowledge base articles
- How-to tutorials

### 2. Chatting

1. Go to the "Chat Interface" page
2. Type your question in the chat input
3. The system will:
   - Enhance your query for better results
   - Extract relevant metadata filters
   - Retrieve matching documents
   - Find related content and prerequisites
   - Generate a comprehensive answer

**Chat Features:**

- View enhanced queries and applied filters
- See prerequisites for topics
- Explore related documents
- Check sources with confidence scores
- Persistent conversation memory

### 3. System Monitoring

The "System Stats" page shows:

- Vector store statistics
- Knowledge graph metrics
- Configuration details
- Performance information

## 🛠️ Customization

### Adding New Metadata Fields

1. Update `METADATA_FIELDS` in `config.py`
2. Modify the extraction prompts in `MetadataExtractor`
3. Update the UI display logic in `streamlit_app.py`

### Changing Models

- **LLM Model**: Update `GROQ_MODEL` in config or environment
- **Embedding Model**: Update `EMBEDDING_MODEL` (requires rebuilding vector store)

### Custom Chunking Strategy

Modify the `extract_sections()` method in `MetadataExtractor` to implement custom section splitting logic.

### UI Customization

The Streamlit app uses custom CSS for styling. Modify the CSS in `streamlit_app.py` to change the appearance.

## 🔍 Troubleshooting

### Common Issues

**"No documents found"**

- Make sure you've processed URLs first
- Check that URLs are accessible and contain content
- Verify the vector store was created successfully

**"Error processing URLs"**

- Check your internet connection
- Verify URLs are valid and accessible
- Some sites may block automated scraping

**"Groq API errors"**

- Verify your API key is correct
- Check your Groq account usage limits
- Ensure you have sufficient credits

**Memory issues with large documents**

- Reduce `CHUNK_SIZE` in config
- Process fewer URLs at once
- Increase `TOP_K_DOCUMENTS` for more focused results

### Performance Optimization

- Use SSD storage for vector database
- Increase available RAM for better performance
- Consider using GPU for embeddings (modify requirements.txt)
- Batch process URLs instead of processing individually

## 🤝 Contributing

This is a comprehensive RAG system with many extension possibilities:

- Add support for more document formats (PDF, DOCX, etc.)
- Implement advanced graph algorithms for better relationship discovery
- Add multilingual support
- Create API endpoints for programmatic access
- Implement user authentication and document access controls

## 📄 License

This project is open source. Feel free to use, modify, and distribute according to your needs.

## 🙏 Acknowledgments

Built with:

- [LangChain](https://python.langchain.com/) for RAG pipeline
- [LangGraph](https://python.langchain.com/docs/langgraph) for agent workflows
- [Groq](https://groq.com/) for fast LLM inference
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Streamlit](https://streamlit.io/) for the beautiful UI
- [Sentence Transformers](https://www.sbert.net/) for embeddings

## 🚀 Next Steps

After setup, try these example use cases:

1. **Technical Documentation**: Process your company's API docs or user guides
2. **Educational Content**: Add course materials or tutorials
3. **Knowledge Base**: Import FAQ or support articles
4. **Research Papers**: Process academic or technical papers
5. **Code Documentation**: Add developer documentation and guides

The system is designed to handle complex, interconnected documentation with sophisticated retrieval capabilities. Enjoy building your intelligent documentation assistant! 🎉
