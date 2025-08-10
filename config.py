import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # Model Configuration
    GROQ_MODEL = "llama3-8b-8192"  # You can change this to other Groq models
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    # Vector Store Configuration
    VECTOR_DB_PATH = "./vector_db"
    COLLECTION_NAME = "technical_docs"
    
    # Graph Database (Optional - using NetworkX for now)
    GRAPH_DB_PATH = "./graph_db"
    
    # Chunking Configuration
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Retrieval Configuration
    TOP_K_DOCUMENTS = 5
    SIMILARITY_THRESHOLD = 0.3
    
    # Memory Configuration
    MEMORY_KEY = "chat_history"
    MAX_MEMORY_TOKENS = 4000
    
    # Web Scraping Configuration
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3
    
    # Metadata Fields
    METADATA_FIELDS = [
        "title",
        "author", 
        "department",
        "topic",
        "prerequisites",
        "redirects",
        "url",
        "section",
        "subsection"
    ]