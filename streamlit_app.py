import streamlit as st
import json
from typing import Dict, List
import time
import os
from datetime import datetime
from config import Config
from vector_store_manager import VectorStoreManager
from rag_agent import RAGAgent

# Page config
st.set_page_config(
    page_title="RAG Documentation Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .stats-container {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .source-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .prereq-card {
        background: #fff3cd;
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 0.3rem 0;
    }
    
    .related-card {
        background: #d1ecf1;
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 0.3rem 0;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    
    .url-input {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vector_manager' not in st.session_state:
    st.session_state.vector_manager = VectorStoreManager()

if 'rag_agent' not in st.session_state:
    st.session_state.rag_agent = RAGAgent()

if 'processing_status' not in st.session_state:
    st.session_state.processing_status = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'session_memory' not in st.session_state:
    st.session_state.session_memory = []

def main():
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>📚 RAG Documentation Assistant</h1>
        <p>Advanced Technical Documentation Chat with Intelligent Retrieval</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for navigation
    with st.sidebar:
        st.markdown("### 🔧 Navigation")
        page = st.radio(
            "Select Page:",
            ["💬 Chat Interface", "📥 Document Processing", "📊 System Stats"],
            index=0
        )
        
        st.markdown("---")
        
        # API Key status
        if Config.GROQ_API_KEY:
            st.success("✅ Groq API Key Configured")
        else:
            st.error("❌ Groq API Key Missing")
            st.info("Please set GROQ_API_KEY in your environment variables")
        
        st.markdown("---")
        
        # Quick stats
        stats = st.session_state.rag_agent.get_stats()
        st.markdown("### 📈 Quick Stats")
        if 'total_documents' in stats:
            st.metric("Documents", stats['total_documents'])
            st.metric("Graph Nodes", stats.get('graph_nodes', 0))
        
        # Clear memory button
        if st.button("🗑️ Clear Chat Memory"):
            st.session_state.chat_history = []
            st.session_state.session_memory = []
            st.success("Chat memory cleared!")
    
    # Main content based on selected page
    if page == "💬 Chat Interface":
        show_chat_interface()
    elif page == "📥 Document Processing":
        show_document_processing()
    elif page == "📊 System Stats":
        show_system_stats()

def show_chat_interface():
    st.markdown("## 💬 Chat with Your Documentation")
    
    # Check if documents are available
    stats = st.session_state.rag_agent.get_stats()
    if stats.get('total_documents', 0) == 0:
        st.warning("⚠️ No documents found in the knowledge base. Please add documents first using the Document Processing page.")
        return
    
    # Chat input
    with st.container():
        user_input = st.chat_input("Ask me anything about your documentation...")
        
        if user_input:
            process_chat_message(user_input)
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            display_message(message)

def process_chat_message(user_input: str):
    """Process user input and generate response"""
    
    # Add user message to history
    user_message = {
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }
    st.session_state.chat_history.append(user_message)
    
    # Show processing indicator
    with st.spinner("🔍 Searching knowledge base and generating response..."):
        try:
            # Process query with RAG agent
            response = st.session_state.rag_agent.process_query(
                user_input, 
                st.session_state.session_memory
            )
            
            # Update session memory
            st.session_state.session_memory = response.get('memory', [])
            
            # Create assistant message
            assistant_message = {
                "role": "assistant",
                "content": response['answer'],
                "sources": response['sources'],
                "related_documents": response['related_documents'],
                "prerequisites": response['prerequisites'],
                "enhanced_query": response['enhanced_query'],
                "metadata_filters": response['metadata_filters'],
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
            
            st.session_state.chat_history.append(assistant_message)
            
        except Exception as e:
            st.error(f"Error processing your request: {str(e)}")
            error_message = {
                "role": "assistant",
                "content": "I apologize, but I encountered an error while processing your request. Please try again.",
                "sources": [],
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
            st.session_state.chat_history.append(error_message)
    
    # Rerun to update the display
    st.rerun()

def display_message(message: Dict):
    """Display a chat message with proper formatting"""
    
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(f"**{message['timestamp']}**")
            st.markdown(message["content"])
    
    else:  # assistant message
        with st.chat_message("assistant"):
            st.markdown(f"**{message['timestamp']}**")
            st.markdown(message["content"])
            
            # Show enhanced query if available
            if message.get('enhanced_query') and message['enhanced_query'] != message.get('original_query'):
                with st.expander("🔍 Query Enhancement Details"):
                    st.markdown(f"**Enhanced Query:** {message['enhanced_query']}")
                    if message.get('metadata_filters'):
                        st.markdown(f"**Applied Filters:** {message['metadata_filters']}")
            
            # Show prerequisites
            if message.get('prerequisites'):
                with st.expander("⚠️ Prerequisites", expanded=False):
                    for prereq in message['prerequisites']:
                        st.markdown(f"""
                        <div class="prereq-card">
                            <strong>{prereq.get('title', 'Unknown')}</strong>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Show related documents
            if message.get('related_documents'):
                with st.expander("🔗 Related Documents", expanded=False):
                    for doc in message['related_documents']:
                        st.markdown(f"""
                        <div class="related-card">
                            <strong>{doc.get('title', 'Unknown')}</strong><br>
                            <em>Relationship: {doc.get('relationship', 'related')}</em>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Show sources
            if message.get('sources'):
                with st.expander("📚 Sources", expanded=True):
                    for i, source in enumerate(message['sources'], 1):
                        similarity_score = source.get('similarity_score', 0)
                        confidence = f"{similarity_score:.2f}" if similarity_score > 0 else "N/A"
                        
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>Source {i}: {source.get('title', 'Unknown Document')}</strong><br>
                            <em>Section: {source.get('section', 'Main Content')}</em><br>
                            <em>Confidence: {confidence}</em><br>
                            {f'<a href="{source["url"]}" target="_blank">🔗 View Document</a>' if source.get('url') else ''}
                        </div>
                        """, unsafe_allow_html=True)

def show_document_processing():
    st.markdown("## 📥 Document Processing")
    
    st.markdown("""
    Add technical documentation URLs to your knowledge base. The system will:
    - Extract metadata (title, author, department, topic, prerequisites)
    - Create semantic embeddings for intelligent search
    - Build a knowledge graph of document relationships
    """)
    
    # URL input section
    st.markdown("### 🌐 Add URLs")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Text area for URLs
        urls_input = st.text_area(
            "Enter URLs (one per line):",
            placeholder="https://docs.example.com/guide1\nhttps://docs.example.com/guide2",
            height=200
        )
    
    with col2:
        st.markdown("**Examples:**")
        st.markdown("""
        - Documentation sites
        - API references  
        - Technical guides
        - How-to articles
        - Knowledge bases
        """)
    
    # Process button
    process_col1, process_col2, process_col3 = st.columns([1, 2, 1])
    with process_col2:
        if st.button("🚀 Process URLs", type="primary", use_container_width=True):
            if urls_input.strip():
                process_urls(urls_input)
            else:
                st.warning("Please enter at least one URL.")
    
    # Show processing status
    if st.session_state.processing_status:
        display_processing_status()

def process_urls(urls_input: str):
    """Process the input URLs"""
    # Parse URLs
    urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
    
    if not urls:
        st.warning("No valid URLs found.")
        return
    
    # Validate URLs
    valid_urls = []
    for url in urls:
        if url.startswith(('http://', 'https://')):
            valid_urls.append(url)
        else:
            st.warning(f"Invalid URL format: {url}")
    
    if not valid_urls:
        st.error("No valid URLs to process.")
        return
    
    # Process URLs
    with st.spinner(f"Processing {len(valid_urls)} URLs..."):
        start_time = time.time()
        
        try:
            results = st.session_state.vector_manager.process_urls(valid_urls)
            
            processing_time = time.time() - start_time
            
            # Store results in session state
            st.session_state.processing_status = {
                'results': results,
                'processing_time': processing_time,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            st.success(f"Processing completed in {processing_time:.2f} seconds!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error processing URLs: {str(e)}")

def display_processing_status():
    """Display the results of URL processing"""
    status = st.session_state.processing_status
    results = status['results']
    
    st.markdown("### 📊 Processing Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("✅ Processed URLs", results['processed_urls'])
    
    with col2:
        st.metric("📄 Total Chunks", results['total_chunks'])
    
    with col3:
        st.metric("❌ Failed URLs", len(results['failed_urls']))
    
    with col4:
        st.metric("⏱️ Processing Time", f"{status['processing_time']:.1f}s")
    
    # Success and failure details
    if results['success_urls']:
        with st.expander("✅ Successfully Processed URLs", expanded=True):
            for url in results['success_urls']:
                st.markdown(f"- {url}")
    
    if results['failed_urls']:
        with st.expander("❌ Failed URLs", expanded=False):
            for url in results['failed_urls']:
                st.markdown(f"- {url}")
    
    # Clear results button
    if st.button("Clear Results"):
        st.session_state.processing_status = None
        st.rerun()

def show_system_stats():
    st.markdown("## 📊 System Statistics")
    
    # Get comprehensive stats
    stats = st.session_state.rag_agent.get_stats()
    
    if 'error' in stats:
        st.error(f"Error loading stats: {stats['error']}")
        return
    
    # Display stats in cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="stats-container">
            <h3>📚 Vector Store</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("Total Documents", stats.get('total_documents', 0))
        st.metric("Collection Name", stats.get('collection_name', 'N/A'))
        st.metric("Embedding Model", stats.get('embedding_model', 'N/A'))
    
    with col2:
        st.markdown("""
        <div class="stats-container">
            <h3>🕸️ Knowledge Graph</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("Graph Nodes", stats.get('graph_nodes', 0))
        st.metric("Graph Edges", stats.get('graph_edges', 0))
        
        # Calculate graph density if we have nodes
        nodes = stats.get('graph_nodes', 0)
        edges = stats.get('graph_edges', 0)
        if nodes > 1:
            max_edges = nodes * (nodes - 1)
            density = edges / max_edges if max_edges > 0 else 0
            st.metric("Graph Density", f"{density:.3f}")
    
    # Configuration info
    st.markdown("### ⚙️ Configuration")
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        st.markdown("**Model Configuration:**")
        st.text(f"LLM Model: {Config.GROQ_MODEL}")
        st.text(f"Embedding Model: {Config.EMBEDDING_MODEL}")
        st.text(f"Chunk Size: {Config.CHUNK_SIZE}")
        st.text(f"Chunk Overlap: {Config.CHUNK_OVERLAP}")
    
    with config_col2:
        st.markdown("**Retrieval Configuration:**")
        st.text(f"Top-K Documents: {Config.TOP_K_DOCUMENTS}")
        st.text(f"Similarity Threshold: {Config.SIMILARITY_THRESHOLD}")
        st.text(f"Memory Window: {Config.MAX_MEMORY_TOKENS}")

if __name__ == "__main__":
    main()