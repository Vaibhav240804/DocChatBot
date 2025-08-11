import chromadb
from chromadb.config import Settings
import os
import json
import networkx as nx
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import pickle
from config import Config
from metadata_extractor import MetadataExtractor

class VectorStoreManager:
    def __init__(self):
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize ChromaDB
        os.makedirs(Config.VECTOR_DB_PATH, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(
            path=Config.VECTOR_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize vector store
        self.vector_store = Chroma(
            client=self.chroma_client,
            collection_name=Config.COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=Config.VECTOR_DB_PATH
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize graph for relationships
        self.knowledge_graph = nx.DiGraph()
        self.graph_path = os.path.join(Config.GRAPH_DB_PATH, "knowledge_graph.pkl")
        os.makedirs(Config.GRAPH_DB_PATH, exist_ok=True)
        self._load_graph()
        
        # Initialize metadata extractor
        self.metadata_extractor = MetadataExtractor()

    def _load_graph(self):
        """Load knowledge graph from disk"""
        if os.path.exists(self.graph_path):
            try:
                with open(self.graph_path, 'rb') as f:
                    self.knowledge_graph = pickle.load(f)
                print(f"Loaded knowledge graph with {self.knowledge_graph.number_of_nodes()} nodes")
            except Exception as e:
                print(f"Error loading graph: {e}")
                self.knowledge_graph = nx.DiGraph()

    def _save_graph(self):
        """Save knowledge graph to disk"""
        try:
            with open(self.graph_path, 'wb') as f:
                pickle.dump(self.knowledge_graph, f)
            print(f"Saved knowledge graph with {self.knowledge_graph.number_of_nodes()} nodes")
        except Exception as e:
            print(f"Error saving graph: {e}")

    def process_urls(self, urls: List[str]) -> Dict[str, Any]:
        """Process a list of URLs and create embeddings"""
        results = {
            "processed_urls": 0,
            "total_chunks": 0,
            "failed_urls": [],
            "success_urls": []
        }
        
        all_documents = []
        all_metadatas = []
        all_ids = []
        
        for url in urls:
            try:
                print(f"Processing URL: {url}")
                
                # Extract metadata and content
                document_metadata = self.metadata_extractor.extract_full_metadata(url)
                
                if not document_metadata.get('text_content'):
                    results["failed_urls"].append(url)
                    continue
                
                # Split into sections
                sections = self.metadata_extractor.extract_sections(
                    document_metadata['text_content'], 
                    document_metadata
                )
                
                # Process each section
                for i, section in enumerate(sections):
                    text_content = section.get('text_content', '')
                    
                    if len(text_content.strip()) < 50:  # Skip very short sections
                        continue
                    
                    # Split into chunks
                    chunks = self.text_splitter.split_text(text_content)
                    
                    for j, chunk in enumerate(chunks):
                        if len(chunk.strip()) < 50:  # Skip very short chunks
                            continue
                            
                        # Prepare metadata for this chunk
                        chunk_metadata = section.copy()
                        chunk_metadata.pop('text_content', None)  # Remove text from metadata
                        
                        # Convert lists to strings for ChromaDB compatibility
                        for key, value in chunk_metadata.items():
                            if isinstance(value, list):
                                # Convert list to comma-separated string
                                chunk_metadata[key] = ', '.join(str(item) for item in value) if value else ''
                            elif value is None:
                                chunk_metadata[key] = ''
                            else:
                                chunk_metadata[key] = str(value)
                        
                        chunk_metadata['chunk_id'] = j
                        chunk_metadata['total_chunks'] = len(chunks)
                        chunk_metadata['section_index'] = i
                        
                        # Create unique ID
                        chunk_id = f"{url}#{i}#{j}"
                        
                        all_documents.append(chunk)
                        all_metadatas.append(chunk_metadata)
                        all_ids.append(chunk_id)
                
                # Add to knowledge graph
                self._add_to_knowledge_graph(document_metadata, sections)
                
                results["processed_urls"] += 1
                results["success_urls"].append(url)
                
            except Exception as e:
                print(f"Error processing URL {url}: {str(e)}")
                results["failed_urls"].append(url)
        
        # Add all documents to vector store
        if all_documents:
            try:
                self.vector_store.add_texts(
                    texts=all_documents,
                    metadatas=all_metadatas,
                    ids=all_ids
                )
                results["total_chunks"] = len(all_documents)
                print(f"Added {len(all_documents)} chunks to vector store")
            except Exception as e:
                print(f"Error adding to vector store: {str(e)}")
        
        # Save the graph
        self._save_graph()
        
        return results

    def _add_to_knowledge_graph(self, document_metadata: Dict, sections: List[Dict]):
        """Add document and relationships to knowledge graph"""
        try:
            url = document_metadata.get('url', '')
            title = document_metadata.get('title', 'Unknown')
            topic = document_metadata.get('topic', '')
            
            # Prepare node attributes (avoid keyword conflicts)
            node_attrs = {}
            for k, v in document_metadata.items():
                if v and k not in ['text_content', 'title', 'topic']:  # Exclude title and topic to avoid conflict
                    # Convert lists to strings for storage
                    if isinstance(v, list):
                        node_attrs[f"{k}_list"] = str(v)
                    else:
                        node_attrs[k] = str(v)
            
            # Add main document node
            self.knowledge_graph.add_node(
                url,
                node_title=title,  # Use node_title instead of title
                topic=topic if len(topic) == 1 else ' '.join(topic),
                node_type='document',
                **node_attrs
            )
            
            # Add section nodes and relationships
            for section in sections:
                section_id = section.get('section_id', '')
                section_title = section.get('section', 'Unknown Section')
                
                if section_id:
                    # Prepare section attributes
                    section_attrs = {}
                    for k, v in section.items():
                        if v and k not in ['text_content', 'title', 'section_id']:
                            if isinstance(v, list):
                                section_attrs[f"{k}_list"] = str(v)
                            else:
                                section_attrs[k] = str(v)
                    
                    self.knowledge_graph.add_node(
                        section_id,
                        node_title=section_title,
                        node_type='section',
                        parent_document=url,
                        **section_attrs
                    )
                    
                    # Add edge from document to section
                    self.knowledge_graph.add_edge(url, section_id, relationship='contains')
            
            # Add prerequisite relationships
            prerequisites = document_metadata.get('prerequisites', [])
            if prerequisites and isinstance(prerequisites, list):
                for prereq in prerequisites:
                    if isinstance(prereq, str) and prereq.strip():
                        prereq_node = f"prerequisite:{prereq.strip()}"
                        self.knowledge_graph.add_node(
                            prereq_node,
                            node_title=prereq.strip(),
                            node_type='prerequisite'
                        )
                        self.knowledge_graph.add_edge(prereq_node, url, relationship='prerequisite_for')
            
            # Add redirect relationships
            redirects = document_metadata.get('redirects', [])
            if redirects and isinstance(redirects, list):
                for redirect in redirects:
                    if isinstance(redirect, str) and redirect.strip():
                        self.knowledge_graph.add_edge(url, redirect, relationship='redirects_to')
            
        except Exception as e:
            print(f"Error adding to knowledge graph: {str(e)}")

    def search_similar(self, query: str, filters: Optional[Dict] = None, k: int = None) -> List[Dict]:
        """Search for similar documents with optional metadata filters"""
        if k is None:
            k = Config.TOP_K_DOCUMENTS
        
        try:
            # Build metadata filter for ChromaDB
            where_clause = {}
            if filters:
                for key, value in filters.items():
                    if value and key in Config.METADATA_FIELDS:
                        if isinstance(value, list):
                            # For list fields, search if any item is contained in the string
                            # Since we stored lists as comma-separated strings
                            if value:
                                # Use $contains operator for string matching
                                search_term = str(value[0])  # Use first item for now
                                where_clause[key] = {"$contains": search_term}
                        else:
                            where_clause[key] = {"$contains": str(value)}
            
            # Perform similarity search
            if where_clause:
                results = self.vector_store.similarity_search_with_score(
                    query, 
                    k=k, 
                    filter=where_clause
                )
            else:
                results = self.vector_store.similarity_search_with_score(query, k=k)
            
            # Format results
            formatted_results = []
            for doc, score in results:
                if score >= Config.SIMILARITY_THRESHOLD:
                    result = {
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'similarity_score': score
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"Error in similarity search: {str(e)}")
            # Fallback to simple search without filters
            try:
                results = self.vector_store.similarity_search_with_score(query, k=k)
                formatted_results = []
                for doc, score in results:
                    result = {
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'similarity_score': score
                    }
                    formatted_results.append(result)
                return formatted_results
            except Exception as e2:
                print(f"Error in fallback search: {str(e2)}")
                return []

    def get_related_documents(self, url: str, relationship_type: str = None) -> List[Dict]:
        """Get related documents from knowledge graph"""
        try:
            if url not in self.knowledge_graph:
                return []
            
            related = []
            
            if relationship_type:
                # Get specific relationship type
                for neighbor in self.knowledge_graph.neighbors(url):
                    edge_data = self.knowledge_graph[url][neighbor]
                    if edge_data.get('relationship') == relationship_type:
                        node_data = self.knowledge_graph.nodes[neighbor]
                        related.append({
                            'id': neighbor,
                            'relationship': relationship_type,
                            'title': node_data.get('node_title', 'Unknown'),
                            **{k: v for k, v in node_data.items() if k != 'node_title'}
                        })
            else:
                # Get all related nodes
                for neighbor in self.knowledge_graph.neighbors(url):
                    edge_data = self.knowledge_graph[url][neighbor]
                    node_data = self.knowledge_graph.nodes[neighbor]
                    related.append({
                        'id': neighbor,
                        'relationship': edge_data.get('relationship', 'unknown'),
                        'title': node_data.get('node_title', 'Unknown'),
                        **{k: v for k, v in node_data.items() if k != 'node_title'}
                    })
            
            return related
            
        except Exception as e:
            print(f"Error getting related documents: {str(e)}")
            return []

    def get_prerequisites_chain(self, url: str) -> List[Dict]:
        """Get prerequisite chain for a document"""
        try:
            prerequisites = []
            
            # Find all prerequisite nodes pointing to this document
            for node in self.knowledge_graph.nodes():
                if self.knowledge_graph.has_edge(node, url):
                    edge_data = self.knowledge_graph[node][url]
                    if edge_data.get('relationship') == 'prerequisite_for':
                        node_data = self.knowledge_graph.nodes[node]
                        prerequisites.append({
                            'id': node,
                            'title': node_data.get('node_title', 'Unknown'),
                            **{k: v for k, v in node_data.items() if k != 'node_title'}
                        })
            
            return prerequisites
            
        except Exception as e:
            print(f"Error getting prerequisites chain: {str(e)}")
            return []

    def get_collection_stats(self) -> Dict:
        """Get statistics about the vector store collection"""
        try:
            collection = self.chroma_client.get_collection(Config.COLLECTION_NAME)
            count = collection.count()
            
            return {
                "total_documents": count,
                "collection_name": Config.COLLECTION_NAME,
                "embedding_model": Config.EMBEDDING_MODEL,
                "graph_nodes": self.knowledge_graph.number_of_nodes(),
                "graph_edges": self.knowledge_graph.number_of_edges()
            }
        except Exception as e:
            print(f"Error getting collection stats: {str(e)}")
            return {"error": str(e)}
        
    def get_all_graph_nodes(self) -> List[Dict]:
        """Get all nodes in the knowledge graph"""
        try:
            nodes = []
            for node in self.knowledge_graph.nodes():
                node_data = self.knowledge_graph.nodes[node]
                nodes.append({
                    'id': node,
                    'title': node_data.get('node_title', 'Unknown'),
                    **{k: v for k, v in node_data.items() if k != 'node_title'}
                })
            return nodes
        except Exception as e:
            print(f"Error getting all graph nodes: {str(e)}")
            return []