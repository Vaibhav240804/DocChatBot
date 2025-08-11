from typing import Dict, List, Any, Annotated, TypedDict
from langgraph.graph import StateGraph, END, START
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage
import json
import re
from config import Config
from vector_store_manager import VectorStoreManager

class AgentState(TypedDict):
    messages: List[Dict]
    query: str
    enhanced_query: str
    metadata_filters: Dict
    retrieved_documents: List[Dict]
    related_documents: List[Dict]
    prerequisites: List[Dict]
    context: str
    answer: str
    sources: List[Dict]
    memory: List[Dict]

class RAGAgent:
    def __init__(self):
        # Initialize LLM
        self.llm = ChatGroq(
            groq_api_key=Config.GROQ_API_KEY,
            model_name=Config.GROQ_MODEL,
            temperature=0.1
        )
        
        # Initialize vector store manager
        self.vector_manager = VectorStoreManager()
        
        # Initialize memory
        self.memory = ConversationBufferWindowMemory(
            k=10,  # Keep last 10 exchanges
            memory_key=Config.MEMORY_KEY,
            return_messages=True
        )
        
        # Initialize prompts
        self._setup_prompts()
        
        # Build the graph
        self.graph = self._build_graph()

    def _setup_prompts(self):
        """Setup all the prompts used by the agent"""
        
        self.query_enhancement_prompt = ChatPromptTemplate.from_template(
        """
        You are an expert at refining and expanding user queries for technical documentation search in domains such as software architecture, cloud infrastructure, APIs, frameworks, and related engineering concepts.

        Inputs:

        Original Query: {query}
        Chat History: {chat_history}

        Your task is to enhance the original query to maximize retrieval of relevant, high-quality documentation by:
            - Adding precise technical terms and common synonyms used in professional documentation.
            - Leveraging chat history to include relevant context, requirements, or constraints mentioned earlier.
            - Expanding abbreviations and acronyms into their full forms, while keeping the short forms if widely used.
            - Incorporating closely related concepts or technologies that improve coverage without introducing irrelevant scope creep.
            - Maintaining accuracy and specificity — avoid speculative details or assumptions not supported by the provided query or history.

        Output Requirements:
            - Return one single enhanced query.
            - Keep it concise yet complete (clear, information-rich, and under 30 words unless critical context requires more).
            - Do not change the original query’s intent.
            - Don't add anything extra like what you have improved or a summary.

        Few-Shot Examples:

        Example 1
        Original Query:
        Azure VNet peering setup
        Chat History:
        User previously asked about security best practices for connecting multiple regions.
        Enhanced Query:
        Azure Virtual Network (VNet) peering configuration and security best practices for multi-region deployment in Microsoft Azure

        Example 2
        Original Query:
        gcp iam roles
        Chat History:
        Earlier discussion focused on granting least privilege access for Cloud Storage.
        Enhanced Query:
        Google Cloud Platform (GCP) Identity and Access Management (IAM) roles for implementing least privilege access in Cloud Storage

        Example 3
        Original Query:
        k8s ingress setup
        Chat History:
        User previously discussed SSL termination and custom domain configuration.
        Enhanced Query:
        Kubernetes (k8s) Ingress controller setup with SSL/TLS termination and custom domain configuration

        Example 4
        Original Query:
        aws s3 replication
        Chat History:
        No relevant prior context.
        Enhanced Query:
        Amazon Web Services (AWS) S3 cross-region replication setup and configuration

        Final Output Format:
        Enhanced Query:
        """)
        
        self.metadata_extraction_prompt = ChatPromptTemplate.from_template("""
        Extract metadata filters from the user's query to help narrow down document search.
        
        Query: {query}
        
        Extract any mentions of:
        - topic: subject area or domain(If there are multiple topics, concatinate them and return as single string not an array)
        - author: specific author names(If there are multiple authors, concatinate them and return as single string not an array)
        - department: organizational department(If there are multiple departments, concatinate them and return as single string not an array)
        - section: specific section names
        - prerequisites: mentioned prerequisites or dependencies
        
        Return a JSON object with these fields (use null if not found):
        {{
            "topic": "extracted topic",
            "author": "extracted author", 
            "department": "extracted department",
            "section": "extracted section",
            "prerequisites": ["list", "of", "prerequisites"]
        }}
        
        JSON Response:
        """)
        
        self.answer_generation_prompt = ChatPromptTemplate.from_template("""
        You are a helpful technical documentation assistant. Answer the user's question using the provided context.
        
        User Question: {query}
        Chat History: {chat_history}
        
        Retrieved Context:
        {context}
        
        Related Documents:
        {related_docs}
        
        Prerequisites:
        {prerequisites}
        
        Instructions:
        1. Answer the question directly and comprehensively
        2. Use information from the context provided
        3. If prerequisites exist, mention them early in your response
        4. Include relevant details from related documents if they add value
        5. Be specific and cite your sources when possible
        6. If the context doesn't contain enough information, say so clearly
        7. Consider the chat history for context continuity
        
        Answer:
        """)

    def _build_graph(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("enhance_query", self.enhance_query)
        workflow.add_node("extract_metadata", self.extract_metadata_filters)
        workflow.add_node("retrieve_documents", self.retrieve_documents)
        workflow.add_node("get_related_docs", self.get_related_documents)
        workflow.add_node("get_prerequisites", self.get_prerequisites)
        workflow.add_node("generate_answer", self.generate_answer)
        
        # Define the flow
        workflow.set_entry_point("enhance_query")
        
        workflow.add_edge("enhance_query", "extract_metadata")
        workflow.add_edge("extract_metadata", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "get_related_docs")
        workflow.add_edge("get_related_docs", "get_prerequisites")
        workflow.add_edge("get_prerequisites", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        return workflow.compile()

    def enhance_query(self, state: AgentState) -> AgentState:
        """Enhance the user query using chat history context"""
        try:
            # Get chat history
            chat_history = self._format_chat_history(state.get("memory", []))
            
            # Enhance query
            response = self.llm.invoke(
                self.query_enhancement_prompt.format(
                    query=state["query"],
                    chat_history=chat_history
                )
            )
            
            state["enhanced_query"] = response.content.strip()
            print(f"Enhanced query: {state['enhanced_query']}")
            
        except Exception as e:
            print(f"Error enhancing query: {str(e)}")
            state["enhanced_query"] = state["query"]  # Fallback to original
        
        return state

    def extract_metadata_filters(self, state: AgentState) -> AgentState:
        """Extract metadata filters from both original and enhanced queries"""
        try:
            # Combine original and enhanced queries
            combined_query = f"{state['query']} {state['enhanced_query']}"
            
            response = self.llm.invoke(
                self.metadata_extraction_prompt.format(query=combined_query)
            )
            
            # Parse JSON response
            try:
                filters = json.loads(response.content.strip())
                # Clean up None values and empty lists
                state["metadata_filters"] = {k: v for k, v in filters.items() if v}
                print(f"Extracted filters: {state['metadata_filters']}")
            except json.JSONDecodeError:
                state["metadata_filters"] = {}
                
        except Exception as e:
            print(f"Error extracting metadata: {str(e)}")
            state["metadata_filters"] = {}
        
        return state

    def retrieve_documents(self, state: AgentState) -> AgentState:
        """Retrieve relevant documents using both queries and filters"""
        try:
            # Search with original query
            original_results = self.vector_manager.search_similar(
                state["enhanced_query"], 
                state["metadata_filters"], 
                k=4
            )
            
            # Search with enhanced query
            enhanced_results = self.vector_manager.search_similar(
                state["enhanced_query"], 
                k=4
            )
            
            # Combine and deduplicate results
            all_results = []
            seen_ids = set()
            
            for results in [original_results, enhanced_results]:
                for result in results:
                    result_id = result.get('metadata', {}).get('section_id', '')
                    if result_id and result_id not in seen_ids:
                        all_results.append(result)
                        seen_ids.add(result_id)
            
            # Sort by similarity score and take top results
            all_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            state["retrieved_documents"] = all_results[:Config.TOP_K_DOCUMENTS]
            
            print(f"Retrieved {len(state['retrieved_documents'])} documents")
            
        except Exception as e:
            print(f"Error retrieving documents: {str(e)}")
            state["retrieved_documents"] = []
        
        return state

    def get_related_documents(self, state: AgentState) -> AgentState:
        """Get related documents from knowledge graph"""
        try:
            related_docs = []
            
            # Get related documents for each retrieved document
            for doc in state["retrieved_documents"]:
                url = doc.get('metadata', {}).get('url', '')
                if url:
                    related = self.vector_manager.get_related_documents(url)
                    related_docs.extend(related)
            
            # Remove duplicates
            unique_related = []
            seen_ids = set()
            for doc in related_docs:
                doc_id = doc.get('id', '')
                if doc_id and doc_id not in seen_ids:
                    unique_related.append(doc)
                    seen_ids.add(doc_id)
            
            state["related_documents"] = unique_related[:5]  # Limit to 5
            print(f"Found {len(state['related_documents'])} related documents")
            
        except Exception as e:
            print(f"Error getting related documents: {str(e)}")
            state["related_documents"] = []
        
        return state

    def get_prerequisites(self, state: AgentState) -> AgentState:
        """Get prerequisite information"""
        try:
            prerequisites = []
            
            # Get prerequisites for each retrieved document
            for doc in state["retrieved_documents"]:
                url = doc.get('metadata', {}).get('url', '')
                if url:
                    prereqs = self.vector_manager.get_prerequisites_chain(url)
                    prerequisites.extend(prereqs)
            
            # Remove duplicates
            unique_prereqs = []
            seen_ids = set()
            for prereq in prerequisites:
                prereq_id = prereq.get('id', '')
                if prereq_id and prereq_id not in seen_ids:
                    unique_prereqs.append(prereq)
                    seen_ids.add(prereq_id)
            
            state["prerequisites"] = unique_prereqs
            print(f"Found {len(state['prerequisites'])} prerequisites")
            
        except Exception as e:
            print(f"Error getting prerequisites: {str(e)}")
            state["prerequisites"] = []
        
        return state

    def generate_answer(self, state: AgentState) -> AgentState:
        """Generate the final answer"""
        try:
            # Prepare context from retrieved documents
            context_parts = []
            sources = []
            
            for i, doc in enumerate(state["retrieved_documents"]):
                content = doc.get('content', '')
                metadata = doc.get('metadata', {})
                
                context_parts.append(f"Document {i+1}:\n{content}")
                
                # Prepare source information
                source = {
                    'title': metadata.get('title', 'Unknown'),
                    'url': metadata.get('url', ''),
                    'section': metadata.get('section', ''),
                    'content': content,
                    'similarity_score': doc.get('similarity_score', 0)
                }
                sources.append(source)
            
            context = "\n\n".join(context_parts)
            
            # Format related documents
            related_docs_text = ""
            if state["related_documents"]:
                related_parts = []
                for doc in state["related_documents"]:
                    title = doc.get('title', 'Unknown')
                    relationship = doc.get('relationship', 'related')
                    related_parts.append(f"- {title} ({relationship})")
                related_docs_text = "\n".join(related_parts)
            
            # Format prerequisites
            prerequisites_text = ""
            if state["prerequisites"]:
                prereq_parts = []
                for prereq in state["prerequisites"]:
                    title = prereq.get('title', 'Unknown')
                    prereq_parts.append(f"- {title}")
                prerequisites_text = "\n".join(prereq_parts)
            
            # Get chat history
            chat_history = self._format_chat_history(state.get("memory", []))
            print(f"This is an enhanced query: {state['enhanced_query']}")
            # Generate answer
            response = self.llm.invoke(
                self.answer_generation_prompt.format(
                    query=state["enhanced_query"],
                    chat_history=chat_history,
                    context=context,
                    related_docs=related_docs_text,
                    prerequisites=prerequisites_text
                )
            )
            
            state["answer"] = response.content
            state["context"] = context
            state["sources"] = sources
            
            print("Generated answer successfully")
            
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            state["answer"] = "I apologize, but I encountered an error while generating the answer."
            state["sources"] = []
        
        return state

    def _format_chat_history(self, memory: List[Dict]) -> str:
        """Format chat history for prompt"""
        if not memory:
            return "No previous conversation."
        
        formatted = []
        for item in memory[-6:]:  # Last 6 exchanges
            if item.get('type') == 'human':
                formatted.append(f"Human: {item.get('content', '')}")
            elif item.get('type') == 'ai':
                formatted.append(f"Assistant: {item.get('content', '')}")
        
        return "\n".join(formatted)

    def process_query(self, query: str, session_memory: List[Dict] = None) -> Dict:
        """Process a user query and return the response"""
        # Initialize state
        initial_state = {
            "messages": [],
            "query": query,
            "enhanced_query": "",
            "metadata_filters": {},
            "retrieved_documents": [],
            "related_documents": [],
            "prerequisites": [],
            "context": "",
            "answer": "",
            "sources": [],
            "memory": session_memory or []
        }
        
        try:
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            
            # Update memory with new interaction
            if session_memory is not None:
                session_memory.append({"type": "human", "content": query})
                session_memory.append({"type": "ai", "content": final_state["answer"]})
                
                # Keep memory manageable
                if len(session_memory) > 20:
                    session_memory = session_memory[-20:]
            
            return {
                "answer": final_state["answer"],
                "sources": final_state["sources"],
                "related_documents": final_state["related_documents"],
                "prerequisites": final_state["prerequisites"],
                "enhanced_query": final_state["enhanced_query"],
                "metadata_filters": final_state["metadata_filters"],
                "memory": session_memory
            }
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return {
                "answer": "I apologize, but I encountered an error while processing your query.",
                "sources": [],
                "related_documents": [],
                "prerequisites": [],
                "enhanced_query": query,
                "metadata_filters": {},
                "memory": session_memory
            }

    def get_stats(self) -> Dict:
        """Get system statistics"""
        return self.vector_manager.get_collection_stats()