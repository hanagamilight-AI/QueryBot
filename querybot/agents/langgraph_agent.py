"""
LangGraph-based Agent Architecture for QueryBot
Implements a multi-agent system with specialized roles for political intelligence analysis
"""
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from datetime import datetime
from loguru import logger
import operator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from tools.mcp_tools import MCP_TOOLS, get_tool, ToolResult
from config.settings import settings
from cache.hybrid_cache import get_cache, CacheNamespace
from embeddings.embedder import get_embedder


# ============================================================================
# STATE DEFINITIONS
# ============================================================================

class AgentState(TypedDict):
    """
    Complete state representation for the agent graph
    Tracks all information flowing through the graph
    """
    # Input/Output
    query: str
    session_id: str
    response: str
    
    # Context and memory
    conversation_history: List[Dict[str, str]]
    retrieved_documents: List[Dict[str, Any]]
    
    # Query analysis
    query_intent: str  # classification of query type
    query_entities: Dict[str, str]  # extracted entities (constituency, party, etc.)
    query_timeframe: Optional[str]
    
    # Tool execution
    tool_calls: List[Dict[str, Any]]
    tool_results: List[ToolResult]
    
    # Reasoning trace
    reasoning_steps: List[str]
    confidence_score: float
    
    # Metadata
    created_at: datetime
    updated_at: datetime
    iteration_count: int


# ============================================================================
# NODE IMPLEMENTATIONS
# ============================================================================

def query_analysis_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyze user query to extract intent, entities, and required actions
    
    This node performs:
    - Intent classification (factual, analytical, comparative, trend)
    - Entity extraction (constituency, party, candidate, date ranges)
    - Query complexity assessment
    """
    logger.info(f"Analyzing query: {state['query']}")
    
    query = state['query'].lower()
    
    # Simple intent classification (in production, use LLM)
    intent = "factual"
    if any(word in query for word in ['compare', 'versus', 'vs', 'difference']):
        intent = "comparative"
    elif any(word in query for word in ['trend', 'change', 'evolution', 'history']):
        intent = "trend"
    elif any(word in query for word in ['analyze', 'why', 'reason', 'impact']):
        intent = "analytical"
    elif any(word in query for word in ['predict', 'forecast', 'will']):
        intent = "predictive"
    
    # Extract entities (simplified - in production use NER model)
    entities = {}
    
    # Look for party names
    parties = ['bjp', 'congress', 'aap', 'dmk', 'aitc', 'bsp', 'ssp', 'ncp']
    for party in parties:
        if party in query:
            entities['party'] = party.upper()
            break
    
    # Look for common constituency patterns (simplified)
    if 'constituency' in query or 'ward' in query:
        # Extract constituency name (simplified)
        words = query.split()
        for i, word in enumerate(words):
            if word in ['constituency', 'ward'] and i + 1 < len(words):
                entities['constituency'] = words[i + 1].capitalize()
                break
    
    # Look for state names (sample)
    states = ['delhi', 'maharashtra', 'tamil nadu', 'uttar pradesh', 'gujarat']
    for state_name in states:
        if state_name in query:
            entities['state'] = state_name.title()
            break
    
    # Determine timeframe
    timeframe = None
    if 'recent' in query or 'latest' in query:
        timeframe = "last_3_months"
    elif 'last year' in query or 'past year' in query:
        timeframe = "last_year"
    elif '2024' in query:
        timeframe = "year_2024"
    
    reasoning = [
        f"Query classified as: {intent}",
        f"Entities extracted: {entities}",
        f"Timeframe: {timeframe}"
    ]
    
    return {
        "query_intent": intent,
        "query_entities": entities,
        "query_timeframe": timeframe,
        "reasoning_steps": reasoning,
        "updated_at": datetime.utcnow()
    }


def retrieval_node(state: AgentState) -> Dict[str, Any]:
    """
    Retrieve relevant documents using vector search and database queries
    
    Based on query analysis, this node:
    - Performs semantic search via pgvector
    - Executes structured database queries
    - Aggregates results from multiple sources
    - Checks cache first for faster responses
    """
    logger.info("Executing retrieval strategies")
    
    # Check cache first if enabled
    if settings.CACHE_ENABLED:
        try:
            cache = get_cache(similarity_threshold=settings.CACHE_SIMILARITY_THRESHOLD)
            
            # Generate embedding for semantic cache lookup
            embedder = get_embedder()
            query_vector = embedder.embed_query(state['query'])
            
            # Try to get cached response
            cache_hit = cache.get(
                query=state['query'],
                parameters=state.get('query_entities', {}),
                namespace=CacheNamespace.QUERY_RESPONSE,
                query_vector=query_vector
            )
            
            if cache_hit:
                logger.info(f"Cache HIT ({cache_hit.hit_type}) - similarity: {cache_hit.similarity_score}")
                
                # Return cached response directly
                return {
                    "retrieved_documents": cache_hit.entry.value.get('documents', []),
                    "tool_results": [],
                    "reasoning_steps": [f"Response retrieved from cache ({cache_hit.hit_type} match)",
                                       f"Cache retrieval time: {cache_hit.retrieval_time_ms:.2f}ms"],
                    "from_cache": True,
                    "cached_response": cache_hit.entry.value.get('response'),
                    "updated_at": datetime.utcnow()
                }
        except Exception as e:
            logger.warning(f"Cache lookup failed, proceeding with normal retrieval: {e}")
    
    retrieved_docs = []
    tool_results = []
    reasoning = []
    
    # Get tools
    vector_tool = get_tool("vector_search")
    db_tool = get_tool("database_query")
    
    # Strategy 1: Vector similarity search
    filters = state.get('query_entities', {})
    vector_result = vector_tool.execute(
        query=state['query'],
        limit=10,
        filters=filters if filters else None
    )
    
    if vector_result.success:
        retrieved_docs.extend(vector_result.data)
        tool_results.append(vector_result)
        reasoning.append(f"Vector search returned {len(vector_result.data)} documents")
    else:
        reasoning.append(f"Vector search failed: {vector_result.error}")
    
    # Strategy 2: Structured database query based on intent
    if state['query_intent'] in ['factual', 'comparative']:
        query_type = "general"
        if 'election' in state['query'].lower():
            query_type = "elections"
        elif 'survey' in state['query'].lower() or 'sentiment' in state['query'].lower():
            query_type = "surveys"
        elif 'manifesto' in state['query'].lower() or 'promise' in state['query'].lower():
            query_type = "manifestos"
        
        db_result = db_tool.execute(
            query_type=query_type,
            filters=filters if filters else None,
            limit=10
        )
        
        if db_result.success:
            tool_results.append(db_result)
            reasoning.append(f"Database query returned {len(db_result.data)} records")
        else:
            reasoning.append(f"Database query failed: {db_result.error}")
    
    # Strategy 3: External API for recent events
    if state['query_timeframe'] == "last_3_months" or 'recent' in state['query'].lower():
        api_tool = get_tool("external_api")
        api_result = api_tool.execute(
            api_type="news",
            query=state['query'],
            location=filters.get('state') if filters else None
        )
        
        if api_result.success:
            tool_results.append(api_result)
            reasoning.append(f"External API returned {len(api_result.data)} articles")
    
    # Deduplicate and rank results
    unique_docs = []
    seen_ids = set()
    for doc in retrieved_docs:
        doc_id = doc.get('id')
        if doc_id and doc_id not in seen_ids:
            unique_docs.append(doc)
            seen_ids.add(doc_id)
    
    return {
        "retrieved_documents": unique_docs[:15],  # Limit context
        "tool_results": tool_results,
        "reasoning_steps": reasoning,
        "from_cache": False,
        "updated_at": datetime.utcnow()
    }


def synthesis_node(state: AgentState) -> Dict[str, Any]:
    """
    Synthesize retrieved information into coherent response
    
    This node:
    - Analyzes retrieved documents
    - Identifies key insights and patterns
    - Generates natural language response
    - Cites sources appropriately
    - Caches the response for future use
    """
    logger.info("Synthesizing response from retrieved information")
    
    # Check if we have a cached response from retrieval
    if state.get('from_cache') and state.get('cached_response'):
        logger.info("Using cached response from retrieval node")
        return {
            "response": state['cached_response'],
            "confidence_score": 0.95,  # High confidence for cached responses
            "reasoning_steps": state.get('reasoning_steps', []) + ["Response served from cache"],
            "from_cache": True,
            "updated_at": datetime.utcnow()
        }
    
    docs = state.get('retrieved_documents', [])
    query = state['query']
    intent = state.get('query_intent', 'factual')
    
    reasoning = []
    response = ""
    confidence = 0.5
    
    if not docs:
        response = "I couldn't find relevant information in the database to answer your query. Please try rephrasing or ask about specific constituencies, parties, or election data."
        confidence = 0.2
        reasoning.append("No relevant documents retrieved")
    else:
        # Generate response based on intent
        if intent == 'factual':
            response, confidence = _generate_factual_response(query, docs)
        elif intent == 'comparative':
            response, confidence = _generate_comparative_response(query, docs)
        elif intent == 'trend':
            response, confidence = _generate_trend_response(query, docs)
        elif intent == 'analytical':
            response, confidence = _generate_analytical_response(query, docs)
        else:
            response, confidence = _generate_factual_response(query, docs)
        
        reasoning.append(f"Synthesized response from {len(docs)} documents")
        reasoning.append(f"Confidence score: {confidence}")
    
    # Add source citations
    if docs and len(docs) > 0:
        sources = _extract_sources(docs)
        response += "\n\n**Sources:**\n" + "\n".join(sources[:5])
    
    result = {
        "response": response,
        "confidence_score": confidence,
        "reasoning_steps": reasoning,
        "from_cache": False,
        "updated_at": datetime.utcnow()
    }
    
    # Cache the response if not from cache and cache is enabled
    if not state.get('from_cache') and settings.CACHE_ENABLED:
        try:
            cache = get_cache(similarity_threshold=settings.CACHE_SIMILARITY_THRESHOLD)
            embedder = get_embedder()
            query_vector = embedder.embed_query(query)
            
            # Store response with documents
            cache_value = {
                "response": response,
                "documents": docs,
                "confidence": confidence,
                "intent": intent
            }
            
            cache.set(
                query=query,
                value=cache_value,
                parameters=state.get('query_entities', {}),
                namespace=CacheNamespace.QUERY_RESPONSE,
                ttl=settings.CACHE_TTL_DEFAULT,
                query_vector=query_vector
            )
            reasoning.append("Response cached for future queries")
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")
    
    return result


def _generate_factual_response(query: str, docs: List[Dict]) -> tuple:
    """Generate factual response"""
    # In production, use LLM here
    content_snippets = [doc.get('content', '')[:200] for doc in docs[:5]]
    
    response = "Based on the available data:\n\n"
    for i, snippet in enumerate(content_snippets, 1):
        response += f"{i}. {snippet}...\n\n"
    
    confidence = min(0.9, 0.5 + len(docs) * 0.05)
    return response, confidence


def _generate_comparative_response(query: str, docs: List[Dict]) -> tuple:
    """Generate comparative response"""
    response = "Here's a comparison based on the data:\n\n"
    
    # Group by party or candidate
    by_party = {}
    for doc in docs:
        party = doc.get('party', 'Unknown')
        if party not in by_party:
            by_party[party] = []
        by_party[party].append(doc)
    
    for party, party_docs in by_party.items():
        response += f"**{party}**:\n"
        for doc in party_docs[:2]:
            response += f"- {doc.get('content', '')[:150]}...\n"
        response += "\n"
    
    confidence = min(0.85, 0.5 + len(docs) * 0.05)
    return response, confidence


def _generate_trend_response(query: str, docs: List[Dict]) -> tuple:
    """Generate trend analysis response"""
    response = "Trend analysis based on available data:\n\n"
    
    # Sort by date if available
    dated_docs = [d for d in docs if d.get('document_date')]
    if dated_docs:
        dated_docs.sort(key=lambda x: x['document_date'], reverse=True)
        
        response += "**Chronological Overview:**\n"
        for doc in dated_docs[:5]:
            date_str = doc['document_date'][:10] if doc['document_date'] else 'Unknown'
            response += f"- {date_str}: {doc.get('content', '')[:150]}...\n"
    else:
        response += "Limited temporal data available.\n"
        for doc in docs[:3]:
            response += f"- {doc.get('content', '')[:150]}...\n"
    
    confidence = min(0.8, 0.5 + len(docs) * 0.05)
    return response, confidence


def _generate_analytical_response(query: str, docs: List[Dict]) -> tuple:
    """Generate analytical response"""
    response = "Analysis based on the data:\n\n"
    
    # Extract key themes
    response += "**Key Findings:**\n"
    for doc in docs[:5]:
        content = doc.get('content', '')
        if doc.get('sentiment_score'):
            sentiment = "positive" if doc['sentiment_score'] > 0.5 else "negative" if doc['sentiment_score'] < 0.3 else "neutral"
            response += f"- [{sentiment}] {content[:150]}...\n"
        else:
            response += f"- {content[:150]}...\n"
    
    confidence = min(0.85, 0.5 + len(docs) * 0.05)
    return response, confidence


def _extract_sources(docs: List[Dict]) -> List[str]:
    """Extract source citations from documents"""
    sources = []
    for doc in docs[:5]:
        source_type = doc.get('source_type', 'unknown')
        constituency = doc.get('constituency', '')
        date = doc.get('document_date', '')[:10] if doc.get('document_date') else ''
        
        source_str = f"[{source_type.title()}"
        if constituency:
            source_str += f" - {constituency}"
        if date:
            source_str += f" - {date}"
        source_str += "]"
        
        sources.append(source_str)
    
    return sources


def memory_node(state: AgentState) -> Dict[str, Any]:
    """
    Update conversation memory with current interaction
    
    Stores:
    - User query
    - Assistant response
    - Context for future turns
    """
    memory_tool = get_tool("conversation_memory")
    
    # Store user query
    memory_tool.execute(
        action="store",
        session_id=state['session_id'],
        role="user",
        message=state['query']
    )
    
    # Store assistant response
    memory_tool.execute(
        action="store",
        session_id=state['session_id'],
        role="assistant",
        message=state['response']
    )
    
    # Retrieve conversation history
    history_result = memory_tool.execute(
        action="retrieve",
        session_id=state['session_id'],
        limit=10
    )
    
    return {
        "conversation_history": history_result.data if history_result.success else [],
        "updated_at": datetime.utcnow()
    }


def validation_node(state: AgentState) -> Dict[str, Any]:
    """
    Validate response quality and completeness
    
    Checks:
    - Response relevance to query
    - Source attribution
    - Confidence threshold
    """
    reasoning = state.get('reasoning_steps', [])
    confidence = state.get('confidence_score', 0)
    
    # Validation checks
    if confidence < 0.3:
        reasoning.append("Low confidence - adding disclaimer")
        state['response'] = "**Note:** The following information has low confidence. Please verify with official sources.\n\n" + state['response']
    
    if not state.get('retrieved_documents'):
        reasoning.append("No sources - flagging response")
        state['response'] += "\n\n*No supporting documents found in database.*"
    
    reasoning.append("Validation complete")
    
    return {
        "reasoning_steps": reasoning,
        "updated_at": datetime.utcnow()
    }


# ============================================================================
# CONDITIONAL EDGES
# ============================================================================

def should_continue(state: AgentState) -> str:
    """
    Determine if additional iterations are needed
    """
    iteration_count = state.get('iteration_count', 0)
    confidence = state.get('confidence_score', 0)
    
    # Max iterations reached
    if iteration_count >= settings.GRAPH_MAX_ITERATIONS:
        return "end"
    
    # Low confidence - might need more retrieval
    if confidence < 0.4 and not state.get('retrieved_documents'):
        return "retry_retrieval"
    
    return "continue"


def route_by_intent(state: AgentState) -> str:
    """
    Route to different processing paths based on query intent
    """
    intent = state.get('query_intent', 'factual')
    
    if intent in ['factual', 'comparative']:
        return "standard"
    elif intent == 'trend':
        return "temporal"
    elif intent == 'analytical':
        return "deep_analysis"
    else:
        return "standard"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def build_agent_graph():
    """
    Construct the LangGraph workflow for QueryBot
    
    Architecture:
    1. Query Analysis → 2. Retrieval → 3. Synthesis → 4. Validation → 5. Memory
    
    With conditional routing based on intent and confidence
    """
    
    # Initialize the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("query_analysis", query_analysis_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("validation", validation_node)
    workflow.add_node("memory", memory_node)
    
    # Set entry point
    workflow.set_entry_point("query_analysis")
    
    # Define edges
    workflow.add_edge("query_analysis", "retrieval")
    workflow.add_edge("retrieval", "synthesis")
    workflow.add_edge("synthesis", "validation")
    workflow.add_edge("validation", "memory")
    workflow.add_edge("memory", END)
    
    # Compile with memory saver
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    logger.info("Agent graph compiled successfully")
    return app


# ============================================================================
# AGENT INTERFACE
# ============================================================================

class QueryBotAgent:
    """
    Main agent interface for QueryBot
    Wraps the LangGraph with convenient methods
    """
    
    def __init__(self):
        self.graph = build_agent_graph()
        logger.info("QueryBot Agent initialized")
    
    def query(
        self, 
        question: str, 
        session_id: str = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Process a user query through the agent
        
        Args:
            question: User's question
            session_id: Session identifier for memory
            stream: Whether to stream responses
            
        Returns:
            Dictionary with response and metadata
        """
        if not session_id:
            session_id = f"session_{datetime.utcnow().timestamp()}"
        
        # Prepare initial state
        initial_state = {
            "query": question,
            "session_id": session_id,
            "response": "",
            "conversation_history": [],
            "retrieved_documents": [],
            "query_intent": "",
            "query_entities": {},
            "query_timeframe": None,
            "tool_calls": [],
            "tool_results": [],
            "reasoning_steps": [],
            "confidence_score": 0.0,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "iteration_count": 0
        }
        
        # Configure checkpoint for session
        config = {"configurable": {"thread_id": session_id}}
        
        # Run the graph
        result = self.graph.invoke(initial_state, config=config)
        
        logger.info(f"Query processed. Confidence: {result.get('confidence_score', 0):.2f}")
        
        return {
            "response": result.get('response', ''),
            "confidence": result.get('confidence_score', 0),
            "sources": len(result.get('retrieved_documents', [])),
            "reasoning": result.get('reasoning_steps', []),
            "session_id": session_id
        }
    
    def chat(
        self,
        message: str,
        session_id: str
    ) -> str:
        """
        Chat interface maintaining conversation context
        
        Args:
            message: User message
            session_id: Session identifier
            
        Returns:
            Agent response string
        """
        result = self.query(message, session_id=session_id)
        return result['response']
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """Retrieve conversation history for a session"""
        memory_tool = get_tool("conversation_memory")
        result = memory_tool.execute(
            action="retrieve",
            session_id=session_id,
            limit=20
        )
        return result.data if result.success else []
    
    def clear_session(self, session_id: str) -> bool:
        """Clear conversation history for a session"""
        memory_tool = get_tool("conversation_memory")
        result = memory_tool.execute(
            action="clear",
            session_id=session_id
        )
        return result.success


# Singleton instance
_agent_instance: Optional[QueryBotAgent] = None


def get_agent() -> QueryBotAgent:
    """Get or create the agent singleton"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = QueryBotAgent()
    return _agent_instance
