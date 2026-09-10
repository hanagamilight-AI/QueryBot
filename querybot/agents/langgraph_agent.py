"""
LangGraph-based Agent Architecture for QueryBot
Implements a multi-agent system with LLM-driven reasoning for political intelligence analysis
"""
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from datetime import datetime
from loguru import logger
import operator
import json
import re

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from tools.mcp_tools import MCP_TOOLS, get_tool, ToolResult
from config.settings import settings
from cache.hybrid_cache import get_cache, CacheNamespace
from embeddings.embedder import get_embedding_service, EmbeddingService
from models.adapters import get_model_adapter, ModelAdapter
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


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
    resolved_query: str  # Query with pronouns resolved
    
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
# LLM AGENT PROMPTS
# ============================================================================

QUERY_ANALYSIS_PROMPT = """You are the Query Analysis Agent for a Political Intelligence System.
Your goal is to analyze the user's query in the context of conversation history.

TASKS:
1. Resolve any pronouns (e.g., "they", "it", "that party", "the former") using the conversation history.
2. Extract key entities: Constituency, Party, Candidate, Date, Topic.
3. Detect the language of the query.
4. Identify if the query is ambiguous.

Return ONLY a valid JSON object with this exact structure:
{
    "resolved_query": "The query with pronouns replaced by specific entities from history",
    "entities": {"entity_name": "entity_type"},
    "language": "en",
    "ambiguity_score": 0.1
}

Conversation History:
{history}

Current Query: {query}
"""

INTENT_PLANNING_PROMPT = """You are the Intent Planning Agent for a Political Intelligence System.
Based on the resolved query and entities, determine the user's intent and create an execution plan.

INTENT CATEGORIES:
- FACTUAL: Direct data lookup (e.g., "What was the voter turnout in Patna?")
- COMPARATIVE: Compare two or more entities (e.g., "Compare BJP vs Congress performance")
- TREND: Time-series analysis (e.g., "How has voting pattern changed over 10 years?")
- ANALYTICAL: Complex reasoning requiring multiple steps (e.g., "What impact did policy X have on demographic Y?")
- CONVERSATIONAL: General chat, greetings, or non-data queries

Return ONLY a valid JSON object:
{
    "intent": "FACTUAL",
    "confidence": 0.95,
    "sub_queries": ["decomposed sub-query 1", "sub-query 2"],
    "required_tools": ["vector_search", "sql_query"],
    "parallel_groups": [[0, 1]],
    "execution_strategy": "sequential|parallel"
}

Resolved Query: {resolved_query}
Entities: {entities}
"""

RETRIEVAL_PROMPT = """You are the Retrieval Agent for a Political Intelligence System.
Your job is to formulate optimal search queries for the vector database and SQL filters.

Given the sub-queries and entities, generate specific search strings and metadata filters.
Return ONLY a valid JSON object:
{
    "search_steps": [
        {"query": "vector search string", "filters": {"constituency": "...", "date_gte": "..."}, "tool": "vector_search"}
    ]
}

Sub-queries: {sub_queries}
Entities: {entities}
"""

SYNTHESIS_PROMPT = """You are the Synthesis Agent for a Political Intelligence System.
Synthesize the retrieved documents into a coherent, accurate, and neutral political intelligence report.

GUIDELINES:
- Cite sources explicitly (e.g., [Source: Election Commission 2020])
- Do not hallucinate facts. If data is missing, state it clearly.
- Maintain a professional, objective, and analytical tone.
- Structure the response logically with clear sections.

Retrieved Documents:
{documents}

Original Query: {query}
Intent: {intent}
"""

VALIDATION_PROMPT = """You are the Validation Agent for a Political Intelligence System.
Verify the synthesized response against the retrieved context.

CHECKS:
1. Check for hallucinations (facts not supported by the retrieved context).
2. Check for bias or non-neutral language.
3. Ensure all claims are properly cited.
4. Verify the response directly answers the original query.

Return ONLY a valid JSON object:
{
    "is_valid": true,
    "issues": [],
    "confidence_score": 0.92,
    "suggested_corrections": ""
}

Original Query: {query}
Retrieved Context: {context}
Synthesized Response: {response}
"""


# ============================================================================
# NODE IMPLEMENTATIONS
# ============================================================================

def _call_llm(prompt_template: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper function to call LLM with prompt and parse JSON response.
    Uses the configured model adapter from settings.
    """
    try:
        # Get model adapter
        adapter = get_model_adapter(provider=settings.LLM_PROVIDER)
        
        # Format prompt with variables
        prompt = prompt_template.format(**variables)
        
        # Create messages for LLM
        messages = [
            SystemMessage(content="You are a helpful assistant for a Political Intelligence System. Always respond with valid JSON only, no additional text."),
            HumanMessage(content=prompt)
        ]
        
        # Call LLM
        response = adapter.generate(messages)
        
        # Parse JSON response
        response_text = response.content.strip()
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        result = json.loads(response_text)
        return result
        
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return {}


def query_analysis_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyze user query using LLM to extract entities and resolve pronouns.
    
    This node performs:
    - Pronoun resolution using conversation history (via LLM)
    - Entity extraction (constituency, party, candidate, date ranges)
    - Language detection
    - Ambiguity detection
    """
    logger.info(f"Analyzing query with LLM: {state['query']}")
    
    # Format conversation history for prompt
    history_text = ""
    conv_history = state.get('conversation_history', [])
    if conv_history:
        for turn in conv_history[-5:]:  # Last 5 turns
            role = turn.get('role', 'user')
            message = turn.get('message', '')
            history_text += f"{role.capitalize()}: {message}\n"
    
    # Call LLM for query analysis
    llm_result = _call_llm(QUERY_ANALYSIS_PROMPT, {
        "history": history_text,
        "query": state['query']
    })
    
    resolved_query = llm_result.get('resolved_query', state['query'])
    entities = llm_result.get('entities', {})
    language = llm_result.get('language', 'en')
    ambiguity_score = llm_result.get('ambiguity_score', 0.5)
    
    reasoning = [
        f"LLM resolved query: {resolved_query}",
        f"Entities extracted: {entities}",
        f"Language detected: {language}",
        f"Ambiguity score: {ambiguity_score}"
    ]
    
    return {
        "resolved_query": resolved_query,
        "query_entities": entities,
        "query_language": language,
        "ambiguity_score": ambiguity_score,
        "reasoning_steps": reasoning,
        "updated_at": datetime.utcnow()
    }


def intent_planning_node(state: AgentState) -> Dict[str, Any]:
    """
    Determine query intent and create execution plan using LLM.
    
    This node performs:
    - Intent classification (FACTUAL, COMPARATIVE, TREND, ANALYTICAL, CONVERSATIONAL)
    - Query decomposition into sub-queries
    - Tool selection
    - Parallel execution planning
    """
    logger.info(f"Planning execution with LLM for: {state.get('resolved_query', state['query'])}")
    
    # Call LLM for intent planning
    llm_result = _call_llm(INTENT_PLANNING_PROMPT, {
        "resolved_query": state.get('resolved_query', state['query']),
        "entities": state.get('query_entities', {})
    })
    
    intent = llm_result.get('intent', 'FACTUAL')
    confidence = llm_result.get('confidence', 0.5)
    sub_queries = llm_result.get('sub_queries', [state['query']])
    required_tools = llm_result.get('required_tools', ['vector_search'])
    parallel_groups = llm_result.get('parallel_groups', [[0]])
    execution_strategy = llm_result.get('execution_strategy', 'sequential')
    
    reasoning = [
        f"Intent classified: {intent} (confidence: {confidence})",
        f"Sub-queries: {sub_queries}",
        f"Required tools: {required_tools}",
        f"Execution strategy: {execution_strategy}"
    ]
    
    return {
        "query_intent": intent,
        "intent_confidence": confidence,
        "execution_plan": {
            "sub_queries": sub_queries,
            "required_tools": required_tools,
            "parallel_groups": parallel_groups,
            "execution_strategy": execution_strategy
        },
        "reasoning_steps": reasoning,
        "updated_at": datetime.utcnow()
    }

def query_analysis_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyze user query to extract entities and context ONLY.
    
    This node performs:
    - Entity extraction (constituency, party, candidate, date ranges)
    - Pronoun resolution using conversation history
    - Language detection
    - Ambiguity detection
    
    NOTE: Intent classification is handled by a separate node.
    """
    logger.info(f"Analyzing query for entities: {state['query']}")
    
    query = state['query']
    
    # Extract entities (simplified - in production use NER model via LLM)
    entities = {}
    
    # Look for party names
    parties = ['bjp', 'congress', 'aap', 'dmk', 'aitc', 'bsp', 'ssp', 'ncp', 
               'bharatiya janata party', 'indian national congress']
    query_lower = query.lower()
    for party in parties:
        if party in query_lower:
            entities['party'] = party.upper() if len(party) < 10 else party.title()
            break
    
    # Look for common constituency patterns
    if 'constituency' in query_lower or 'ward' in query_lower:
        words = query.split()
        for i, word in enumerate(words):
            if word.lower() in ['constituency', 'ward'] and i + 1 < len(words):
                entities['constituency'] = words[i + 1].capitalize()
                break
    
    # Look for state names
    states = ['delhi', 'maharashtra', 'tamil nadu', 'uttar pradesh', 'gujarat', 
              'bihar', 'karnataka', 'west bengal', 'rajasthan']
    for state_name in states:
        if state_name in query_lower:
            entities['state'] = state_name.title()
            break
    
    # Detect language (simplified)
    language = "en"
    # Could add more sophisticated language detection here
    
    # Check for ambiguity (missing critical entities)
    is_ambiguous = False
    clarification_question = None
    
    if len(entities) == 0 and not any(word in query_lower for word in ['election', 'vote', 'poll']):
        is_ambiguous = True
        clarification_question = "Could you please specify which constituency, state, or party you are asking about?"
    
    reasoning = [
        f"Entities extracted: {entities}",
        f"Language detected: {language}",
        f"Ambiguity check: {'Needs clarification' if is_ambiguous else 'Clear'}"
    ]
    
    return {
        "query_entities": entities,
        "query_language": language,
        "is_ambiguous": is_ambiguous,
        "clarification_question": clarification_question,
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
    
    NOTE: Cache checking is handled earlier in the pipeline for efficiency.
    This node only executes if there was a cache MISS.
    """
    logger.info("Executing retrieval strategies (Cache Miss)")
    
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
            embedder = get_embedding_service()
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
# INTENT PLANNING NODE (Combined Classification + Planning)
# ============================================================================

def intent_planning_node(state: AgentState) -> Dict[str, Any]:
    """
    Combined Intent Classification and Planning Agent
    
    This node performs:
    1. Intent Classification: Determines query type (factual, comparative, trend, analytical)
    2. Complexity Assessment: Evaluates query difficulty
    3. Query Decomposition: Breaks complex queries into sub-queries
    4. Tool Selection: Identifies required MCP tools
    5. Execution Planning: Defines parallel/sequential execution strategy
    
    Returns:
        Dictionary with intent, complexity, execution plan, and tool requirements
    """
    logger.info(f"Performing intent classification and planning: {state['query']}")
    
    query = state['query']
    entities = state.get('query_entities', {})
    
    # Intent Classification (rule-based for now, can be enhanced with LLM)
    query_lower = query.lower()
    
    # Determine intent based on keywords
    if any(word in query_lower for word in ['compare', 'versus', 'vs', 'difference between']):
        intent = 'comparative'
    elif any(word in query_lower for word in ['trend', 'over time', 'historical', 'change', 'evolution']):
        intent = 'trend'
    elif any(word in query_lower for word in ['analyze', 'analysis', 'why', 'impact', 'effect', 'correlation']):
        intent = 'analytical'
    else:
        intent = 'factual'
    
    # Complexity assessment
    complexity = 'low'
    if intent in ['analytical', 'comparative'] and len(entities) >= 2:
        complexity = 'high'
    elif intent == 'trend' or len(query.split()) > 15:
        complexity = 'medium'
    
    # Query decomposition for complex queries
    sub_queries = []
    parallel_groups = []
    
    if intent == 'comparative' and 'party' in entities:
        # Split comparison into separate entity lookups
        sub_queries.append(f"{entities.get('party', '')} position on query topic")
        parallel_groups.append([0])  # Can run in parallel
    elif intent == 'analytical':
        sub_queries.append("background context")
        sub_queries.append("specific analysis data")
        parallel_groups.append([0, 1])  # Both can run in parallel
    
    # Tool selection based on intent
    required_tools = ['vector_search']  # Always need vector search
    
    if intent in ['factual', 'comparative']:
        required_tools.append('database_query')
    
    if intent == 'trend' or 'recent' in query_lower:
        required_tools.append('external_api')
    
    if complexity == 'high':
        required_tools.append('conversation_memory')  # May need context
    
    reasoning = [
        f"Intent classified as: {intent}",
        f"Complexity level: {complexity}",
        f"Required tools: {required_tools}",
        f"Sub-queries planned: {len(sub_queries)}" if sub_queries else "No decomposition needed"
    ]
    
    return {
        "query_intent": intent,
        "query_complexity": complexity,
        "sub_queries": sub_queries,
        "parallel_groups": parallel_groups,
        "required_tools": required_tools,
        "reasoning_steps": reasoning,
        "updated_at": datetime.utcnow()
    }


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================
# CACHE CHECK NODE (Early Exit Optimization)
# ============================================================================

def cache_check_node(state: AgentState) -> Dict[str, Any]:
    """
    Check parameter-aware hybrid cache for existing response.
    
    This node runs BEFORE intent planning to enable early exit on cache hits,
    avoiding expensive LLM calls and database queries.
    
    Returns:
        Dictionary with cache hit/miss status and cached response if available
    """
    logger.info("Checking parameter-aware hybrid cache")
    
    if not settings.CACHE_ENABLED:
        logger.debug("Cache disabled, proceeding to planning")
        return {
            "cache_hit": False,
            "updated_at": datetime.utcnow()
        }
    
    try:
        cache = get_cache(similarity_threshold=settings.CACHE_SIMILARITY_THRESHOLD)
        embedder = get_embedding_service()
        
        # Generate embedding for semantic cache lookup
        query_vector = embedder.embed_query(state['query'])
        
        # Check cache with parameters (constituency, party, etc.)
        cache_hit_result = cache.get(
            query=state['query'],
            parameters=state.get('query_entities', {}),
            namespace=CacheNamespace.QUERY_RESPONSE,
            query_vector=query_vector
        )
        
        if cache_hit_result:
            logger.info(f"CACHE HIT ({cache_hit_result.hit_type}) - similarity: {cache_hit_result.similarity_score:.3f}")
            
            return {
                "cache_hit": True,
                "cached_response": cache_hit_result.entry.value.get('response'),
                "retrieved_documents": cache_hit_result.entry.value.get('documents', []),
                "confidence_score": cache_hit_result.entry.value.get('confidence', 0.95),
                "cache_hit_type": cache_hit_result.hit_type,
                "cache_similarity": cache_hit_result.similarity_score,
                "reasoning_steps": [
                    f"Response retrieved from cache ({cache_hit_result.hit_type} match)",
                    f"Cache retrieval time: {cache_hit_result.retrieval_time_ms:.2f}ms",
                    f"Similarity score: {cache_hit_result.similarity_score:.3f}"
                ],
                "from_cache": True,
                "updated_at": datetime.utcnow()
            }
        else:
            logger.debug("Cache MISS - proceeding to intent planning")
            return {
                "cache_hit": False,
                "updated_at": datetime.utcnow()
            }
            
    except Exception as e:
        logger.warning(f"Cache check failed, proceeding with normal flow: {e}")
        return {
            "cache_hit": False,
            "cache_error": str(e),
            "updated_at": datetime.utcnow()
        }


def route_from_cache_check(state: AgentState) -> str:
    """Route based on cache hit or miss"""
    if state.get('cache_hit'):
        return "hit"
    return "miss"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def build_agent_graph():
    """
    Construct the LangGraph workflow for QueryBot
    
    Architecture:
    1. Query Analysis → 2. Cache Check → 3. Intent Planning → 4. Retrieval → 5. Synthesis → 6. Validation → 7. Memory
    
    With early-exit optimization: Cache hits bypass all heavy processing.
    """
    
    # Initialize the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("query_analysis", query_analysis_node)
    workflow.add_node("cache_check", cache_check_node)  # Early exit point
    workflow.add_node("intent_planning", intent_planning_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("validation", validation_node)
    workflow.add_node("memory", memory_node)
    
    # Set entry point
    workflow.set_entry_point("query_analysis")
    
    # Define edges
    workflow.add_edge("query_analysis", "cache_check")
    
    # Conditional edge: Cache check → Hit (skip to memory/end) or Miss (continue to planning)
    workflow.add_conditional_edges(
        "cache_check",
        route_from_cache_check,
        {
            "hit": "synthesis",    # Cache hit: skip to synthesis (which will use cached response)
            "miss": "intent_planning"  # Cache miss: proceed with full pipeline
        }
    )
    
    workflow.add_edge("intent_planning", "retrieval")
    workflow.add_edge("retrieval", "synthesis")
    workflow.add_edge("synthesis", "validation")
    workflow.add_edge("validation", "memory")
    workflow.add_edge("memory", END)
    
    # Compile with memory saver
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    logger.info("Agent graph compiled successfully with cache optimization layer")
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
