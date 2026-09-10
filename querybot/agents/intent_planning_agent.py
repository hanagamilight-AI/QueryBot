"""
Intent Planning Agent for QueryBot.
Combined agent that classifies intent AND creates execution plans.
This eliminates the need for separate Intent Classification and Planning agents.
"""
from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal
from datetime import datetime
from loguru import logger
from pydantic import BaseModel, Field
import uuid

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from config.settings import settings
from models.adapters import get_model_adapter, ModelAdapter


# ============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# ============================================================================

class PlanStep(BaseModel):
    """A single step in the execution plan."""
    step_id: int = Field(..., description="Unique ID for the step")
    action: Literal["SEARCH_VECTOR", "QUERY_DB", "CALL_TOOL", "CALCULATE", "COMPARE"] = Field(
        ..., description="The type of action to perform"
    )
    description: str = Field(..., description="Human-readable description of the step")
    tool_name: Optional[str] = Field(None, description="Name of the MCP tool to use (if CALL_TOOL)")
    query_params: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Parameters for the action (e.g., query text, filters, constituency)"
    )
    depends_on: List[int] = Field(
        default_factory=list, 
        description="List of step_ids this step depends on (for ordering)"
    )
    parallel_group: Optional[int] = Field(
        None, 
        description="Group ID for steps that can be executed in parallel"
    )


class ExecutionPlan(BaseModel):
    """The complete execution plan for a query."""
    plan_id: str = Field(default_factory=lambda: f"plan-{uuid.uuid4().hex[:8]}", description="Unique ID for this plan")
    intent: str = Field(..., description="The classified intent driving this plan")
    intent_category: Literal["factual", "comparative", "trend", "analytical", "predictive", "summary"] = Field(
        ..., description="Category of the intent"
    )
    complexity: Literal["low", "medium", "high"] = Field(..., description="Complexity level")
    summary: str = Field(..., description="Brief summary of the strategy")
    steps: List[PlanStep] = Field(..., description="Ordered list of steps to execute")
    required_tools: List[str] = Field(default_factory=list, description="Tools needed for execution")
    execution_mode: Literal["parallel", "sequential", "hybrid"] = Field(
        default="sequential", 
        description="Execution strategy"
    )


class IntentPlanningResult(BaseModel):
    """Combined result from intent classification and planning."""
    intent: str
    intent_category: str
    complexity: str
    confidence: float
    execution_plan: ExecutionPlan
    reasoning: List[str]


# ============================================================================
# AGENT STATE
# ============================================================================

class AgentState(TypedDict):
    """State for the intent planning agent."""
    query: str
    query_entities: Dict[str, Any]
    conversation_history: List[Dict[str, str]]
    intent: Optional[Dict[str, Any]]
    execution_plan: Optional[Dict[str, Any]]
    reasoning_steps: List[str]
    updated_at: datetime


# ============================================================================
# MODEL ADAPTER SETUP
# ============================================================================

def get_planning_model() -> ModelAdapter:
    """Get the model adapter for planning tasks."""
    return get_model_adapter(
        provider=settings.LLM_PROVIDER,
        model_name=settings.LLM_MODEL,
        api_key=settings.OPENROUTER_API_KEY if settings.LLM_PROVIDER == "openrouter" else settings.OPENAI_API_KEY,
        base_url="https://openrouter.ai/api/v1" if settings.LLM_PROVIDER == "openrouter" else None
    )


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT = """
You are an expert Political Intelligence Planner with dual capabilities:
1. **Intent Classification**: Accurately identify what the user is asking
2. **Strategic Planning**: Create executable plans to answer complex queries

### INTENT CATEGORIES:
- **factual**: Direct data lookup (e.g., "What was the turnout in Patna 2020?")
- **comparative**: Comparing entities (e.g., "Compare BJP vs Congress in Bihar")
- **trend**: Time-series analysis (e.g., "Voting trends over 10 years")
- **analytical**: Requires reasoning (e.g., "Why did Party X lose?")
- **predictive**: Forecasting (e.g., "What are the chances of Party Y winning?")
- **summary**: Overview requests (e.g., "Summarize the election results")

### ACTION TYPES:
- **SEARCH_VECTOR**: Semantic search in pgvector (manifestos, speeches, documents)
- **QUERY_DB**: Structured SQL queries (election results, demographics, surveys)
- **CALL_TOOL**: External API calls (news, social media, real-time data)
- **CALCULATE**: Mathematical operations (percentages, growth rates, statistics)
- **COMPARE**: Side-by-side analysis of retrieved data

### PLANNING RULES:
1. **Simple Queries** (factual): Single step, direct retrieval
2. **Comparative Queries**: Parallel retrieval of all entities being compared
3. **Analytical Queries**: Multi-step with verification steps
4. **Trend Queries**: Sequential time-based retrieval
5. Always minimize steps while ensuring completeness
6. Identify which steps can run in PARALLEL (same parallel_group)
7. Specify dependencies clearly (depends_on field)

### OUTPUT FORMAT:
Return ONLY valid JSON matching the ExecutionPlan schema:
{
    "intent": "user's intent in their own words",
    "intent_category": "factual|comparative|trend|analytical|predictive|summary",
    "complexity": "low|medium|high",
    "confidence": 0.0-1.0,
    "execution_plan": {
        "summary": "brief strategy overview",
        "steps": [
            {
                "step_id": 1,
                "action": "SEARCH_VECTOR",
                "description": "...",
                "tool_name": null,
                "query_params": {"query": "...", "filters": {...}},
                "depends_on": [],
                "parallel_group": null
            }
        ],
        "required_tools": ["vector_search", "database_query"],
        "execution_mode": "parallel|sequential|hybrid"
    },
    "reasoning": ["step-by-step reasoning"]
}

### EXAMPLES:

**Example 1 - Factual Query:**
Query: "What was the voter turnout in Patna during 2020 election?"
→ Intent: factual, Complexity: low, Steps: 1 (QUERY_DB)

**Example 2 - Comparative Query:**
Query: "Compare manifesto promises of BJP and Congress on agriculture in Bihar"
→ Intent: comparative, Complexity: medium, Steps: 2 (parallel SEARCH_VECTOR for both parties)

**Example 3 - Analytical Query:**
Query: "Why did AAP lose seats in Punjab 2024 compared to 2022?"
→ Intent: analytical, Complexity: high, Steps: 4 (sequential with comparison and external context)
"""


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def classify_intent_and_plan(state: AgentState) -> Dict[str, Any]:
    """
    Combined function that classifies intent AND creates execution plan.
    
    This replaces the separate intent_classification_node and planning_node.
    """
    logger.info(f"Classifying intent and planning execution for: {state['query']}")
    
    query = state['query']
    entities = state.get('query_entities', {})
    history = state.get('conversation_history', [])
    
    # Build context
    context = f"""
USER QUERY: {query}

EXTRACTED ENTITIES: {entities}

CONVERSATION HISTORY: {history[-3:] if history else 'No previous context'}

Generate a comprehensive intent classification and execution plan.
"""
    
    try:
        # Get model
        model = get_planning_model()
        
        # Generate response with structured output
        response = model.generate(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": context}
            ],
            temperature=0.3,  # Lower temperature for consistent planning
            max_tokens=2000
        )
        
        # Parse response
        import json
        try:
            # Try to extract JSON from response
            response_text = response.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            result_data = json.loads(response_text)
            
            # Validate and create Pydantic models
            intent_result = IntentPlanningResult(**result_data)
            
            reasoning = state.get('reasoning_steps', []) + intent_result.reasoning
            reasoning.append(f"Intent classified as: {intent_result.intent_category} (confidence: {intent_result.confidence})")
            reasoning.append(f"Execution plan created with {len(intent_result.execution_plan.steps)} steps")
            
            return {
                "intent": {
                    "category": intent_result.intent_category,
                    "description": intent_result.intent,
                    "confidence": intent_result.confidence,
                    "complexity": intent_result.complexity
                },
                "execution_plan": intent_result.execution_plan.model_dump(),
                "reasoning_steps": reasoning,
                "updated_at": datetime.utcnow()
            }
            
        except (json.JSONDecodeError, Exception) as parse_error:
            logger.warning(f"Failed to parse structured response: {parse_error}")
            # Fallback to simple plan
            return _create_fallback_plan(state, str(parse_error))
            
    except Exception as e:
        logger.error(f"Intent planning failed: {e}")
        return _create_fallback_plan(state, str(e))


def _create_fallback_plan(state: AgentState, error_msg: str) -> Dict[str, Any]:
    """Create a simple fallback plan when LLM planning fails."""
    query = state['query']
    entities = state.get('query_entities', {})
    
    fallback_plan = ExecutionPlan(
        intent=query,
        intent_category="factual",
        complexity="low",
        summary="Fallback plan due to planning error",
        steps=[
            PlanStep(
                step_id=1,
                action="SEARCH_VECTOR",
                description="Perform general semantic search",
                query_params={"query": query, "filters": entities},
                depends_on=[],
                parallel_group=None
            )
        ],
        required_tools=["vector_search"],
        execution_mode="sequential"
    )
    
    reasoning = state.get('reasoning_steps', []) + [
        f"Planning failed: {error_msg}",
        "Using fallback single-step retrieval plan"
    ]
    
    return {
        "intent": {
            "category": "factual",
            "description": query,
            "confidence": 0.5,
            "complexity": "low"
        },
        "execution_plan": fallback_plan.model_dump(),
        "reasoning_steps": reasoning,
        "updated_at": datetime.utcnow()
    }


# ============================================================================
# LANGGRAPH NODE WRAPPER
# ============================================================================

def intent_planning_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node wrapper for the combined intent planning agent.
    """
    logger.info(f"[IntentPlanningAgent] Processing query: {state['query'][:100]}...")
    return classify_intent_and_plan(state)


# ============================================================================
# STANDALONE GRAPH (FOR TESTING)
# ============================================================================

def create_intent_planning_graph():
    """Create a standalone LangGraph for intent planning (useful for testing)."""
    
    workflow = StateGraph(AgentState)
    
    # Add node
    workflow.add_node("intent_planning", intent_planning_node)
    
    # Set entry point
    workflow.set_entry_point("intent_planning")
    
    # Add edge to end
    workflow.add_edge("intent_planning", END)
    
    # Compile with memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def run_intent_planning(query: str, entities: Dict[str, Any] = None, session_id: str = "test") -> Dict[str, Any]:
    """
    Run intent planning as a standalone operation.
    Useful for testing and debugging.
    """
    graph = create_intent_planning_graph()
    
    initial_state: AgentState = {
        "query": query,
        "query_entities": entities or {},
        "conversation_history": [],
        "intent": None,
        "execution_plan": None,
        "reasoning_steps": [],
        "updated_at": datetime.utcnow()
    }
    
    config = {"configurable": {"thread_id": session_id}}
    result = graph.invoke(initial_state, config=config)
    
    return result


if __name__ == "__main__":
    # Test the agent
    test_query = "Compare the manifesto promises of BJP and Congress regarding agriculture in Bihar"
    result = run_intent_planning(test_query)
    
    print("\n=== INTENT PLANNING RESULT ===\n")
    print(f"Intent Category: {result['intent']['category']}")
    print(f"Complexity: {result['intent']['complexity']}")
    print(f"Confidence: {result['intent']['confidence']}")
    print(f"\nExecution Plan Summary: {result['execution_plan']['summary']}")
    print(f"Number of Steps: {len(result['execution_plan']['steps'])}")
    print(f"Execution Mode: {result['execution_plan']['execution_mode']}")
    print(f"Required Tools: {result['execution_plan']['required_tools']}")
    
    print("\n=== STEPS ===\n")
    for step in result['execution_plan']['steps']:
        print(f"Step {step['step_id']}: {step['action']}")
        print(f"  Description: {step['description']}")
        print(f"  Dependencies: {step['depends_on']}")
        print(f"  Parallel Group: {step['parallel_group']}")
        print()
