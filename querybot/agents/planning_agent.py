"""
Planning Agent for QueryBot.
Decomposes complex queries into executable steps based on intent and entities.
"""
from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from querybot.config.settings import settings


# Define AgentState locally to avoid circular imports
class AgentState(Dict[str, Any]):
    """Minimal state dict for planning agent."""
    pass


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
    plan_id: str = Field(..., description="Unique ID for this plan")
    intent: str = Field(..., description="The classified intent driving this plan")
    summary: str = Field(..., description="Brief summary of the strategy")
    steps: List[PlanStep] = Field(..., description="Ordered list of steps to execute")
    estimated_complexity: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        ..., description="Estimated complexity of the plan"
    )


planning_agent = Agent(
    name="Planning Agent",
    role="Strategic Query Planner",
    model=OpenAIChat(
        id=settings.LLM_MODEL,
        api_key=settings.OPENROUTER_API_KEY if settings.LLM_PROVIDER == "openrouter" else settings.OPENAI_API_KEY,
        base_url="https://openrouter.ai/api/v1" if settings.LLM_PROVIDER == "openrouter" else None,
    ),
    description="You are an expert strategic planner for political intelligence queries. "
                "Your job is to decompose complex user queries into a structured, executable plan of steps.",
    instructions=[
        "Analyze the user query, extracted entities, and classified intent.",
        "Break down the query into logical, atomic steps required to answer it.",
        "Identify which steps can be run in parallel (assign same parallel_group).",
        "Identify dependencies between steps (use depends_on).",
        "Select the appropriate action type for each step:",
        "  - SEARCH_VECTOR: For semantic search in pgvector (manifestos, speeches).",
        "  - QUERY_DB: For structured SQL queries (election results, demographics).",
        "  - CALL_TOOL: For external API calls (news, social media).",
        "  - CALCULATE: For mathematical operations (turnout %, growth rates).",
        "  - COMPARE: For side-by-side analysis of retrieved data.",
        "Ensure the plan is efficient and minimizes unnecessary steps.",
        "Output ONLY the valid JSON ExecutionPlan object.",
    ],
    markdown=False,
)


def plan_query(state: AgentState) -> dict:
    """
    Generate an execution plan based on query analysis and intent.
    This is the node function used in LangGraph workflow.
    """
    query = state["query"]
    entities = state.get("entities", {})
    intent = state.get("intent", "UNKNOWN")
    intent_category = intent.get("category", "GENERAL") if isinstance(intent, dict) else intent

    # Construct context for the planner
    context = f"""
    User Query: {query}
    Classified Intent: {intent_category}
    Extracted Entities: {entities}
    
    Create a detailed execution plan to answer this query.
    """

    try:
        response = planning_agent.run(context)
        plan: ExecutionPlan = response.content
        
        return {
            "execution_plan": plan.model_dump(),
            "reasoning_steps": state.get("reasoning_steps", []) + [f"Generated execution plan with {len(plan.steps)} steps"]
        }
    except Exception as e:
        # Fallback to a simple single-step plan if planning fails
        fallback_plan = ExecutionPlan(
            plan_id="fallback-001",
            intent=intent_category,
            summary="Fallback simple retrieval plan",
            steps=[
                PlanStep(
                    step_id=1,
                    action="SEARCH_VECTOR",
                    description="Perform general semantic search",
                    query_params={"query": query}
                )
            ],
            estimated_complexity="LOW"
        )
        return {
            "execution_plan": fallback_plan.model_dump(),
            "reasoning_steps": state.get("reasoning_steps", []) + [f"Planning failed, using fallback plan: {str(e)}"]
        }


def planning_node(state: AgentState) -> dict:
    """
    Planning node wrapper for LangGraph integration.
    Calls the plan_query function and returns the result.
    """
    from loguru import logger
    logger.info(f"Planning execution strategy for query: {state['query']}")
    return plan_query(state)
