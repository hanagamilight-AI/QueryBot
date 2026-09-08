"""
FastAPI REST API for QueryBot Political Intelligence AI
Exposes the agent as endpoints for dashboard consumption
"""
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger
import uuid

from config.settings import settings
from agents.langgraph_agent import get_agent, QueryBotAgent
from database.models import init_db, get_db
from data_ingestion.ingestion import run_ingestion_pipeline


# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
    ## QueryBot - Political Intelligence Conversational AI
    
    A sophisticated agent-powered system for querying political intelligence data including:
    - Election results and trends
    - Survey data and sentiment analysis
    - Party manifestos and promises
    - Social media sentiment
    - Constituency-level analytics
    
    ### Features
    - **Semantic Search**: Vector-based similarity search across all documents
    - **Multi-turn Conversations**: Context-aware chat with memory
    - **Source Attribution**: All responses cite original sources
    - **Confidence Scoring**: Transparency about answer reliability
    """,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    question: str = Field(..., min_length=1, max_length=2000, description="User's question")
    session_id: Optional[str] = Field(None, description="Session ID for conversation context")
    include_sources: bool = Field(True, description="Whether to include source citations")
    include_reasoning: bool = Field(False, description="Whether to include reasoning trace")


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    response: str
    confidence: float
    sources_count: int
    session_id: str
    reasoning: Optional[List[str]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(..., description="Session ID for maintaining conversation context")


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    message: str
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class IngestionRequest(BaseModel):
    """Request model for data ingestion"""
    source_directory: str
    source_type: str = Field(..., description="Type: election, survey, social_media, manifesto")


class IngestionResponse(BaseModel):
    """Response model for data ingestion"""
    success: bool
    documents_ingested: int
    message: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    components: Dict[str, str]


# ============================================================================
# DEPENDENCIES
# ============================================================================

def get_querybot_agent() -> QueryBotAgent:
    """Dependency to get agent instance"""
    return get_agent()


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": "Political Intelligence Conversational AI",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint for monitoring
    """
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        components={
            "api": "ok",
            "agent": "initialized",
            "database": "connected"
        }
    )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_endpoint(
    request: QueryRequest,
    agent: QueryBotAgent = Depends(get_querybot_agent)
):
    """
    Process a single query through the political intelligence agent
    
    Returns:
    - Natural language response
    - Confidence score
    - Source count
    - Optional reasoning trace
    """
    try:
        # Generate session ID if not provided
        session_id = request.session_id or f"session_{uuid.uuid4().hex[:12]}"
        
        # Process query through agent
        result = agent.query(
            question=request.question,
            session_id=session_id,
            stream=False
        )
        
        # Build response
        response = QueryResponse(
            response=result['response'],
            confidence=result['confidence'],
            sources_count=result['sources'],
            session_id=session_id,
            reasoning=result['reasoning'] if request.include_reasoning else None
        )
        
        logger.info(f"Query processed. Session: {session_id}, Confidence: {result['confidence']:.2f}")
        
        return response
        
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(
    request: ChatRequest,
    agent: QueryBotAgent = Depends(get_querybot_agent)
):
    """
    Chat endpoint maintaining conversation context
    
    Use the same session_id across multiple requests to maintain context.
    """
    try:
        response_text = agent.chat(
            message=request.message,
            session_id=request.session_id
        )
        
        logger.info(f"Chat message processed. Session: {request.session_id}")
        
        return ChatResponse(
            message=response_text,
            session_id=request.session_id
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversation/{session_id}", tags=["Conversation"])
async def get_conversation(
    session_id: str,
    agent: QueryBotAgent = Depends(get_querybot_agent)
):
    """
    Retrieve conversation history for a session
    """
    try:
        history = agent.get_conversation_history(session_id)
        return {
            "session_id": session_id,
            "messages": history,
            "count": len(history)
        }
    except Exception as e:
        logger.error(f"Get conversation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/conversation/{session_id}", tags=["Conversation"])
async def clear_conversation(
    session_id: str,
    agent: QueryBotAgent = Depends(get_querybot_agent)
):
    """
    Clear conversation history for a session
    """
    try:
        success = agent.clear_session(session_id)
        return {
            "success": success,
            "session_id": session_id,
            "message": "Conversation cleared" if success else "Failed to clear conversation"
        }
    except Exception as e:
        logger.error(f"Clear conversation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestionResponse, tags=["Data Management"])
async def ingest_data(
    request: IngestionRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger data ingestion pipeline
    
    Can be run in background for large datasets.
    """
    try:
        # For small datasets, run synchronously
        # For production, use background_tasks or a task queue
        
        from data_ingestion.ingestion import DataIngestionPipeline
        from pathlib import Path
        
        pipeline = DataIngestionPipeline()
        count = pipeline.ingest_from_directory(
            request.source_directory,
            request.source_type
        )
        pipeline.close()
        
        logger.info(f"Ingested {count} documents from {request.source_directory}")
        
        return IngestionResponse(
            success=True,
            documents_ingested=count,
            message=f"Successfully ingested {count} documents"
        )
        
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        return IngestionResponse(
            success=False,
            documents_ingested=0,
            message=f"Ingestion failed: {str(e)}"
        )


@app.post("/ingest/all", tags=["Data Management"])
async def ingest_all_data(background_tasks: BackgroundTasks):
    """
    Run complete ingestion pipeline across all configured directories
    """
    try:
        # Run in background for large datasets
        def run_pipeline():
            try:
                count = run_ingestion_pipeline()
                logger.info(f"Complete ingestion finished: {count} documents")
            except Exception as e:
                logger.error(f"Pipeline error: {e}")
        
        background_tasks.add_task(run_pipeline)
        
        return {
            "status": "started",
            "message": "Ingestion pipeline started in background"
        }
        
    except Exception as e:
        logger.error(f"Ingestion start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tools", tags=["System"])
async def list_tools():
    """
    List available MCP tools
    """
    from tools.mcp_tools import list_tools as get_tools
    return {
        "tools": get_tools()
    }


@app.get("/stats", tags=["System"])
async def get_stats(db=Depends(get_db)):
    """
    Get database statistics
    """
    try:
        from database.models import Document, DocumentChunk
        
        total_docs = db.query(Document).count()
        total_chunks = db.query(DocumentChunk).count()
        
        # Count by source type
        by_source = db.query(
            Document.source_type, 
            db.func.count(Document.id)
        ).group_by(Document.source_type).all()
        
        source_counts = {source: count for source, count in by_source}
        
        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "by_source_type": source_counts
        }
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# STARTUP EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    
    # Initialize database
    try:
        init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.warning(f"Database initialization skipped: {e}")
    
    # Initialize agent
    try:
        get_agent()
        logger.info("Agent initialized")
    except Exception as e:
        logger.error(f"Agent initialization error: {e}")
    
    logger.info("Startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down QueryBot API")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else settings.API_WORKERS
    )
