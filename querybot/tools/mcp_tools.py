"""
MCP (Model Context Protocol) Tools for QueryBot
Exposes database, pgvector retriever, and external APIs as callable tools
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from database.models import SessionLocal, Document, DocumentChunk, ConversationHistory
from embeddings.embedder import get_embedding_service
from config.settings import settings


class ToolResult(BaseModel):
    """Standardized tool result format"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DatabaseTool:
    """
    MCP Tool for SQL database operations
    Allows agents to query structured political data
    """
    
    name = "database_query"
    description = """
    Query the political intelligence database for structured information.
    Supports filtering by constituency, state, party, candidate, date range, and source type.
    """
    
    def execute(self, query_type: str, filters: Dict[str, Any] = None, limit: int = 10) -> ToolResult:
        """
        Execute database query
        
        Args:
            query_type: Type of query (elections, surveys, manifestos, social_media)
            filters: Dictionary of filter criteria
            limit: Maximum number of results
            
        Returns:
            ToolResult with query results
        """
        try:
            session = SessionLocal()
            
            # Build query based on type
            if query_type == "elections":
                results = self._query_elections(session, filters, limit)
            elif query_type == "surveys":
                results = self._query_surveys(session, filters, limit)
            elif query_type == "manifestos":
                results = self._query_manifestos(session, filters, limit)
            elif query_type == "social_media":
                results = self._query_social_media(session, filters, limit)
            else:
                results = self._query_general(session, filters, limit)
            
            return ToolResult(
                success=True,
                data=results,
                metadata={"count": len(results), "query_type": query_type}
            )
        except Exception as e:
            logger.error(f"Database query error: {e}")
            return ToolResult(success=False, error=str(e))
        finally:
            session.close()
    
    def _query_elections(self, session, filters: dict, limit: int):
        """Query election data"""
        query = session.query(Document).filter(Document.source_type == "election")
        
        if filters:
            if 'constituency' in filters:
                query = query.filter(Document.constituency == filters['constituency'])
            if 'state' in filters:
                query = query.filter(Document.state == filters['state'])
            if 'party' in filters:
                query = query.filter(Document.party == filters['party'])
            if 'date_from' in filters:
                query = query.filter(Document.document_date >= filters['date_from'])
            if 'date_to' in filters:
                query = query.filter(Document.document_date <= filters['date_to'])
        
        return [self._doc_to_dict(doc) for doc in query.limit(limit).all()]
    
    def _query_surveys(self, session, filters: dict, limit: int):
        """Query survey data"""
        query = session.query(Document).filter(Document.source_type == "survey")
        
        if filters:
            if 'constituency' in filters:
                query = query.filter(Document.constituency == filters['constituency'])
            if 'sentiment_min' in filters:
                query = query.filter(Document.sentiment_score >= filters['sentiment_min'])
        
        return [self._doc_to_dict(doc) for doc in query.limit(limit).all()]
    
    def _query_manifestos(self, session, filters: dict, limit: int):
        """Query manifesto data"""
        query = session.query(Document).filter(Document.source_type == "manifesto")
        
        if filters:
            if 'party' in filters:
                query = query.filter(Document.party == filters['party'])
            if 'candidate' in filters:
                query = query.filter(Document.candidate == filters['candidate'])
        
        return [self._doc_to_dict(doc) for doc in query.limit(limit).all()]
    
    def _query_social_media(self, session, filters: dict, limit: int):
        """Query social media data"""
        query = session.query(Document).filter(Document.source_type == "social_media")
        
        if filters:
            if 'constituency' in filters:
                query = query.filter(Document.constituency == filters['constituency'])
            if 'sentiment_min' in filters:
                query = query.filter(Document.sentiment_score >= filters['sentiment_min'])
        
        return [self._doc_to_dict(doc) for doc in query.limit(limit).all()]
    
    def _query_general(self, session, filters: dict, limit: int):
        """General query across all source types"""
        query = session.query(Document)
        
        if filters:
            if 'source_type' in filters:
                query = query.filter(Document.source_type == filters['source_type'])
            if 'constituency' in filters:
                query = query.filter(Document.constituency == filters['constituency'])
            if 'state' in filters:
                query = query.filter(Document.state == filters['state'])
            if 'party' in filters:
                query = query.filter(Document.party == filters['party'])
        
        return [self._doc_to_dict(doc) for doc in query.limit(limit).all()]
    
    def _doc_to_dict(self, doc: Document) -> dict:
        """Convert Document to dictionary"""
        return {
            "id": doc.id,
            "content": doc.content[:500],  # Truncate for display
            "source_type": doc.source_type,
            "constituency": doc.constituency,
            "state": doc.state,
            "party": doc.party,
            "candidate": doc.candidate,
            "document_date": doc.document_date.isoformat() if doc.document_date else None,
            "sentiment_score": doc.sentiment_score
        }


class VectorRetrieverTool:
    """
    MCP Tool for vector similarity search using pgvector
    Enables semantic search across political intelligence corpus
    """
    
    name = "vector_search"
    description = """
    Perform semantic similarity search across political documents.
    Finds documents most similar to the query in meaning, not just keywords.
    Supports metadata filtering for refined results.
    """
    
    def __init__(self):
        self.embedding_service = get_embedding_service()
    
    def execute(
        self, 
        query: str, 
        limit: int = 5,
        filters: Dict[str, Any] = None,
        min_similarity: float = 0.5
    ) -> ToolResult:
        """
        Execute vector similarity search
        
        Args:
            query: Search query text
            limit: Maximum number of results
            filters: Metadata filters (constituency, party, etc.)
            min_similarity: Minimum similarity threshold
            
        Returns:
            ToolResult with search results
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.embed_query(query)
            
            # Perform search
            results = self._similarity_search(query_embedding, limit, filters, min_similarity)
            
            return ToolResult(
                success=True,
                data=results,
                metadata={
                    "query": query,
                    "count": len(results),
                    "filters": filters
                }
            )
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return ToolResult(success=False, error=str(e))
    
    def _similarity_search(
        self, 
        embedding: np.ndarray, 
        limit: int, 
        filters: dict,
        min_similarity: float
    ):
        """Execute similarity search with pgvector"""
        session = SessionLocal()
        
        try:
            # Convert embedding to list for PostgreSQL
            embedding_str = "[" + ",".join(map(str, embedding.tolist())) + "]"
            
            # Build query with cosine similarity
            from sqlalchemy import text
            
            query = text("""
                SELECT d.id, d.content, d.source_type, d.constituency, d.state,
                       d.party, d.candidate, d.document_date, d.sentiment_score,
                       d.metadata_json,
                       1 - (d.embedding <=> :embedding::vector) as similarity
                FROM documents d
                WHERE 1 - (d.embedding <=> :embedding::vector) >= :min_sim
            """)
            
            params = {
                "embedding": embedding_str,
                "min_sim": min_similarity
            }
            
            # Add filters
            if filters:
                filter_clauses = []
                for key, value in filters.items():
                    if key in ['constituency', 'state', 'party', 'candidate', 'source_type']:
                        filter_clauses.append(f"d.{key} = :{key}")
                        params[key] = value
                
                if filter_clauses:
                    query = text(str(query) + " AND " + " AND ".join(filter_clauses))
            
            query = query.order_by(text("similarity DESC")).limit(limit)
            
            results = session.execute(query, params).fetchall()
            
            return [
                {
                    "id": row.id,
                    "content": row.content,
                    "source_type": row.source_type,
                    "constituency": row.constituency,
                    "state": row.state,
                    "party": row.party,
                    "candidate": row.candidate,
                    "document_date": row.document_date.isoformat() if row.document_date else None,
                    "sentiment_score": row.sentiment_score,
                    "metadata": row.metadata_json,
                    "similarity": float(row.similarity)
                }
                for row in results
            ]
        finally:
            session.close()


class ExternalAPITool:
    """
    MCP Tool for external API integrations
    News APIs, social listening platforms, etc.
    """
    
    name = "external_api"
    description = """
    Fetch real-time data from external sources like news APIs and social media platforms.
    Use for current events and trending topics not yet in the database.
    """
    
    def execute(self, api_type: str, query: str, location: str = None) -> ToolResult:
        """
        Call external API
        
        Args:
            api_type: Type of API (news, twitter, reddit)
            query: Search query
            location: Geographic filter
            
        Returns:
            ToolResult with API results
        """
        try:
            if api_type == "news":
                results = self._fetch_news(query, location)
            elif api_type == "twitter":
                results = self._fetch_twitter(query, location)
            elif api_type == "reddit":
                results = self._fetch_reddit(query)
            else:
                return ToolResult(success=False, error=f"Unknown API type: {api_type}")
            
            return ToolResult(
                success=True,
                data=results,
                metadata={"api_type": api_type, "query": query}
            )
        except Exception as e:
            logger.error(f"External API error: {e}")
            return ToolResult(success=False, error=str(e))
    
    def _fetch_news(self, query: str, location: str = None) -> List[Dict]:
        """Fetch news articles (mock implementation)"""
        # In production, integrate with NewsAPI, GNews, etc.
        logger.info(f"Fetching news for: {query}, location: {location}")
        return [
            {
                "source": "news_api",
                "title": f"News about {query}",
                "content": "Mock news content for demonstration",
                "url": "https://example.com/news",
                "published_at": datetime.utcnow().isoformat()
            }
        ]
    
    def _fetch_twitter(self, query: str, location: str = None) -> List[Dict]:
        """Fetch Twitter data (mock implementation)"""
        # In production, integrate with Twitter API v2
        logger.info(f"Fetching tweets for: {query}")
        return [
            {
                "source": "twitter",
                "text": f"Mock tweet about {query}",
                "author": "@mock_user",
                "created_at": datetime.utcnow().isoformat()
            }
        ]
    
    def _fetch_reddit(self, query: str) -> List[Dict]:
        """Fetch Reddit posts (mock implementation)"""
        # In production, integrate with Reddit API
        logger.info(f"Fetching Reddit posts for: {query}")
        return [
            {
                "source": "reddit",
                "title": f"Discussion about {query}",
                "content": "Mock Reddit post content",
                "subreddit": "r/politics",
                "score": 150
            }
        ]


class ConversationMemoryTool:
    """
    MCP Tool for managing conversation history
    Enables long-term memory for the agent
    """
    
    name = "conversation_memory"
    description = """
    Store and retrieve conversation history for maintaining context across interactions.
    Supports session-based memory management.
    """
    
    def execute(self, action: str, session_id: str, **kwargs) -> ToolResult:
        """
        Manage conversation memory
        
        Args:
            action: Action to perform (store, retrieve, clear)
            session_id: Unique session identifier
            kwargs: Additional parameters based on action
            
        Returns:
            ToolResult with memory operation results
        """
        try:
            if action == "store":
                return self._store_message(session_id, kwargs.get('role'), kwargs.get('message'))
            elif action == "retrieve":
                return self._retrieve_history(session_id, kwargs.get('limit', 10))
            elif action == "clear":
                return self._clear_history(session_id)
            else:
                return ToolResult(success=False, error=f"Unknown action: {action}")
        except Exception as e:
            logger.error(f"Conversation memory error: {e}")
            return ToolResult(success=False, error=str(e))
    
    def _store_message(self, session_id: str, role: str, message: str) -> ToolResult:
        """Store a message in conversation history"""
        session = SessionLocal()
        try:
            conv = ConversationHistory(
                session_id=session_id,
                role=role,
                message=message
            )
            session.add(conv)
            session.commit()
            return ToolResult(success=True, metadata={"session_id": session_id})
        finally:
            session.close()
    
    def _retrieve_history(self, session_id: str, limit: int) -> ToolResult:
        """Retrieve conversation history"""
        session = SessionLocal()
        try:
            history = session.query(ConversationHistory).filter(
                ConversationHistory.session_id == session_id
            ).order_by(
                ConversationHistory.timestamp.desc()
            ).limit(limit).all()
            
            return ToolResult(
                success=True,
                data=[
                    {"role": h.role, "message": h.message, "timestamp": h.timestamp.isoformat()}
                    for h in reversed(history)
                ],
                metadata={"session_id": session_id, "count": len(history)}
            )
        finally:
            session.close()
    
    def _clear_history(self, session_id: str) -> ToolResult:
        """Clear conversation history for a session"""
        session = SessionLocal()
        try:
            session.query(ConversationHistory).filter(
                ConversationHistory.session_id == session_id
            ).delete()
            session.commit()
            return ToolResult(success=True, metadata={"session_id": session_id})
        finally:
            session.close()


# Registry of all available MCP tools
MCP_TOOLS = {
    "database_query": DatabaseTool(),
    "vector_search": VectorRetrieverTool(),
    "external_api": ExternalAPITool(),
    "conversation_memory": ConversationMemoryTool()
}


def get_tool(tool_name: str):
    """Get a tool by name"""
    return MCP_TOOLS.get(tool_name)


def list_tools() -> List[Dict]:
    """List all available tools with their descriptions"""
    return [
        {
            "name": tool.name,
            "description": tool.description
        }
        for tool in MCP_TOOLS.values()
    ]
