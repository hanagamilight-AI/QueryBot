"""
Database models and connection management for QueryBot
Using PostgreSQL with pgvector extension for hybrid search
"""
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import VECTOR
from datetime import datetime
from typing import List, Optional
import numpy as np

from config.settings import settings

Base = declarative_base()


class Document(Base):
    """Main document table for storing political intelligence data"""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(Text, nullable=False)
    embedding = Column(VECTOR(settings.PGVECTOR_DIMENSION))
    
    # Metadata fields for filtering
    source_type = Column(String(50))  # election, survey, social_media, manifesto
    source_id = Column(String(200))  # Original source identifier
    constituency = Column(String(100))
    state = Column(String(100))
    party = Column(String(100))
    candidate = Column(String(200))
    language = Column(String(20), default="en")
    sentiment_score = Column(Float)
    
    # Temporal metadata
    document_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Additional metadata as JSON
    metadata_json = Column(JSON)
    
    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    """Chunked documents for better retrieval granularity"""
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(VECTOR(settings.PGVECTOR_DIMENSION))
    
    # Reference to parent document
    document = relationship("Document", back_populates="chunks")


class ConversationHistory(Base):
    """Store conversation history for agent memory"""
    __tablename__ = "conversation_history"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # user, assistant, system
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metadata_json = Column(JSON)


# Database connection
engine = create_engine(settings.DATABASE_URL, echo=settings.DEBUG)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Dependency for FastAPI to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables and pgvector extension"""
    # Create pgvector extension if not exists
    with engine.connect() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully!")


def similarity_search(query_embedding: np.ndarray, limit: int = 5, filters: dict = None):
    """
    Perform vector similarity search with optional metadata filters
    """
    from sqlalchemy import text
    
    embedding_str = "[" + ",".join(map(str, query_embedding.tolist())) + "]"
    
    query = text("""
        SELECT d.id, d.content, d.source_type, d.constituency, d.state, 
               d.party, d.candidate, d.document_date, d.sentiment_score,
               1 - (d.embedding <=> :embedding) as similarity
        FROM documents d
        WHERE 1=1
    """)
    
    # Add filters dynamically
    params = {"embedding": embedding_str}
    
    if filters:
        filter_clauses = []
        for key, value in filters.items():
            if hasattr(Document, key):
                filter_clauses.append(f"d.{key} = :{key}")
                params[key] = value
        
        if filter_clauses:
            query = text(str(query) + " AND " + " AND ".join(filter_clauses))
    
    query = query.order_by(text("similarity DESC")).limit(limit)
    
    with SessionLocal() as session:
        results = session.execute(query, params).fetchall()
        return results
