"""
Embedding service for document vectorization
Supports multiple embedding providers via Agno abstraction
"""
from typing import List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger

from config.settings import settings


class EmbeddingService:
    """Unified embedding service supporting multiple providers"""
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.model = None
        self.dimension = settings.PGVECTOR_DIMENSION
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            actual_dim = self.model.get_sentence_embedding_dimension()
            if actual_dim != self.dimension:
                logger.warning(
                    f"Model dimension ({actual_dim}) != configured dimension ({self.dimension}). "
                    f"Updating configuration."
                )
                self.dimension = actual_dim
            logger.info(f"Model loaded successfully with dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text to embed
            
        Returns:
            Numpy array of embeddings
        """
        if not text or not text.strip():
            return np.zeros(self.dimension)
        
        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(self.dimension)
    
    def embed_texts(self, texts: List[str], batch_size: int = None) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts in batch
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            List of numpy arrays containing embeddings
        """
        if not texts:
            return []
        
        batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        embeddings = []
        
        try:
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=len(texts) > 100
                )
                embeddings.extend([emb.astype(np.float32) for emb in batch_embeddings])
            
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            return [np.zeros(self.dimension) for _ in texts]
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Specialized embedding for search queries
        Can be overridden for asymmetric retrieval setups
        """
        return self.embed_text(query)
    
    def embed_documents(self, documents: List[str]) -> List[np.ndarray]:
        """
        Specialized embedding for documents
        Can be overridden for asymmetric retrieval setups
        """
        return self.embed_texts(documents)


# Singleton instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the embedding service singleton"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
