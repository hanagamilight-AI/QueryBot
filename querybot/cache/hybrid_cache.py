"""
Parameter-Aware Hybrid Cache for QueryBot

Implements a sophisticated caching layer using Redis with:
- Parameter-aware cache key generation
- Hybrid caching (exact match + semantic similarity)
- TTL management with adaptive expiration
- Cache invalidation strategies
- Hit/miss metrics tracking
"""

import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

import redis
from loguru import logger

from config.settings import settings


class CacheStrategy(Enum):
    """Cache strategy types"""
    EXACT_MATCH = "exact_match"
    SEMANTIC_SIMILAR = "semantic_similar"
    HYBRID = "hybrid"


class CacheNamespace(Enum):
    """Cache namespaces for different data types"""
    QUERY_RESPONSE = "query_response"
    TOOL_RESULT = "tool_result"
    EMBEDDING = "embedding"
    AGENT_STATE = "agent_state"
    GUARDRAIL_CHECK = "guardrail_check"


@dataclass
class CacheEntry:
    """Represents a cached entry with metadata"""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: int = 3600  # Default 1 hour
    parameters_hash: str = ""
    semantic_vector: Optional[List[float]] = None
    namespace: str = "default"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "access_count": self.access_count,
            "ttl": self.ttl,
            "parameters_hash": self.parameters_hash,
            "semantic_vector": self.semantic_vector,
            "namespace": self.namespace
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        """Create from dictionary"""
        return cls(**data)
    
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        return time.time() > (self.created_at + self.ttl)
    
    def touch(self):
        """Update access time and count"""
        self.accessed_at = time.time()
        self.access_count += 1


@dataclass
class CacheHit:
    """Represents a cache hit with metadata"""
    entry: CacheEntry
    hit_type: str  # "exact" or "semantic"
    similarity_score: Optional[float] = None
    retrieval_time_ms: float = 0.0


class ParameterAwareCacheKeyGenerator:
    """Generates cache keys based on query parameters"""
    
    @staticmethod
    def generate_key(
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        namespace: CacheNamespace = CacheNamespace.QUERY_RESPONSE
    ) -> str:
        """
        Generate a deterministic cache key from query and parameters
        
        Args:
            query: The user query string
            parameters: Additional parameters (constituency, date_range, etc.)
            namespace: Cache namespace
            
        Returns:
            Unique cache key string
        """
        # Normalize query
        normalized_query = " ".join(query.lower().split())
        
        # Sort parameters for deterministic ordering
        if parameters:
            sorted_params = json.dumps(parameters, sort_keys=True, default=str)
        else:
            sorted_params = ""
        
        # Create combined string
        combined = f"{namespace.value}:{normalized_query}:{sorted_params}"
        
        # Generate hash
        key_hash = hashlib.sha256(combined.encode()).hexdigest()[:16]
        
        return f"querybot:{namespace.value}:{key_hash}"
    
    @staticmethod
    def generate_parameters_hash(parameters: Dict[str, Any]) -> str:
        """Generate hash of parameters only"""
        sorted_params = json.dumps(parameters, sort_keys=True, default=str)
        return hashlib.md5(sorted_params.encode()).hexdigest()


class SemanticSimilarityMatcher:
    """Matches semantically similar queries using vector embeddings"""
    
    def __init__(self, redis_client: redis.Redis, similarity_threshold: float = 0.85):
        self.redis = redis_client
        self.similarity_threshold = similarity_threshold
        self.index_name = "querybot:semantic_index"
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2:
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def find_similar(
        self,
        query_vector: List[float],
        namespace: CacheNamespace = CacheNamespace.QUERY_RESPONSE,
        limit: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find semantically similar cached entries
        
        Args:
            query_vector: Embedding vector of the query
            namespace: Cache namespace to search
            limit: Maximum number of results
            
        Returns:
            List of (cache_key, similarity_score) tuples
        """
        pattern = f"querybot:{namespace.value}:*"
        keys = self.redis.keys(pattern)
        
        if not keys:
            return []
        
        results = []
        for key in keys:
            try:
                data = self.redis.get(key)
                if not data:
                    continue
                
                entry_data = json.loads(data)
                stored_vector = entry_data.get("semantic_vector")
                
                if stored_vector:
                    similarity = self._cosine_similarity(query_vector, stored_vector)
                    if similarity >= self.similarity_threshold:
                        results.append((key.decode() if isinstance(key, bytes) else key, similarity))
            except Exception as e:
                logger.warning(f"Error checking semantic similarity for key: {e}")
        
        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]


class AdaptiveTTLManager:
    """Manages adaptive TTL based on access patterns"""
    
    def __init__(
        self,
        base_ttl: int = 3600,
        max_ttl: int = 86400,
        min_ttl: int = 300
    ):
        self.base_ttl = base_ttl
        self.max_ttl = max_ttl
        self.min_ttl = min_ttl
    
    def calculate_ttl(self, entry: CacheEntry) -> int:
        """
        Calculate adaptive TTL based on access patterns
        
        Factors:
        - Access frequency (more accesses = longer TTL)
        - Recency (recently accessed = longer TTL)
        - Time since creation (older = shorter TTL)
        """
        now = time.time()
        age = now - entry.created_at
        time_since_access = now - entry.accessed_at
        
        # Base multiplier from access count (logarithmic scaling)
        access_multiplier = min(3.0, 1.0 + (entry.access_count * 0.2))
        
        # Recency bonus (recently accessed gets bonus)
        recency_bonus = 1.5 if time_since_access < 300 else 1.0
        
        # Age penalty (older entries get shorter TTL)
        age_factor = max(0.5, 1.0 - (age / 86400))  # Reduce by half after 1 day
        
        calculated_ttl = int(
            self.base_ttl * access_multiplier * recency_bonus * age_factor
        )
        
        return max(self.min_ttl, min(self.max_ttl, calculated_ttl))
    
    def update_entry_ttl(self, redis_client: redis.Redis, entry: CacheEntry) -> None:
        """Update TTL of an entry in Redis"""
        new_ttl = self.calculate_ttl(entry)
        key = entry.key
        
        try:
            # Re-set with new TTL
            redis_client.setex(key, new_ttl, json.dumps(entry.to_dict()))
            entry.ttl = new_ttl
            logger.debug(f"Updated TTL for {key} to {new_ttl}s")
        except Exception as e:
            logger.error(f"Failed to update TTL for {key}: {e}")


class ParameterAwareHybridCache:
    """
    Main hybrid cache implementation with parameter awareness
    
    Features:
    - Exact match caching
    - Semantic similarity matching
    - Parameter-sensitive keys
    - Adaptive TTL
    - Hit/miss metrics
    - Automatic invalidation
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        similarity_threshold: float = 0.85,
        enable_semantic: bool = True,
        enable_adaptive_ttl: bool = True
    ):
        self.redis_url = redis_url or settings.REDIS_URL
        self.similarity_threshold = similarity_threshold
        self.enable_semantic = enable_semantic
        self.enable_adaptive_ttl = enable_adaptive_ttl
        
        # Initialize Redis client
        self.redis = redis.from_url(
            self.redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5
        )
        
        # Initialize components
        self.key_generator = ParameterAwareCacheKeyGenerator()
        self.semantic_matcher = SemanticSimilarityMatcher(
            self.redis, 
            similarity_threshold
        )
        self.ttl_manager = AdaptiveTTLManager()
        
        # Metrics
        self.hits = 0
        self.misses = 0
        self.semantic_hits = 0
        
        logger.info(f"Initialized ParameterAwareHybridCache with Redis: {self.redis_url}")
    
    def _test_connection(self) -> bool:
        """Test Redis connection"""
        try:
            self.redis.ping()
            return True
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            return False
    
    def get(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        namespace: CacheNamespace = CacheNamespace.QUERY_RESPONSE,
        query_vector: Optional[List[float]] = None
    ) -> Optional[CacheHit]:
        """
        Get cached response with hybrid matching
        
        Args:
            query: User query string
            parameters: Query parameters (constituency, date_range, etc.)
            namespace: Cache namespace
            query_vector: Optional embedding vector for semantic search
            
        Returns:
            CacheHit if found, None otherwise
        """
        start_time = time.time()
        
        # Try exact match first
        exact_key = self.key_generator.generate_key(query, parameters, namespace)
        
        try:
            data = self.redis.get(exact_key)
            if data:
                entry_data = json.loads(data)
                
                # Check expiration
                if entry_data["created_at"] + entry_data["ttl"] < time.time():
                    self.redis.delete(exact_key)
                    self.misses += 1
                    return None
                
                # Update access stats
                entry = CacheEntry.from_dict(entry_data)
                entry.touch()
                
                # Update TTL if enabled
                if self.enable_adaptive_ttl:
                    self.ttl_manager.update_entry_ttl(self.redis, entry)
                else:
                    # Just update the entry
                    self.redis.setex(
                        exact_key, 
                        entry.ttl, 
                        json.dumps(entry.to_dict())
                    )
                
                retrieval_time = (time.time() - start_time) * 1000
                self.hits += 1
                
                logger.debug(f"Cache HIT (exact): {exact_key}")
                return CacheHit(
                    entry=entry,
                    hit_type="exact",
                    retrieval_time_ms=retrieval_time
                )
        except Exception as e:
            logger.error(f"Error retrieving exact match: {e}")
        
        # Try semantic match if enabled and vector provided
        if self.enable_semantic and query_vector:
            try:
                similar = self.semantic_matcher.find_similar(
                    query_vector, 
                    namespace,
                    limit=1
                )
                
                if similar:
                    similar_key, similarity = similar[0]
                    data = self.redis.get(similar_key)
                    
                    if data:
                        entry_data = json.loads(data)
                        
                        # Check expiration
                        if entry_data["created_at"] + entry_data["ttl"] < time.time():
                            self.redis.delete(similar_key)
                            self.misses += 1
                            return None
                        
                        entry = CacheEntry.from_dict(entry_data)
                        entry.touch()
                        
                        if self.enable_adaptive_ttl:
                            self.ttl_manager.update_entry_ttl(self.redis, entry)
                        else:
                            self.redis.setex(
                                similar_key,
                                entry.ttl,
                                json.dumps(entry.to_dict())
                            )
                        
                        retrieval_time = (time.time() - start_time) * 1000
                        self.hits += 1
                        self.semantic_hits += 1
                        
                        logger.debug(f"Cache HIT (semantic): {similar_key} (similarity: {similarity:.3f})")
                        return CacheHit(
                            entry=entry,
                            hit_type="semantic",
                            similarity_score=similarity,
                            retrieval_time_ms=retrieval_time
                        )
            except Exception as e:
                logger.error(f"Error retrieving semantic match: {e}")
        
        self.misses += 1
        logger.debug(f"Cache MISS for query: {query[:50]}...")
        return None
    
    def set(
        self,
        query: str,
        value: Any,
        parameters: Optional[Dict[str, Any]] = None,
        namespace: CacheNamespace = CacheNamespace.QUERY_RESPONSE,
        ttl: Optional[int] = None,
        query_vector: Optional[List[float]] = None
    ) -> str:
        """
        Store response in cache
        
        Args:
            query: User query string
            value: Response to cache
            parameters: Query parameters
            namespace: Cache namespace
            ttl: Optional custom TTL
            query_vector: Optional embedding vector for semantic search
            
        Returns:
            Cache key
        """
        key = self.key_generator.generate_key(query, parameters, namespace)
        params_hash = self.key_generator.generate_parameters_hash(parameters or {})
        
        entry = CacheEntry(
            key=key,
            value=value,
            ttl=ttl or 3600,
            parameters_hash=params_hash,
            semantic_vector=query_vector,
            namespace=namespace.value
        )
        
        try:
            self.redis.setex(key, entry.ttl, json.dumps(entry.to_dict()))
            logger.debug(f"Cache SET: {key} (TTL: {entry.ttl}s)")
            return key
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            raise
    
    def invalidate(
        self,
        query: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        namespace: Optional[CacheNamespace] = None,
        pattern: Optional[str] = None
    ) -> int:
        """
        Invalidate cache entries
        
        Args:
            query: Specific query to invalidate
            parameters: Parameters to match
            namespace: Namespace to clear
            pattern: Custom pattern to match
            
        Returns:
            Number of keys deleted
        """
        keys_to_delete = []
        
        if pattern:
            keys_to_delete = self.redis.keys(pattern)
        elif query:
            key = self.key_generator.generate_key(
                query, 
                parameters, 
                namespace or CacheNamespace.QUERY_RESPONSE
            )
            keys_to_delete = [key]
        elif namespace:
            pattern = f"querybot:{namespace.value}:*"
            keys_to_delete = self.redis.keys(pattern)
        
        if not keys_to_delete:
            return 0
        
        try:
            deleted = self.redis.delete(*keys_to_delete)
            logger.info(f"Invalidated {deleted} cache entries")
            return deleted
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")
            return 0
    
    def invalidate_by_parameter(
        self,
        param_name: str,
        param_value: Any,
        namespace: CacheNamespace = CacheNamespace.QUERY_RESPONSE
    ) -> int:
        """
        Invalidate all entries matching a specific parameter
        
        Useful when underlying data changes (e.g., new election data)
        """
        pattern = f"querybot:{namespace.value}:*"
        keys = self.redis.keys(pattern)
        deleted_count = 0
        
        target_hash = hashlib.md5(
            json.dumps({param_name: param_value}, sort_keys=True).encode()
        ).hexdigest()
        
        for key in keys:
            try:
                data = self.redis.get(key)
                if not data:
                    continue
                
                entry_data = json.loads(data)
                # Note: This is a simplified check; production would need better param tracking
                self.redis.delete(key)
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Error checking key {key}: {e}")
        
        logger.info(f"Invalidated {deleted_count} entries for parameter {param_name}={param_value}")
        return deleted_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        semantic_hit_rate = (self.semantic_hits / self.hits * 100) if self.hits > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "semantic_hits": self.semantic_hits,
            "hit_rate_percent": round(hit_rate, 2),
            "semantic_hit_rate_percent": round(semantic_hit_rate, 2),
            "total_requests": total_requests,
            "redis_connected": self._test_connection()
        }
    
    def clear_all(self, namespace: Optional[CacheNamespace] = None) -> int:
        """Clear all cache entries"""
        if namespace:
            pattern = f"querybot:{namespace.value}:*"
        else:
            pattern = "querybot:*"
        
        keys = self.redis.keys(pattern)
        if not keys:
            return 0
        
        deleted = self.redis.delete(*keys)
        logger.info(f"Cleared {deleted} cache entries")
        
        # Reset metrics
        self.hits = 0
        self.misses = 0
        self.semantic_hits = 0
        
        return deleted
    
    def warm_cache(
        self,
        queries: List[Tuple[str, Any, Optional[Dict], Optional[List[float]]]],
        namespace: CacheNamespace = CacheNamespace.QUERY_RESPONSE
    ) -> int:
        """
        Pre-warm cache with common queries
        
        Args:
            queries: List of (query, value, parameters, vector) tuples
            namespace: Cache namespace
            
        Returns:
            Number of entries cached
        """
        cached_count = 0
        for query, value, params, vector in queries:
            try:
                self.set(query, value, params, namespace, query_vector=vector)
                cached_count += 1
            except Exception as e:
                logger.error(f"Failed to warm cache for query '{query}': {e}")
        
        logger.info(f"Warmed cache with {cached_count} entries")
        return cached_count


# Singleton instance
_cache_instance: Optional[ParameterAwareHybridCache] = None


def get_cache(
    redis_url: Optional[str] = None,
    similarity_threshold: float = 0.85
) -> ParameterAwareHybridCache:
    """Get or create cache singleton instance"""
    global _cache_instance
    
    if _cache_instance is None:
        _cache_instance = ParameterAwareHybridCache(
            redis_url=redis_url,
            similarity_threshold=similarity_threshold
        )
    
    return _cache_instance


def initialize_cache(
    redis_url: Optional[str] = None,
    similarity_threshold: float = 0.85,
    force_new: bool = False
) -> ParameterAwareHybridCache:
    """Initialize a new cache instance"""
    global _cache_instance
    
    if force_new or _cache_instance is None:
        _cache_instance = ParameterAwareHybridCache(
            redis_url=redis_url,
            similarity_threshold=similarity_threshold
        )
    
    return _cache_instance
