# Parameter-Aware Hybrid Cache for QueryBot

## Overview

The Parameter-Aware Hybrid Cache is a sophisticated caching layer built on Redis that provides intelligent response caching for the QueryBot Political Intelligence AI system. It combines exact matching with semantic similarity search to maximize cache hits while maintaining accuracy.

## Key Features

### 1. **Parameter-Aware Key Generation**
- Generates deterministic cache keys based on query content AND parameters
- Parameters like constituency, party, date range are factored into cache key
- Ensures different parameter combinations don't collide in cache

### 2. **Hybrid Matching Strategy**
- **Exact Match**: Fast hash-based lookup for identical queries
- **Semantic Match**: Vector similarity search for semantically similar queries
- Configurable similarity threshold (default: 0.85)

### 3. **Adaptive TTL (Time-To-Live)**
- Automatically adjusts cache expiration based on:
  - Access frequency (popular queries live longer)
  - Recency (recently accessed entries get bonus)
  - Age (older entries gradually expire faster)
- Base TTL: 1 hour, Max: 24 hours, Min: 5 minutes

### 4. **Namespace Isolation**
- Separate cache namespaces for different data types:
  - `QUERY_RESPONSE`: User query responses
  - `TOOL_RESULT`: MCP tool execution results
  - `EMBEDDING`: Computed embeddings
  - `AGENT_STATE`: Agent conversation states
  - `GUARDRAIL_CHECK`: Validation results

### 5. **Cache Invalidation Strategies**
- By specific query
- By parameter value (e.g., invalidate all Delhi-related when new data arrives)
- By namespace
- Pattern-based invalidation

### 6. **Metrics & Observability**
- Hit/miss rates
- Semantic vs exact hit breakdown
- Retrieval time tracking
- Redis connection health

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              ParameterAwareHybridCache                   │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────────┐    ┌──────────────────────────┐  │
│  │ CacheKeyGenerator │    │ SemanticSimilarityMatcher │  │
│  │ - Query hashing   │    │ - Cosine similarity      │  │
│  │ - Parameter sort  │    │ - Vector comparison      │  │
│  └──────────────────┘    └──────────────────────────┘  │
│                                                         │
│  ┌──────────────────┐    ┌──────────────────────────┐  │
│  │ AdaptiveTTLManager│    │   CacheEntry Manager     │  │
│  │ - Access patterns │    │ - Store/retrieve         │  │
│  │ - Recency bonus   │    │ - Expiration check       │  │
│  └──────────────────┘    └──────────────────────────┘  │
│                                                         │
│                    Redis Backend                        │
└─────────────────────────────────────────────────────────┘
```

## Installation

1. **Add dependencies** (already in requirements.txt):
```bash
pip install redis hiredis
```

2. **Start Redis server**:
```bash
# Using Docker
docker run -d -p 6379:6379 --name querybot-redis redis:latest

# Or install locally
sudo apt-get install redis-server
sudo systemctl start redis
```

3. **Configure** (in `.env`):
```env
REDIS_URL=redis://localhost:6379/0
CACHE_TTL_DEFAULT=3600
CACHE_SIMILARITY_THRESHOLD=0.85
CACHE_ENABLED=true
```

## Usage

### Basic Operations

```python
from cache.hybrid_cache import get_cache, CacheNamespace

# Get cache instance
cache = get_cache()

# Store a response
cache.set(
    query="What was BJP's vote share in Delhi 2024?",
    value={
        "response": "BJP received 45.2%...",
        "documents": [...],
        "confidence": 0.92
    },
    parameters={"constituency": "Delhi", "party": "BJP", "year": 2024},
    namespace=CacheNamespace.QUERY_RESPONSE,
    ttl=3600
)

# Retrieve cached response
hit = cache.get(
    query="What was BJP's vote share in Delhi 2024?",
    parameters={"constituency": "Delhi", "party": "BJP"},
    namespace=CacheNamespace.QUERY_RESPONSE
)

if hit:
    print(f"Cache {hit.hit_type} match!")
    print(f"Response: {hit.entry.value['response']}")
    print(f"Retrieval time: {hit.retrieval_time_ms:.2f}ms")
```

### Semantic Similarity Matching

```python
from embeddings.embedder import get_embedder

embedder = get_embedder()

# Generate query embedding
query_vector = embedder.embed_query("Tell me about Delhi election results")

# Search with semantic matching
hit = cache.get(
    query="Tell me about Delhi election results",
    query_vector=query_vector,  # Enables semantic search
    namespace=CacheNamespace.QUERY_RESPONSE
)

if hit and hit.hit_type == "semantic":
    print(f"Semantic match with similarity: {hit.similarity_score:.3f}")
```

### Cache Warming

```python
# Pre-populate cache with common queries
common_queries = [
    (
        "What is the ruling party in Delhi?",
        {"response": "...", "documents": [], "confidence": 0.95},
        {"constituency": "Delhi"},
        None
    ),
    # ... more queries
]

cache.warm_cache(common_queries, namespace=CacheNamespace.QUERY_RESPONSE)
```

### Invalidation

```python
# Invalidate specific query
cache.invalidate(query="What was BJP's vote share?")

# Invalidate by namespace
cache.invalidate(namespace=CacheNamespace.QUERY_RESPONSE)

# Invalidate all entries with specific parameter
cache.invalidate_by_parameter(
    param_name="constituency",
    param_value="Delhi",
    namespace=CacheNamespace.QUERY_RESPONSE
)

# Clear all cache
cache.clear_all()
```

### Statistics

```python
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate_percent']}%")
print(f"Semantic hits: {stats['semantic_hits']}")
print(f"Total requests: {stats['total_requests']}")
```

## Integration with LangGraph Agent

The cache is automatically integrated into the LangGraph agent workflow:

1. **Retrieval Node**: Checks cache before executing expensive vector searches
2. **Synthesis Node**: Caches generated responses for future use
3. **Automatic**: No code changes needed in agent usage

```python
from agents.langgraph_agent import build_agent_graph
from datetime import datetime

graph = build_agent_graph()

# First query - cache miss, executes full pipeline
result1 = graph.invoke({
    "query": "Voter turnout in Delhi 2024?",
    "session_id": "session_1",
    "conversation_history": [],
    "created_at": datetime.utcnow(),
    "updated_at": datetime.utcnow(),
    "iteration_count": 0
})

# Second identical query - cache hit, instant response
result2 = graph.invoke({
    "query": "Voter turnout in Delhi 2024?",
    "session_id": "session_2",
    "conversation_history": [],
    "created_at": datetime.utcnow(),
    "updated_at": datetime.utcnow(),
    "iteration_count": 0
})

print(f"From cache: {result2.get('from_cache', False)}")
```

## Cache Entry Structure

```python
{
    "key": "querybot:query_response:a1b2c3d4e5f6",
    "value": {
        "response": "Full response text...",
        "documents": [...],
        "confidence": 0.92,
        "intent": "factual"
    },
    "created_at": 1704067200.0,
    "accessed_at": 1704070800.0,
    "access_count": 15,
    "ttl": 7200,
    "parameters_hash": "md5hash123",
    "semantic_vector": [0.1, 0.2, ...],  # Optional embedding
    "namespace": "query_response"
}
```

## Performance Considerations

### When to Use Cache
- ✓ Frequently asked questions
- ✓ Expensive-to-compute responses
- ✓ Static or slowly-changing data
- ✓ Common political queries

### When NOT to Use Cache
- ✗ Real-time data requirements
- ✗ Highly personalized responses
- ✗ Rapidly changing information
- ✗ Security-sensitive queries

### Optimization Tips

1. **Adjust similarity threshold** based on your domain:
   ```python
   # Stricter matching (fewer false positives)
   cache = get_cache(similarity_threshold=0.90)
   
   # Looser matching (more cache hits)
   cache = get_cache(similarity_threshold=0.80)
   ```

2. **Use appropriate TTL** per namespace:
   ```python
   # Short TTL for volatile data
   cache.set(query, value, ttl=300, namespace=CacheNamespace.TOOL_RESULT)
   
   # Long TTL for stable data
   cache.set(query, value, ttl=86400, namespace=CacheNamespace.QUERY_RESPONSE)
   ```

3. **Monitor and adjust** based on metrics:
   ```python
   stats = cache.get_stats()
   if stats['hit_rate_percent'] < 50:
       # Consider lowering similarity threshold or warming cache
   ```

## Testing

Run the demo to see all features in action:

```bash
cd /workspace/querybot
python examples/cache_demo.py
```

This demonstrates:
- Basic caching operations
- Semantic similarity matching
- Adaptive TTL behavior
- Cache invalidation
- Cache warming
- LangGraph integration

## Troubleshooting

### Redis Connection Issues
```python
# Test connection
cache = get_cache()
if not cache._test_connection():
    print("Redis not available, caching disabled")
```

### High Miss Rate
- Check if similarity threshold is too high
- Implement cache warming for common queries
- Verify parameter consistency

### Memory Concerns
- Use shorter TTL values
- Implement LRU eviction policy
- Monitor Redis memory usage

## Future Enhancements

- [ ] Multi-level caching (L1 memory + L2 Redis)
- [ ] Distributed cache coordination
- [ ] Predictive pre-fetching
- [ ] A/B testing for cache strategies
- [ ] Compression for large responses

## License

Part of QueryBot Political Intelligence AI system.
