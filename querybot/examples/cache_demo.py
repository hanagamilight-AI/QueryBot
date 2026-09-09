"""
Example usage and tests for Parameter-Aware Hybrid Cache

Demonstrates:
- Basic cache operations
- Parameter-aware caching
- Semantic similarity matching
- Adaptive TTL
- Cache invalidation
- Integration with LangGraph agent
"""

import asyncio
import time
from typing import List, Dict, Any

from cache.hybrid_cache import (
    ParameterAwareHybridCache,
    CacheNamespace,
    get_cache,
    initialize_cache,
)
from config.settings import settings


def demo_basic_caching():
    """Demonstrate basic cache operations"""
    print("\n" + "="*60)
    print("BASIC CACHING DEMO")
    print("="*60)
    
    # Initialize cache
    cache = initialize_cache(force_new=True)
    
    # Store a response
    query = "What was BJP's vote share in Delhi 2024?"
    response_data = {
        "response": "BJP received 45.2% vote share in Delhi during the 2024 elections...",
        "documents": [{"id": 1, "content": "Election results..."}],
        "confidence": 0.92
    }
    parameters = {"constituency": "Delhi", "party": "BJP", "year": 2024}
    
    print(f"\n1. Storing response for query: '{query}'")
    cache_key = cache.set(
        query=query,
        value=response_data,
        parameters=parameters,
        namespace=CacheNamespace.QUERY_RESPONSE,
        ttl=3600
    )
    print(f"   Cache key: {cache_key}")
    
    # Retrieve the cached response
    print("\n2. Retrieving cached response...")
    hit = cache.get(
        query=query,
        parameters=parameters,
        namespace=CacheNamespace.QUERY_RESPONSE
    )
    
    if hit:
        print(f"   ✓ Cache HIT ({hit.hit_type})")
        print(f"   Retrieval time: {hit.retrieval_time_ms:.2f}ms")
        print(f"   Access count: {hit.entry.access_count}")
        print(f"   Response preview: {hit.entry.value['response'][:50]}...")
    else:
        print("   ✗ Cache MISS")
    
    # Try with different parameters (should miss)
    print("\n3. Trying with different parameters...")
    hit2 = cache.get(
        query=query,
        parameters={"constituency": "Mumbai", "party": "BJP"},
        namespace=CacheNamespace.QUERY_RESPONSE
    )
    
    if hit2:
        print(f"   ✓ Cache HIT")
    else:
        print(f"   ✗ Cache MISS (expected - different parameters)")
    
    return cache


def demo_semantic_matching():
    """Demonstrate semantic similarity matching"""
    print("\n" + "="*60)
    print("SEMANTIC SIMILARITY MATCHING DEMO")
    print("="*60)
    
    cache = initialize_cache(force_new=True, similarity_threshold=0.80)
    
    # Store original query with vector
    original_query = "What were the election results in Delhi constituency?"
    original_vector = [0.1] * 384  # Simulated embedding
    original_vector[0] = 1.0  # Make it distinctive
    
    response_data = {
        "response": "Delhi election results showed...",
        "documents": [],
        "confidence": 0.88
    }
    
    print(f"\n1. Storing original query with vector embedding")
    cache.set(
        query=original_query,
        value=response_data,
        query_vector=original_vector,
        namespace=CacheNamespace.QUERY_RESPONSE
    )
    
    # Try similar query with similar vector
    similar_query = "Tell me about Delhi's election outcomes"
    similar_vector = [0.1] * 384
    similar_vector[0] = 0.95  # Very similar to original
    
    print(f"\n2. Querying with semantically similar query")
    print(f"   Original: '{original_query}'")
    print(f"   Similar:  '{similar_query}'")
    
    hit = cache.get(
        query=similar_query,
        query_vector=similar_vector,
        namespace=CacheNamespace.QUERY_RESPONSE
    )
    
    if hit and hit.hit_type == "semantic":
        print(f"   ✓ Semantic Cache HIT!")
        print(f"   Similarity score: {hit.similarity_score:.3f}")
        print(f"   This demonstrates parameter-aware semantic matching")
    else:
        print(f"   Note: Semantic matching depends on actual embedding vectors")
    
    return cache


def demo_adaptive_ttl():
    """Demonstrate adaptive TTL behavior"""
    print("\n" + "="*60)
    print("ADAPTIVE TTL DEMO")
    print("="*60)
    
    cache = initialize_cache(force_new=True)
    
    query = "Demo query for TTL"
    response = {"response": "Test response", "documents": []}
    
    print("\n1. Setting initial cache entry")
    cache.set(query, response, namespace=CacheNamespace.QUERY_RESPONSE, ttl=3600)
    
    # Access multiple times to trigger adaptive TTL
    print("\n2. Accessing entry multiple times...")
    for i in range(5):
        hit = cache.get(query, namespace=CacheNamespace.QUERY_RESPONSE)
        if hit:
            print(f"   Access {i+1}: TTL adjusted, access_count={hit.entry.access_count}")
        time.sleep(0.1)
    
    stats = cache.get_stats()
    print(f"\n3. Cache statistics:")
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Hit rate: {stats['hit_rate_percent']}%")
    
    return cache


def demo_cache_invalidation():
    """Demonstrate cache invalidation strategies"""
    print("\n" + "="*60)
    print("CACHE INVALIDATION DEMO")
    print("="*60)
    
    cache = initialize_cache(force_new=True)
    
    # Add multiple entries
    queries = [
        ("Query about Delhi", {"constituency": "Delhi"}),
        ("Query about Mumbai", {"constituency": "Mumbai"}),
        ("Query about Chennai", {"constituency": "Chennai"}),
    ]
    
    print("\n1. Adding multiple cache entries")
    for query, params in queries:
        cache.set(
            query=query,
            value={"response": f"Response for {query}", "documents": []},
            parameters=params,
            namespace=CacheNamespace.QUERY_RESPONSE
        )
        print(f"   Added: '{query}'")
    
    # Invalidate by namespace
    print("\n2. Invalidating all QUERY_RESPONSE entries")
    deleted = cache.invalidate(namespace=CacheNamespace.QUERY_RESPONSE)
    print(f"   Deleted {deleted} entries")
    
    # Verify invalidation
    print("\n3. Verifying invalidation")
    hit = cache.get(queries[0][0], parameters=queries[0][1])
    if hit:
        print("   ✗ Entry still exists (unexpected)")
    else:
        print("   ✓ Entry successfully invalidated")
    
    return cache


def demo_cache_warming():
    """Demonstrate cache warming with common queries"""
    print("\n" + "="*60)
    print("CACHE WARMING DEMO")
    print("="*60)
    
    cache = initialize_cache(force_new=True)
    
    # Common political queries
    common_queries = [
        (
            "What is the current ruling party in Delhi?",
            {
                "response": "The current ruling party in Delhi is AAP (Aam Aadmi Party)...",
                "documents": [{"source": "election_data_2024"}],
                "confidence": 0.95
            },
            {"constituency": "Delhi", "type": "ruling_party"},
            None
        ),
        (
            "Show voter turnout trends for Maharashtra",
            {
                "response": "Maharashtra voter turnout has increased by 5% since 2019...",
                "documents": [{"source": "eci_reports"}],
                "confidence": 0.88
            },
            {"state": "Maharashtra", "type": "turnout"},
            None
        ),
        (
            "Compare BJP vs Congress performance in 2024",
            {
                "response": "BJP gained 12 seats while Congress lost 8 seats compared to 2019...",
                "documents": [{"source": "comparative_analysis"}],
                "confidence": 0.91
            },
            {"parties": ["BJP", "Congress"], "year": 2024},
            None
        ),
    ]
    
    print("\n1. Pre-warming cache with common queries")
    cached_count = cache.warm_cache(common_queries, namespace=CacheNamespace.QUERY_RESPONSE)
    print(f"   Successfully cached {cached_count}/{len(common_queries)} queries")
    
    # Test warmed cache
    print("\n2. Testing warmed cache")
    for query, _, params, _ in common_queries:
        hit = cache.get(query, parameters=params)
        status = "✓ HIT" if hit else "✗ MISS"
        print(f"   {status}: '{query[:40]}...'")
    
    return cache


def demo_statistics():
    """Display comprehensive cache statistics"""
    print("\n" + "="*60)
    print("CACHE STATISTICS")
    print("="*60)
    
    cache = get_cache()
    stats = cache.get_stats()
    
    print(f"""
    Redis Connection: {'✓ Connected' if stats['redis_connected'] else '✗ Disconnected'}
    
    Request Metrics:
    ├─ Total Requests: {stats['total_requests']}
    ├─ Cache Hits:     {stats['hits']}
    ├─ Cache Misses:   {stats['misses']}
    └─ Semantic Hits:  {stats['semantic_hits']}
    
    Performance:
    ├─ Hit Rate:              {stats['hit_rate_percent']}%
    └─ Semantic Hit Rate:     {stats['semantic_hit_rate_percent']}%
    """)


async def demo_langgraph_integration():
    """Demonstrate cache integration with LangGraph agent"""
    print("\n" + "="*60)
    print("LANGGRAPH INTEGRATION DEMO")
    print("="*60)
    
    from agents.langgraph_agent import build_agent_graph
    from datetime import datetime
    
    # Build the graph
    print("\n1. Building LangGraph agent with cache integration")
    graph = build_agent_graph()
    print("   ✓ Graph built successfully")
    
    # First query (cache miss)
    query = "What was the voter turnout in Delhi 2024?"
    print(f"\n2. First query (expecting cache MISS): '{query}'")
    
    initial_state = {
        "query": query,
        "session_id": "demo_session_1",
        "conversation_history": [],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "iteration_count": 0
    }
    
    start = time.time()
    result1 = graph.invoke(initial_state)
    time1 = time.time() - start
    
    print(f"   Response time: {time1:.3f}s")
    print(f"   From cache: {result1.get('from_cache', False)}")
    
    # Second identical query (cache hit)
    print(f"\n3. Second identical query (expecting cache HIT)")
    
    initial_state["session_id"] = "demo_session_2"
    initial_state["updated_at"] = datetime.utcnow()
    
    start = time.time()
    result2 = graph.invoke(initial_state)
    time2 = time.time() - start
    
    print(f"   Response time: {time2:.3f}s")
    print(f"   From cache: {result2.get('from_cache', False)}")
    print(f"   Speedup: {time1/time2:.2f}x faster" if time2 < time1 else "")
    
    # Show cache stats
    demo_statistics()


def run_all_demos():
    """Run all demonstration scenarios"""
    print("\n" + "#"*60)
    print("# PARAMETER-AWARE HYBRID CACHE - DEMONSTRATION")
    print("#"*60)
    
    try:
        # Run demos
        demo_basic_caching()
        demo_semantic_matching()
        demo_adaptive_ttl()
        demo_cache_invalidation()
        demo_cache_warming()
        demo_statistics()
        
        # Async integration demo
        print("\n\n" + "#"*60)
        print("# LANGGRAPH INTEGRATION")
        print("#"*60)
        asyncio.run(demo_langgraph_integration())
        
        print("\n" + "="*60)
        print("ALL DEMOS COMPLETED SUCCESSFULLY")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n⚠ Demo encountered an error: {e}")
        print("Note: Some demos require Redis to be running")
        print("      Start Redis with: docker run -d -p 6379:6379 redis:latest")


if __name__ == "__main__":
    run_all_demos()
