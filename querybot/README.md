# QueryBot - Political Intelligence Conversational AI

## Overview

QueryBot is a sophisticated agent-powered conversational AI system designed for political intelligence analysis. Built with **LangGraph**, **PostgreSQL + pgvector**, **Redis**, and **FastAPI**, it enables natural language querying of election data, survey results, social media sentiment, and manifesto documents. The system achieves **~70% reduction in manual political analysis effort** through autonomous agentic workflows.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CLIENT LAYER                                  │
│                    Dashboard / REST API Consumers                       │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ HTTP/REST
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           API LAYER (FastAPI)                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ │
│  │  /query         │  │  /chat          │  │  /evaluate             │ │
│  │  Single queries │  │  Multi-turn     │  │  Evaluation harness    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │              Parameter-Aware Hybrid Cache (Redis)                   ││
│  │   • Response caching with parameter hashing                         ││
│  │   • TTL-based expiration                                            ││
│  │   • Cache invalidation on data updates                              ││
│  └─────────────────────────────────────────────────────────────────────┘│
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      AGENT ORCHESTRATION (LangGraph)                    │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    State Graph Controller                        │  │
│  │  Manages conversation state, context, and node transitions       │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │
│  │   QUERY      │ →  │   RETRIEVAL  │ →  │   SYNTHESIS  │             │
│  │   ANALYSIS   │    │     NODE     │    │     NODE     │             │
│  │   NODE       │    │              │    │              │             │
│  │ • Intent     │    │ • Vector     │    │ • Response   │             │
│  │   classify   │    │   search     │    │   generation │             │
│  │ • Entity     │    │ • SQL query  │    │ • Source     │             │
│  │   extract    │    │ • Hybrid     │    │   attribution│             │
│  │ • Guardrail  │    │   retrieval  │    │ • Confidence │             │
│  │   check      │    │ • Re-ranking │    │   scoring    │             │
│  └──────────────┘    └──────────────┘    └──────────────┘             │
│         │                                      │                       │
│         │                   ┌──────────────────┘                       │
│         │                   ▼                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │
│  │   INTENT     │    │  VALIDATION  │ ←  │   MEMORY     │             │
│  │CLASSIFICATION│    │     NODE     │    │     NODE     │             │
│  │ • Factual    │    │ • Output     │    │ • Short-term │             │
│  │ • Comparative│    │   validation │    │   (session)  │             │
│  │ • Trend      │    │ • Guardrail  │    │ • Long-term  │             │
│  │ • Analytical │    │   compliance │    │   (vector)   │             │
│  └──────────────┘    └──────────────┘    └──────────────┘             │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │              GUARDRAILS & SAFETY LAYER                           │  │
│  │  • Input/Output validation  • Action confirmation                │  │
│  │  • Rate limiting            • Retry/Rollback logic               │  │
│  │  • Political domain rules   • PII/Sensitive data detection       │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      MCP TOOL LAYER (Model Context Protocol)            │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌─────────────┐ │
│  │   DATABASE    │ │    VECTOR     │ │   EXTERNAL    │ │ CONVERSATION│ │
│  │     TOOL      │ │    SEARCH     │ │     APIs      │ │   MEMORY    │ │
│  │ • SQL queries │ │ • Semantic    │ │ • News APIs   │ │ • Session   │ │
│  │ • Filters     │ │   search      │ │ • Social      │ │   storage   │ │
│  │ • Aggregations│ │ • Re-ranking  │ │   listening   │ │ • History   │ │
│  └───────────────┘ └───────────────┘ └───────────────┘ └─────────────┘ │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                      │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │              PostgreSQL + pgvector Extension                      │ │
│  │  ┌─────────────────────┐  ┌─────────────────────────────────────┐ │ │
│  │  │   Relational Data   │  │        Vector Embeddings            │ │ │
│  │  │  • Election results │  │  • Document chunks (768-dim)        │ │ │
│  │  │  • Survey data      │  │  • Semantic similarity search       │ │ │
│  │  │  • Constituencies   │  │  • Metadata-filtered retrieval      │ │ │
│  │  │  • Manifestos       │  │  • Hybrid (vector + SQL) queries    │ │ │
│  │  │  • Social posts     │  │                                     │ │ │
│  │  └─────────────────────┘  └─────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                    Redis Cache Layer                              │ │
│  │  • Parameter-aware response caching                               │ │
│  │  • Session state storage                                          │ │
│  │  • Rate limiting counters                                         │ │
│  └───────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                   CROSS-CUTTING CONCERNS                                │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────┐ │
│  │   OBSERVABILITY     │  │    EVALUATION       │  │  MODEL ADAPTERS │ │
│  │  • Langfuse traces  │  │  • Test harness     │  │  • OpenRouter   │ │
│  │  • LangSmith runs   │  │  • Golden datasets  │  │  • Local models │ │
│  │  • Metrics export   │  │  • LLM-as-judge     │  │  • Adapter API  │ │
│  │  • Grafana dash     │  │  • Regression tests │  │                 │ │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Agent Architecture

The QueryBot system uses a **multi-agent architecture** orchestrated by LangGraph's state machine. Each agent is a specialized node responsible for a specific task in the query processing pipeline.

### 1. Query Analysis Agent

**Purpose**: Initial query processing and entity extraction

**Responsibilities**:
- Parse natural language input (tokenization, normalization)
- Extract entities (constituencies, parties, dates, candidates, locations)
- Detect query language for multi-language support
- Identify query ambiguity or missing context
- Apply input guardrails (length limits, harmful content detection, PII filtering)
- Pass structured query to Intent Planning Agent for classification

**Input**: Raw user query + session context
**Output**: Structured query with extracted entities and metadata

**Entity Extraction Examples**:
```json
{
  "original_query": "What was BJP's performance in Maharashtra 2019?",
  "entities": {
    "parties": ["BJP"],
    "constituencies": [],
    "states": ["Maharashtra"],
    "years": ["2019"],
    "candidates": []
  },
  "language": "en",
  "ambiguity_score": 0.1,
  "requires_clarification": false
}
```

**Guardrails Applied**:
- Input length validation (max 2000 chars)
- Harmful content detection
- SQL injection pattern blocking
- PII redaction for sensitive personal data
- Language detection for proper routing

**Note**: Intent classification is NOT performed here - it's delegated to the Intent Planning Agent for better separation of concerns.

---

### 2. Intent Planning Agent (Combined Intent Classification + Planning)

**Purpose**: Classify query intent AND create execution plan for retrieval

**Why Combined?**: 
- Eliminates redundant processing between separate intent and planning agents
- Ensures intent classification directly informs retrieval strategy
- Reduces latency by performing classification and planning in single step
- Better alignment between identified intent and execution plan

**Responsibilities**:
1. **Intent Classification**:
   - Determine query type (factual, comparative, trend, analytical, sentiment)
   - Assess complexity level (simple, moderate, complex)
   - Identify required data sources

2. **Query Decomposition**:
   - Break complex queries into sub-queries
   - Identify dependencies between sub-tasks
   - Determine parallel vs sequential execution

3. **Tool Selection**:
   - Select appropriate MCP tools (Database, Vector Search, External APIs)
   - Define tool parameters and filters
   - Specify expected output format

4. **Strategy Definition**:
   - Choose retrieval strategy (direct, hybrid, multi-hop)
   - Set confidence thresholds
   - Define fallback mechanisms

**Intent Categories**:
| Intent Type | Description | Example | Retrieval Strategy |
|-------------|-------------|---------|-------------------|
| **Factual** | Single fact lookup | "Who won Delhi 2020?" | Direct vector search + exact match |
| **Comparative** | Compare entities | "BJP vs Congress in Maharashtra" | Multi-query retrieval + comparison |
| **Trend** | Temporal analysis | "Voter turnout trend 2014-2024" | Time-series SQL + vector context |
| **Analytical** | Complex reasoning | "Why did AAP lose in rural areas?" | Multi-hop retrieval + CoT reasoning |
| **Sentiment** | Opinion/sentiment | "Public perception of Modi" | Social media API + sentiment analysis |

**Planning Output Structure**:
```json
{
  "intent": "comparative",
  "complexity": "moderate",
  "sub_queries": [
    {"query": "BJP performance Maharashtra 2019", "tool": "vector_search"},
    {"query": "Congress performance Maharashtra 2019", "tool": "vector_search"}
  ],
  "execution_plan": {
    "parallel_groups": [["sub_query_1", "sub_query_2"]],
    "sequential_steps": ["synthesize_comparison"],
    "dependencies": {"synthesize_comparison": ["sub_query_1", "sub_query_2"]}
  },
  "tools_required": ["vector_search", "sql_query"],
  "confidence_threshold": 0.75,
  "fallback_strategy": "return_partial_results"
}
```

**Decision Logic**:
- Factual → Direct retrieval, single tool call, low latency path
- Comparative → Parallel retrieval for each entity, then synthesis
- Trend → SQL aggregation with time filters + vector context
- Analytical → Multi-hop retrieval with iterative refinement
- Sentiment → External API calls + sentiment scoring

---

### 3. Retrieval Agent

**Purpose**: Fetch relevant information from data sources

**Retrieval Strategies**:

#### A. Vector Search
- Semantic similarity using pgvector (cosine distance)
- Top-K retrieval (configurable, default K=10)
- Metadata filtering (constituency, date range, party, source type)
- Re-ranking using cross-encoder for precision

#### B. Hybrid Retrieval
- Combines vector search with structured SQL
- Example: "Find all BJP rallies in Mumbai after Jan 2024"
  - SQL filter: `party='BJP' AND city='Mumbai' AND date > '2024-01-01'`
  - Vector search: semantic similarity on rally transcripts

#### C. Multi-Hop Retrieval
- For complex analytical queries
- Iteratively retrieves supporting evidence
- Builds knowledge graph of related facts

**Output**: Ranked list of relevant documents with metadata and confidence scores

---

### 4. Synthesis Agent

**Purpose**: Generate coherent, well-attributed responses

**Capabilities**:
- Context-aware response generation using retrieved documents
- Source attribution (cites original documents)
- Confidence scoring based on evidence quality
- Multi-format output (summary, bullet points, tables)
- Handles conflicting information gracefully

**Model Integration**:
- Uses Model Adapter interface for flexibility
- Supports OpenRouter (100+ models), local models (Ollama, vLLM)
- Configurable temperature, max tokens, streaming support

**Response Structure**:
```json
{
  "response": "Natural language answer",
  "confidence_score": 0.87,
  "sources": [
    {"doc_id": "xyz", "source": "Election Commission", "relevance": 0.92}
  ],
  "reasoning_steps": ["Step 1", "Step 2"],
  "follow_up_suggestions": ["Related query 1", "Related query 2"]
}
```

---

### 5. Validation Agent

**Purpose**: Quality assurance before response delivery

**Validation Checks**:
- **Output Validation**: Ensures response answers the original query
- **Hallucination Detection**: Flags unsupported claims
- **Source Attribution**: Verifies all factual claims have citations
- **Confidence Threshold**: Blocks low-confidence responses (<0.6 default)
- **Political Domain Rules**: 
  - Sensitive constituency handling
  - Demographic data restrictions
  - Defamatory content prevention
- **Safety Compliance**: No biased or inflammatory language

**Actions**:
- ✅ Pass → Forward to Memory Agent
- ⚠️ Low confidence → Add disclaimer
- ❌ Fail → Trigger retry or return "I don't know"

---

### 6. Memory Agent

**Purpose**: Manage short-term and long-term context

#### Short-Term Memory (Session State)
- Stored in Redis with TTL (default 30 minutes)
- Tracks conversation history within session
- Maintains entity references ("that party" → resolved to "BJP")
- Enables follow-up questions without repetition

#### Long-Term Memory (Vector Store)
- Persistent storage in PostgreSQL + pgvector
- Conversation embeddings for semantic retrieval
- Cross-session learning (user preferences, common queries)
- Privacy-compliant (PII redacted before storage)

**Memory Operations**:
- `store_conversation(session_id, messages)`
- `retrieve_context(session_id, query)`
- `get_session_history(session_id, limit=10)`
- `clear_session(session_id)`

---

### 7. Tool Execution Agent (MCP)

**Purpose**: Execute external tool calls safely

**Available Tools**:

| Tool | Function | Parameters |
|------|----------|------------|
| `database_query` | Execute SQL with filters | table, filters, aggregations |
| `vector_search` | Semantic search | query, k, filters |
| `external_api` | Fetch real-time data | api_name, endpoint, params |
| `conversation_memory` | Memory operations | operation, session_id, data |

**Safety Features**:
- Action confirmation for high-risk operations
- Rate limiting per session/tool
- Rollback capability for failed transactions
- Audit logging for all tool executions

---

## Key Components

### Parameter-Aware Hybrid Cache (Redis)

**Location**: `/cache/hybrid_cache.py`

**Features**:
- **Parameter Hashing**: Creates unique cache keys from query + parameters
- **Hybrid Strategy**: Caches both exact matches and similar queries
- **TTL Management**: Automatic expiration (configurable per query type)
- **Invalidation**: Cache busting on data updates
- **Metrics**: Hit/miss rates, latency tracking

**Usage**:
```python
from cache.hybrid_cache import HybridCache

cache = HybridCache()
result = cache.get_or_set(
    key_params={"query": "election results", "constituency": "Delhi"},
    compute_fn=lambda: agent.query(...),
    ttl=3600
)
```

---

### Guardrails System

**Location**: `/guardrails/guardrails.py`

**Components**:

1. **InputValidator**: Length limits, harmful content, PII, SQL injection
2. **OutputValidator**: Confidence thresholds, hallucination checks, attribution
3. **ActionConfirmator**: Risk-based confirmation (low/medium/high/critical)
4. **RetryHandler**: Exponential backoff with jitter
5. **RollbackManager**: Compensation actions for failures
6. **RateLimiter**: Per-session rate limiting by action type
7. **PoliticalDomainGuardrails**: Domain-specific safety rules

**Configuration**:
```python
GUARDRAILS_ENABLED = true
GUARDRAILS_INPUT_MAX_LENGTH = 2000
GUARDRAILS_OUTPUT_MIN_CONFIDENCE = 0.6
GUARDRAILS_RATE_LIMIT_QUERIES_PER_MINUTE = 30
GUARDRAILS_REQUIRE_SOURCE_ATTRIBUTION = true
```

---

### Observability Stack

**Location**: `/observability/monitoring.py`

**Integrations**:

1. **Structured Logging**: Loguru with JSON output
2. **Langfuse Tracing**: Distributed tracing for agent workflows
3. **LangSmith Integration**: Run tracking and debugging
4. **Metrics Collection**: Prometheus-compatible metrics
5. **Behavior Monitoring**: Response times, confidence scores, error rates
6. **Grafana Dashboards**: Pre-built visualization templates

**Tracing Example**:
```python
from observability.monitoring import ObservabilityManager

obs = ObservabilityManager()
with obs.trace("query_execution", user_id="user123") as span:
    result = agent.query(question)
    span.set_attribute("confidence", result['confidence_score'])
```

---

### Evaluation Harness

**Location**: `/evaluation/harness.py`

**Features**:
- **Golden Datasets**: Curated test cases with expected answers
- **LLM-as-Judge**: Automated scoring on multiple criteria
- **Regression Testing**: Compare versions for performance drops
- **Category Breakdown**: Factual, comparative, trend, analytical, edge cases

**Scoring Criteria**:
- Relevance (0-5)
- Accuracy (0-5)
- Completeness (0-5)
- Clarity (0-5)
- Source Attribution (0-5)
- Safety Compliance (pass/fail)

**Run Evaluation**:
```python
from evaluation.harness import run_quick_evaluation

report = run_quick_evaluation()
print(f"Overall Score: {report.overall_score}")
print(f"Regressions: {report.regressions}")
```

---

### Model Adapters

**Location**: `/models/adapters.py`

**Supported Providers**:
- **OpenRouter**: 100+ models (GPT-4, Claude, Llama, Mistral, etc.)
- **Local Models**: Ollama, vLLM, TGI
- **Custom**: Implement ModelAdapter ABC for new providers

**Adapter Interface**:
```python
from models.adapters import get_model_adapter

adapter = get_model_adapter(provider="openrouter", model="meta-llama/llama-3-70b-instruct")
response = adapter.generate("Your prompt here")
```

**Configuration**:
```python
LLM_PROVIDER = "openrouter"
OPENROUTER_API_KEY = "your-key"
OPENROUTER_MODEL = "meta-llama/llama-3-70b-instruct"
```

---

## Project Structure

```
querybot/
├── agents/
│   └── langgraph_agent.py          # LangGraph state machine & agent nodes
├── api/
│   └── main.py                     # FastAPI REST endpoints
├── cache/
│   └── hybrid_cache.py             # Parameter-aware Redis cache
├── config/
│   └── settings.py                 # Configuration management
├── database/
│   └── models.py                   # SQLAlchemy models + pgvector
├── data_ingestion/
│   ├── ingestion.py                # Data pipeline
│   └── chunking.py                 # Document chunking strategies
├── embeddings/
│   └── embedder.py                 # Embedding service
├── evaluation/
│   └── harness.py                  # Evaluation harness & LLM judge
├── guardrails/
│   └── guardrails.py               # Input/output validation & safety
├── models/
│   └── adapters.py                 # Model provider adapters
├── observability/
│   └── monitoring.py               # Logging, tracing, metrics
├── tools/
│   └── mcp_tools.py                # MCP tool implementations
├── requirements.txt
├── README.md
└── .env.example
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL 14+ with pgvector extension
- Redis 6+
- pip or poetry

### Installation

1. **Clone and setup virtual environment**
```bash
cd querybot
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Setup PostgreSQL with pgvector**
```sql
CREATE DATABASE querybot;
\c querybot
CREATE EXTENSION IF NOT EXISTS vector;
```

4. **Setup Redis**
```bash
# Install Redis
sudo apt install redis-server  # Ubuntu/Debian
# or
brew install redis  # macOS

# Start Redis
redis-server --daemonize yes

# Verify
redis-cli ping  # Should return PONG
```

5. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your configuration
```

6. **Run the application**
```bash
# Start the API server
python -m api.main

# Or with uvicorn directly
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

7. **Access the API**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## Usage Examples

### Query via API

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What were the election results in Delhi constituency?",
    "include_sources": true
  }'
```

### Chat with Context

```bash
# First message
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "my-session-123",
    "message": "Tell me about BJP performance in Maharashtra"
  }'

# Follow-up (maintains context)
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "my-session-123",
    "message": "How does that compare to Congress?"
  }'
```

### Python SDK-style Usage

```python
from agents.langgraph_agent import get_agent
from cache.hybrid_cache import HybridCache

# Initialize cache
cache = HybridCache()

# Get agent with caching
agent = get_agent()

# Single query with caching
result = cache.get_or_set(
    key_params={"query": "manifesto promises"},
    compute_fn=lambda: agent.query("What are the key manifesto promises?"),
    ttl=3600
)
print(result['response'])

# Chat with context
session_id = "session-123"
response1 = agent.chat("Analyze voter sentiment in urban areas", session_id)
response2 = agent.chat("What about rural areas?", session_id)
```

### Run Evaluation

```python
from evaluation.harness import run_quick_evaluation, create_golden_dataset

# Create golden dataset
dataset = create_golden_dataset()

# Run evaluation
report = run_quick_evaluation(dataset)
print(f"Overall Score: {report.overall_score}/5.0")
print(f"Category Breakdown: {report.category_scores}")
```

---

## MCP (Model Context Protocol) Architecture

### Overview
MCP standardizes how agents interact with external data sources and tools, transforming hardcoded pipeline steps into **autonomous tool-calling capabilities**.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Agent (LangGraph)                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Query      │  │  Retrieval   │  │  Synthesis   │          │
│  │   Analysis   │──│    Agent     │──│    Agent     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         │                  │                                    │
│         └──────────────────┼────────────────────────────────────┤
│                            │ MCP Protocol Layer                 │
└────────────────────────────┼────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   Database    │   │   Vector      │   │  External     │
│     Tool      │   │  Retriever    │   │    API        │
│               │   │     Tool      │   │    Tool       │
│ • SQL queries │   │ • Embedding   │   │ • News API    │
│ • Filtering   │   │ • Similarity  │   │ • Twitter API │
│ • Aggregation │   │ • Re-ranking  │   │ • Reddit API  │
└───────────────┘   └───────────────┘   └───────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│              PostgreSQL + pgvector + External APIs              │
└─────────────────────────────────────────────────────────────────┘
```

### How MCP Enables Multi-Source Data Access

#### 1. **Standardized Tool Interface**
All data sources implement the same `ToolResult` interface:

```python
class ToolResult(BaseModel):
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any]
```

This means the Retrieval Agent can call **any tool** without knowing implementation details.

#### 2. **Tool Abstraction Example**

**Without MCP (Hardcoded):**
```python
# Agent needs to know SQL, vector math, API endpoints
def get_data(query, constituency):
    # SQL query
    results = session.query(Document).filter(
        Document.constituency == constituency
    ).all()
    
    # Vector search
    embedding = embed(query)
    similar = cosine_similarity(embedding, all_embeddings)
    
    # API call
    response = requests.get(f"https://newsapi.com?q={query}")
    
    return combine(results, similar, response.json())
```

**With MCP (Agent-Friendly):**
```python
# Agent just calls tools declaratively
vector_tool = get_tool("vector_search")
db_tool = get_tool("database_query")
api_tool = get_tool("external_api")

# All return same ToolResult format
vector_result = vector_tool.execute(query=query, filters={"constituency": "Patna"})
db_result = db_tool.execute(query_type="elections", filters={"constituency": "Patna"})
news_result = api_tool.execute(api_type="news", query=query, location="Bihar")

# Agent processes uniform results
all_data = [r.data for r in [vector_result, db_result, news_result] if r.success]
```

#### 3. **Available MCP Tools**

| Tool Name | Purpose | Data Source | Parameters |
|-----------|---------|-------------|------------|
| `database_query` | Structured SQL queries | PostgreSQL | query_type, filters, limit |
| `vector_search` | Semantic similarity search | pgvector | query, filters, min_similarity |
| `external_api` | Real-time data fetch | News/Twitter/Reddit APIs | api_type, query, location |
| `conversation_memory` | Session history | PostgreSQL | action, session_id |

#### 4. **Retrieval Agent Workflow with MCP**

```
Step 1: Query Analysis
└─> Determines: intent="comparative", entities={party: "BJP", state: "Bihar"}

Step 2: Tool Selection (Autonomous)
├─> Intent is "comparative" → Need structured data + semantic context
├─> Select tools: [vector_search, database_query]
└─> Skip external_api (not recent query)

Step 3: Parallel Tool Execution
├─> vector_search.execute(query="BJP Bihar manifesto", filters={party: "BJP"})
│   └─> Returns: 10 similar documents with similarity scores
│
└─> database_query.execute(query_type="manifestos", filters={party: "BJP", state: "Bihar"})
    └─> Returns: 8 structured manifesto records

Step 4: Result Aggregation
├─> Combine vector results (semantic relevance)
├─> Merge with DB results (structured metadata)
└─> Deduplicate by document ID

Step 5: Pass to Synthesis Agent
└─> Retrieved documents: 15 unique docs
    Tool results: 2 successful executions
    Reasoning: ["Vector search returned 10 docs", "DB query returned 8 records"]
```

### Benefits of MCP Architecture

| Benefit | Description |
|---------|-------------|
| **Modularity** | Add new data sources by implementing tool interface (no agent changes) |
| **Autonomy** | Agents choose which tools to call based on query analysis |
| **Testability** | Mock individual tools without affecting entire pipeline |
| **Observability** | Track which tools were called, success rates, latency per tool |
| **Fallback Logic** | If one tool fails, agents can try alternative tools |
| **Rate Limiting** | Apply rate limits per tool type (guardrails integration) |

### Adding a New Data Source (Example: YouTube Transcripts)

```python
class YouTubeTranscriptTool:
    name = "youtube_transcript"
    description = "Search and retrieve YouTube video transcripts about political events"
    
    def execute(self, query: str, channel: str = None) -> ToolResult:
        try:
            # Implementation details hidden from agent
            transcripts = search_youtube(query, channel)
            return ToolResult(success=True, data=transcripts)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

# Register tool
register_tool("youtube_transcript", YouTubeTranscriptTool())

# Agent can now use it automatically!
```

---

## Query Flow Examples with MCP Integration

### Example 1: Factual Query Flow

**User Query**: *"What was the voter turnout in Patna during the 2020 election?"*

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 1: User submits query via API                                      │
│ POST /query { "question": "What was the voter turnout in Patna..."}    │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 2: Cache Check (HybridCache)                                       │
│ Key: hash(query="voter turnout Patna 2020", params={})                  │
│ Result: ❌ CACHE MISS (first time query)                                 │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 3: Query Analysis Agent                                            │
│ • Parse: "voter turnout" → metric, "Patna" → constituency,             │
│          "2020" → year                                                  │
│ • Entities: {constituency: "Patna", year: 2020, metric: "turnout"}     │
│ • Guardrails: ✅ Pass (length OK, no harmful content, no PII)          │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 4: Intent Planning Agent (Classification + Planning)               │
│ • Classified as: FACTUAL                                                │
│ • Complexity: SIMPLE                                                    │
│ • Execution Plan:                                                       │
│   - Sub-queries: ["voter turnout Patna 2020"]                          │
│   - Tools: [vector_search, sql_query]                                  │
│   - Strategy: Direct retrieval (single step)                           │
│   - Parallel groups: [] (sequential only)                              │
│   - Confidence threshold: 0.75                                         │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 5: Retrieval Agent                                                 │
│ • Vector Search: pgvector similarity on election documents             │
│   Query: "voter turnout Patna 2020 election"                           │
│   Filters: constituency='Patna', year=2020                             │
│   Results: Top 5 chunks with relevance scores                          │
│ • SQL Query: SELECT turnout FROM election_results                      │
│              WHERE constituency='Patna' AND year=2020                   │
│   Result: 68.4%                                                         │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 6: Synthesis Agent                                                 │
│ • Generates: "The voter turnout in Patna during the 2020 election      │
│             was 68.4%, according to Election Commission data."         │
│ • Confidence: 0.94 (high - exact match found)                           │
│ • Sources: [{doc_id: "ec_2020_patna", source: "Election Commission"}]  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 7: Validation Agent                                                │
│ • Output validation: ✅ Answers the query                               │
│ • Hallucination check: ✅ Claim supported by source                     │
│ • Confidence threshold: ✅ 0.94 > 0.6                                   │
│ • Source attribution: ✅ Present                                        │
│ • Political guardrails: ✅ No sensitive issues                          │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 8: Memory Agent                                                    │
│ • Short-term: Store in Redis (session context, TTL=30min)              │
│ • Long-term: Embed conversation in pgvector for future retrieval       │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 9: Cache Storage                                                   │
│ • Store response in Redis with key hash                                │
│ • TTL: 3600 seconds (1 hour) for factual queries                        │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 10: Response to User                                               │
│ {                                                                       │
│   "response": "The voter turnout in Patna during the 2020 election     │
│               was 68.4%...",                                            │
│   "confidence_score": 0.94,                                             │
│   "sources": [...],                                                     │
│   "cached": false                                                       │
│ }                                                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

**If Same Query Asked Again (within 1 hour)**:
- Step 2 returns ✅ **CACHE HIT**
- Steps 3-9 are **SKIPPED**
- Response served directly from Redis in <10ms
- `cached: true` flag added to response

---

### Example 2: Complex Analytical Query Flow

**User Query**: *"Compare the manifesto promises of Party A vs Party B regarding agriculture in Bihar and analyze their fulfillment status."*

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 1: User submits complex analytical query                           │
│ POST /chat { "session_id": "sess-123",                                  │
│              "message": "Compare manifesto promises..." }               │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 2: Cache Check                                                     │
│ Key: hash(query="compare manifesto Party A Party B agriculture Bihar", │
│       params={session_id: "sess-123"})                                  │
│ Result: ❌ CACHE MISS (complex query, never asked before)               │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 3: Query Analysis Agent                                            │
│ • Parse: Multiple entities detected                                     │
│   - Parties: ["Party A", "Party B"]                                    │
│   - Topic: "agriculture"                                                │
│   - Location: "Bihar"                                                   │
│   - Task: "compare" + "analyze fulfillment"                            │
│ • Entities: {parties: ["Party A", "Party B"], topic: "agriculture",    │
│              state: "Bihar"}                                            │
│ • Guardrails: ✅ Pass                                                   │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 4: Intent Planning Agent (Classification + Planning)               │
│ • Classified as: ANALYTICAL + COMPARATIVE                               │
│ • Complexity: COMPLEX                                                   │
│ • Execution Plan:                                                       │
│   - Sub-queries: [                                                      │
│       "Party A agriculture promises Bihar",                            │
│       "Party B agriculture promises Bihar",                            │
│       "Party A agriculture fulfillment status",                        │
│       "Party B agriculture fulfillment status"                         │
│     ]                                                                   │
│   - Tools: [vector_search, sql_query, external_api]                    │
│   - Strategy: Multi-hop retrieval with parallel execution              │
│   - Parallel groups: [["sub_query_1", "sub_query_2"],                  │
│                       ["sub_query_3", "sub_query_4"]]                  │
│   - Sequential steps: ["synthesize_comparison", "analyze_fulfillment"] │
│   - Dependencies: {                                                     │
│       "synthesize_comparison": ["sub_query_1", "sub_query_2"],         │
│       "analyze_fulfillment": ["sub_query_3", "sub_query_4",            │
│                               "synthesize_comparison"]                 │
│     }                                                                   │
│   - Confidence threshold: 0.70 (lower due to complexity)               │
│   - Fallback: Return partial results if some data unavailable          │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 5: Retrieval Agent (Parallel Execution)                            │
│ ┌─────────────────────────────────┐  ┌────────────────────────────────┐ │
│ │ Query 1: Party A Manifesto      │  │ Query 2: Party B Manifesto     │ │
│ │ • Vector: "Party A agriculture  │  │ • Vector: "Party B agriculture │ │
│ │   promises Bihar"               │  │   promises Bihar"              │ │
│ │ • SQL: party='Party A' AND      │  │ • SQL: party='Party B' AND     │ │
│ │   topic='agriculture'           │  │   topic='agriculture'          │ │
│ │ Results: 8 chunks               │  │ Results: 7 chunks              │ │
│ └─────────────────────────────────┘  └────────────────────────────────┘ │
│                                                                          │
│ ┌─────────────────────────────────┐  ┌────────────────────────────────┐ │
│ │ Query 3: Fulfillment Data       │  │ Query 4: Implementation Stats  │ │
│ │ • Tool: database_query()        │  │ • Tool: external_api()         │ │
│ │ • SQL: SELECT * FROM            │  │ • News API: "Party A Bihar     │ │
│ │   policy_fulfillment WHERE      │  │   agriculture schemes"         │ │
│ │   party IN ('A','B')            │  │ • Social listening sentiment   │ │
│ │ Results: 15 records             │  │ Results: 12 articles           │ │
│ └─────────────────────────────────┘  └────────────────────────────────┘ │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 6: Tool Execution Agent (MCP)                                      │
│ • Executes database_query tool with filters                             │
│ • Calls external_api for real-time news                                 │
│ • Rate limit check: ✅ Within quota (5/30 per minute)                   │
│ • Action confirmation: Not needed (medium risk)                         │
│ • Audit log: All tool calls recorded                                    │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 7: Synthesis Agent                                                 │
│ • Multi-step reasoning:                                                 │
│   1. Extract Party A promises (3 key points)                           │
│   2. Extract Party B promises (3 key points)                           │
│   3. Compare side-by-side in table format                              │
│   4. Assess fulfillment with evidence                                  │
│ • Generates structured response with:                                   │
│   - Comparison table                                                    │
│   - Fulfillment analysis                                                │
│   - Supporting evidence citations                                       │
│ • Confidence: 0.78 (moderate - some data gaps)                          │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 8: Validation Agent                                                │
│ • Output validation: ✅ Comprehensive comparison provided               │
│ • Hallucination check: ⚠️ One claim lacks direct source                │
│   → Action: Add disclaimer "Based on available data..."                │
│ • Confidence threshold: ✅ 0.78 > 0.6                                   │
│ • Source attribution: ✅ All major claims cited                         │
│ • Political guardrails: ✅ Balanced presentation, no bias detected      │
│ • Safety compliance: ✅ No inflammatory language                        │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 9: Memory Agent                                                    │
│ • Short-term (Redis):                                                   │
│   - Store full conversation turn                                        │
│   - Track entities: Party A, Party B, Bihar, agriculture               │
│   - Enable follow-up: "What about education?"                          │
│ • Long-term (pgvector):                                                 │
│   - Embed conversation for cross-session learning                       │
│   - Link to user profile (anonymized)                                   │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 10: Observability Tracing                                          │
│ • Langfuse: Complete trace with all spans                              │
│   - Query Analysis: 45ms                                                │
│   - Retrieval (parallel): 320ms                                         │
│   - Tool Execution: 180ms                                               │
│   - Synthesis: 890ms                                                    │
│   - Validation: 65ms                                                    │
│   Total latency: 1.5s                                                   │
│ • Metrics:                                                              │
│   - Counter: analytical_queries_total++                                 │
│   - Histogram: query_duration_seconds=1.5                               │
│   - Gauge: active_sessions=47                                           │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 11: Cache Storage                                                  │
│ • Store in Redis with longer TTL (2 hours for analytical)              │
│ • Key includes session context for personalization                     │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 12: Response to User                                               │
│ {                                                                       │
│   "response": "## Comparison of Agriculture Manifesto Promises...      │
│               | Promise Area | Party A | Party B | ...                 │
│               ## Fulfillment Analysis...[detailed response]",           │
│   "confidence_score": 0.78,                                             │
│   "sources": [12 citations],                                            │
│   "reasoning_steps": ["Extracted promises", "Compared policies", ...], │
│   "follow_up_suggestions": [                                            │
│     "How do these compare at national level?",                          │
│     "What were the key education promises?"                             │
│   ],                                                                    │
│   "cached": false,                                                      │
│   "latency_ms": 1500                                                    │
│ }                                                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Differences from Example 1**:
- Multi-hop retrieval (4 parallel queries vs 1)
- Tool execution required (database + external APIs)
- Complex reasoning with comparison table
- Lower confidence due to data complexity
- Longer cache TTL for expensive computation
- Full observability tracing for debugging

---

## Agent State Flow

```
┌─────────┐    ┌──────────┐    ┌───────────┐    ┌────────────┐
│  User   │ →  │  Query   │ →  │  Intent   │ →  │ Retrieval  │
│  Query  │    │ Analysis │    │Classifier │    │   Agent    │
└─────────┘    └──────────┘    └───────────┘    └────────────┘
                                                   │
                                                   ▼
┌─────────┐    ┌──────────┐    ┌───────────┐    ┌────────────┐
│Response │ ←  │ Memory   │ ←  │Validation │ ←  │ Synthesis  │
│  +      │    │  Agent   │    │  Agent    │    │   Agent    │
│Sources  │    └──────────┘    └───────────┘    └────────────┘
└─────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│         Guardrails at Each Stage        │
│  • Input validation                     │
│  • Rate limiting                        │
│  • Action confirmation (if needed)      │
│  • Output validation                    │
│  • Retry/rollback                       │
└─────────────────────────────────────────┘
```

---

## MCP Tools

Available tools exposed via Model Context Protocol:

| Tool | Description | Use Case |
|------|-------------|----------|
| `database_query` | Structured SQL queries with filters | Election results, survey aggregates |
| `vector_search` | Semantic similarity search | Document retrieval, manifesto analysis |
| `external_api` | Real-time news/social data | Current events, sentiment tracking |
| `conversation_memory` | Session-based memory management | Multi-turn conversations |

---

## Data Ingestion

```python
from data_ingestion.ingestion import DataIngestionPipeline

pipeline = DataIngestionPipeline()
count = pipeline.ingest_from_directory("data/elections", "election")
pipeline.close()
```

Supported formats: JSON, CSV, TXT, PDF

---

## Configuration

Key environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://...` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379` |
| `PGVECTOR_DIMENSION` | Embedding dimension | `768` |
| `EMBEDDING_MODEL` | Sentence transformer model | `all-MiniLM-L6-v2` |
| `LLM_PROVIDER` | Model provider | `openrouter` |
| `OPENROUTER_API_KEY` | OpenRouter API key | (required) |
| `OPENROUTER_MODEL` | Default model | `meta-llama/llama-3-70b-instruct` |
| `CACHE_TTL_DEFAULT` | Default cache TTL (seconds) | `3600` |
| `GUARDRAILS_ENABLED` | Enable guardrails | `true` |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key | (optional) |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key | (optional) |
| `API_PORT` | Server port | `8000` |

---

## Deployment on AWS EC2

1. **Launch EC2 instance** (Ubuntu 22.04, t3.medium+)
2. **Install dependencies**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib python3-pip redis-server
```
3. **Setup pgvector**
```bash
sudo apt install postgresql-server-dev-all
cd /tmp
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```
4. **Configure Redis**
```bash
sudo systemctl enable redis-server
sudo systemctl start redis-server
```
5. **Deploy application**
```bash
git clone <repo>
pip install -r requirements.txt
```
6. **Run with systemd**
```ini
[Unit]
Description=QueryBot API
After=network.target redis.service postgresql.service

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/querybot
ExecStart=/home/ubuntu/querybot/venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## Performance Optimization

- **Caching**: Parameter-aware Redis cache reduces redundant computations
- **Batch Embedding**: Generate embeddings in batches for bulk ingestion
- **Connection Pooling**: Configure PostgreSQL and Redis connection pools
- **Async Operations**: Use async database and API calls
- **Load Balancing**: Deploy behind load balancer for horizontal scaling
- **Index Tuning**: Optimize pgvector indexes (HNSW, IVFFlat)

---

## Monitoring & Observability

### Metrics to Track
- Query latency (p50, p95, p99)
- Cache hit/miss rates
- Agent node execution times
- Confidence score distributions
- Error rates by category
- Token usage per request

### Dashboards
- Grafana dashboards exported via `ObservabilityManager`
- Langfuse UI for trace visualization
- LangSmith for run debugging

### Alerts
- High error rates (>5%)
- Low confidence responses (<0.5 average)
- Cache miss spikes
- Rate limit violations

---

## Evaluation & Testing

### Before Deployment Checklist
- [ ] Run full evaluation suite on golden dataset
- [ ] Check for regressions vs previous version
- [ ] Verify guardrail compliance on edge cases
- [ ] Load test with realistic query patterns
- [ ] Review trace samples in Langfuse/LangSmith

### Continuous Evaluation
```python
from evaluation.harness import RegressionTester

tester = RegressionTester()
regressions = tester.compare_versions("v1.2.0", "v1.3.0")
if regressions:
    print(f"Found {len(regressions)} regressions!")
    for r in regressions:
        print(f"  - {r.test_case}: {r.old_score} → {r.new_score}")
```

---

## Security & Compliance

### Data Protection
- PII detection and redaction in inputs/outputs
- Encrypted connections (PostgreSQL, Redis)
- Session isolation for multi-tenant deployments
- Audit logging for all data access

### Political Domain Safeguards
- Sensitive constituency handling (requires confirmation)
- Demographic data restrictions (aggregated only)
- Defamatory content prevention
- Bias detection in responses

### Rate Limiting
- Per-session query limits
- Per-tool execution limits
- Burst protection with exponential backoff

---

## License

MIT License

---

## Contact

For questions and support, contact the development team.

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests using the evaluation harness
4. Ensure all guardrails pass
5. Submit a pull request with trace logs
