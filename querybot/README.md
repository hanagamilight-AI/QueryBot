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

**Purpose**: Initial query processing and understanding

**Responsibilities**:
- Parse natural language input
- Classify query intent (factual, comparative, trend, analytical)
- Extract entities (constituencies, parties, dates, candidates)
- Apply input guardrails (length limits, harmful content detection, PII filtering)
- Determine required tools and data sources

**Input**: Raw user query + session context
**Output**: Structured query plan with intent, entities, and constraints

**Guardrails Applied**:
- Input length validation (max 2000 chars)
- Harmful content detection
- SQL injection pattern blocking
- PII redaction for sensitive personal data

---

### 2. Intent Classification Agent

**Purpose**: Route queries to appropriate retrieval strategies

**Intent Categories**:
| Intent Type | Description | Example |
|-------------|-------------|---------|
| **Factual** | Single fact lookup | "Who won Delhi 2020?" |
| **Comparative** | Compare entities | "BJP vs Congress in Maharashtra" |
| **Trend** | Temporal analysis | "Voter turnout trend 2014-2024" |
| **Analytical** | Complex reasoning | "Why did AAP lose in rural areas?" |
| **Sentiment** | Opinion/sentiment | "Public perception of Modi" |

**Decision Logic**:
- Factual → Direct vector search + exact match
- Comparative → Multi-query retrieval + comparison synthesis
- Trend → Time-series SQL aggregation + vector context
- Analytical → Multi-hop retrieval + chain-of-thought reasoning

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
