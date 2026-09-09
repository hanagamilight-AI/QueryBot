# QueryBot — Political Intelligence Conversational AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-latest-orange.svg)](https://langchain-ai.github.io/langgraph/)

**QueryBot** is an advanced multi-agent conversational AI system designed for political intelligence analysis. Built with LangGraph, MCP (Model Context Protocol), and PostgreSQL + pgvector, it enables autonomous retrieval, analysis, and synthesis of political data including election results, surveys, manifestos, and social media sentiment.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                    │
│                    (Web Dashboard / API Consumers)                          │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │ REST API
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            API GATEWAY                                      │
│                  (FastAPI + Guardrails + Rate Limiting)                     │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       AGENT ORCHESTRATION LAYER                             │
│                           (LangGraph Engine)                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    STATE GRAPH WORKFLOW                               │   │
│  │                                                                       │   │
│  │  ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐        │   │
│  │  │   Query      │───▶│     Intent       │───▶│  Retrieval   │        │   │
│  │  │  Analysis    │    │  Classification  │    │    Agent     │        │   │
│  │  │   Agent      │    │     Agent        │    │              │        │   │
│  │  └──────────────┘    └──────────────────┘    └──────┬───────┘        │   │
│  │         │                                           │                 │   │
│  │         │              ┌──────────────────┐         │                 │   │
│  │         │              │   Validation     │◀────────┘                 │   │
│  │         │              │     Agent        │                           │   │
│  │         │              └────────┬─────────┘                           │   │
│  │         │                       │                                     │   │
│  │         │              ┌────────▼─────────┐                           │   │
│  │         │              │   Synthesis      │                           │   │
│  │         │              │     Agent        │                           │   │
│  │         │              └────────┬─────────┘                           │   │
│  │         │                       │                                     │   │
│  │         │              ┌────────▼─────────┐                           │   │
│  │         └─────────────▶│     Memory       │                           │   │
│  │                        │     Agent        │                           │   │
│  │                        └──────────────────┘                           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │ MCP Tools
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TOOL EXECUTION LAYER                               │
│                    (MCP Protocol + Action Handlers)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Vector     │  │  Database    │  │  External    │  │  Conversation│    │
│  │   Search     │  │   Query      │  │    APIs      │  │   Memory     │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DATA LAYER                                       │
│  ┌─────────────────────────────────┐    ┌────────────────────────────────┐  │
│  │   PostgreSQL + pgvector         │    │        Redis Cache             │  │
│  │   - Election Data               │    │   - Response Cache             │  │
│  │   - Survey Results              │    │   - Session State              │  │
│  │   - Manifesto Documents         │    │   - Rate Limiting              │  │
│  │   - Document Embeddings         │    │                                │  │
│  └─────────────────────────────────┘    └────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                      CROSS-CUTTING CONCERNS                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Guardrails │  │Observability │  │  Evaluation  │  │ Model Adapter│    │
│  │   (Input/    │  │ (LangFuse/   │  │  Harness +   │  │ (OpenRouter/ │    │
│  │   Output     │  │  LangSmith)  │  │ Golden Tests │  │  Local)      │    │
│  │   Validation)│  │              │  │              │  │              │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Agent Architecture

QueryBot implements a **multi-agent system** where each agent has a specialized role. Agents communicate through a shared state managed by LangGraph, ensuring coherent workflows and context preservation.

### 1. **Query Analysis Agent** 📊

**Role:** Parse and enrich user queries without making decisions about intent.

**Responsibilities:**
- Extract entities (constituencies, parties, politicians, dates, states)
- Detect query language
- Resolve pronouns using conversation history
- Identify ambiguity and request clarification when needed

**Input:** Raw user query + conversation history  
**Output:** Structured entities, language, ambiguity flag

**Example:**
```
Input: "What was the turnout in Patna?"
Output: {
  entities: { constituency: "Patna", state: "Bihar" },
  language: "en",
  is_ambiguous: false
}
```

---

### 2. **Intent Classification Agent** 🎯

**Role:** Determine the type of query and required processing strategy.

**Responsibilities:**
- Classify intent: `factual`, `comparative`, `trend`, `analytical`, `predictive`, `summary`
- Assess query complexity (low/medium/high)
- Determine required tools (vector_search, database_query, external_api, statistical_analysis)
- Provide confidence score for classification

**Input:** Enriched query from Query Analysis Agent  
**Output:** Intent type, complexity, required tools, confidence

**Example:**
```
Input: "Compare BJP vs Congress performance in Delhi 2020"
Output: {
  intent: "comparative",
  complexity: "medium",
  required_tools: ["vector_search", "database_query"],
  confidence: 0.9
}
```

---

### 3. **Retrieval Agent** 🔍

**Role:** Fetch relevant data from multiple sources using MCP tools.

**Responsibilities:**
- Check parameter-aware hybrid cache (Redis + pgvector similarity)
- Execute vector similarity search via pgvector
- Run structured database queries
- Call external APIs for recent events
- Aggregate and deduplicate results

**Input:** Query intent + entities + timeframe  
**Output:** Retrieved documents, tool execution results, cache metadata

**Cache Strategy:**
- **Exact Match:** Same query + same parameters → Return cached response
- **Semantic Match:** Similar query (cosine similarity > threshold) → Return cached response with similarity score
- **Parameter Match:** Same entities/timeframe → Return partial cached data
- **Miss:** Proceed with full retrieval

---

### 4. **Synthesis Agent** ✍️

**Role:** Generate human-readable responses with proper source attribution.

**Responsibilities:**
- Synthesize retrieved information into coherent answers
- Cite sources (document type, constituency, date)
- Adjust tone based on query type (factual vs analytical)
- Include confidence indicators

**Input:** Retrieved documents + query intent  
**Output:** Formatted response with citations

**Example Output:**
```
Based on election data from the Election Commission [Database - Patna Constituency - 2020]:
The voter turnout in Patna during the 2020 Bihar Assembly elections was 55.2%.

This represents a 3.1% increase from the 2015 elections (52.1%) [Survey Report - Post-Poll Analysis].
```

---

### 5. **Validation Agent** ✅

**Role:** Ensure response quality, accuracy, and safety.

**Responsibilities:**
- Verify response relevance to original query
- Check for hallucination indicators
- Validate source attribution completeness
- Apply domain-specific guardrails (political sensitivity)
- Add disclaimers for low-confidence responses

**Input:** Draft response + retrieved documents + confidence score  
**Output:** Validated response (or flagged for revision)

**Guardrail Checks:**
- Input validation (length, harmful content, PII, SQL injection)
- Output validation (confidence threshold, hallucination detection)
- Action confirmation (for high-risk operations)
- Rate limiting (per session, per action type)
- Political domain rules (sensitive constituencies, demographic restrictions)

---

### 6. **Memory Agent** 🧠

**Role:** Manage short-term and long-term context.

**Responsibilities:**
- **Short-term:** Store conversation turns in-session (LangGraph checkpointer)
- **Long-term:** Persist conversation history in PostgreSQL
- Update vector store with new interaction patterns
- Retrieve relevant historical context for current query

**Storage:**
- **Session State:** In-memory via MemorySaver (temporary)
- **Conversation History:** PostgreSQL `conversation_memories` table
- **Semantic Memory:** pgvector embeddings of past interactions

---

### 7. **Tool Execution Agent** 🛠️

**Role:** Safely execute MCP tools with guardrails and observability.

**Responsibilities:**
- Validate tool inputs before execution
- Confirm high-risk actions (database writes, external API calls)
- Handle retries with exponential backoff
- Log tool usage for observability
- Rollback failed transactions

**Available MCP Tools:**
| Tool | Purpose | Risk Level |
|------|---------|------------|
| `vector_search` | Semantic document retrieval | Low |
| `database_query` | Structured SQL queries | Medium |
| `external_api` | News/social media APIs | Medium |
| `conversation_memory` | Store/retrieve history | Low |
| `statistical_analysis` | Trend calculations | Low |

---

## 🔄 Query Flow Examples

### Example 1: Factual Query Flow

**User Query:** *"What was the voter turnout in Patna during the 2020 election?"*

```
Step 1: User submits query
   │
   ▼
Step 2: API Gateway → Guardrails check (input validation, rate limiting)
   │
   ▼
Step 3: Query Analysis Agent
   ├─ Extracts: constituency="Patna", state="Bihar", year="2020"
   ├─ Language: "en"
   └─ Ambiguity: false
   │
   ▼
Step 4: Intent Classification Agent
   ├─ Intent: "factual"
   ├─ Complexity: "low"
   └─ Required tools: ["vector_search", "database_query"]
   │
   ▼
Step 5: Retrieval Agent
   ├─ Check Cache: MISS (no exact/semantic match)
   ├─ Vector Search: Finds 8 relevant documents (similarity > 0.85)
   ├─ Database Query: Retrieves election_data WHERE constituency='Patna' AND year=2020
   └─ Aggregates: 10 total records
   │
   ▼
Step 6: Synthesis Agent
   ├─ Generates response with citations
   └─ Adds source attribution
   │
   ▼
Step 7: Validation Agent
   ├─ Checks: Relevance ✓, Sources ✓, Confidence > 0.7 ✓
   └─ No modifications needed
   │
   ▼
Step 8: Memory Agent
   ├─ Stores query + response in conversation_history
   └─ Updates session state
   │
   ▼
Step 9: Cache Storage
   ├─ Stores response in Redis with query embedding
   └─ Metadata: {constituency: "Patna", year: "2020", intent: "factual"}
   │
   ▼
Step 10: Response returned to user
   "Based on Election Commission data [Database - Patna - 2020], 
    the voter turnout was 55.2%..."
```

---

### Example 2: Complex Analytical Query Flow

**User Query:** *"Compare the manifesto promises of Party A vs Party B regarding agriculture in Bihar and analyze their fulfillment status."*

```
Step 1: User submits complex analytical query
   │
   ▼
Step 2: Guardrails check (length OK, no harmful content)
   │
   ▼
Step 3: Query Analysis Agent
   ├─ Extracts: party_a="Party A", party_b="Party B", state="Bihar"
   ├─ Topic: "agriculture"
   └─ Ambiguity: false (entities clear)
   │
   ▼
Step 4: Intent Classification Agent
   ├─ Intent: "analytical" (contains "compare" + "analyze")
   ├─ Complexity: "high" (multi-party, multi-step reasoning)
   └─ Required tools: ["vector_search", "database_query", "external_api"]
   │
   ▼
Step 5: Retrieval Agent (Parallel Execution)
   ├─ Thread 1: Vector search for "Party A manifesto agriculture Bihar"
   ├─ Thread 2: Vector search for "Party B manifesto agriculture Bihar"
   ├─ Thread 3: Database query for manifesto_promises table
   └─ Thread 4: External API for recent fulfillment news
   │
   ▼
Step 6: Synthesis Agent (Multi-hop Reasoning)
   ├─ Retrieves Party A promises (3 items)
   ├─ Retrieves Party B promises (4 items)
   ├─ Cross-references with fulfillment data
   ├─ Identifies gaps between promises and outcomes
   └─ Structures comparison table
   │
   ▼
Step 7: Validation Agent
   ├─ Hallucination check: All claims traced to sources ✓
   ├─ Balance check: Both parties covered equally ✓
   ├─ Sensitivity check: No inflammatory language ✓
   └─ Adds disclaimer: "Analysis based on available data up to [date]"
   │
   ▼
Step 8: Memory Agent
   ├─ Stores full conversation turn
   └─ Tags: ["manifesto_analysis", "comparative", "agriculture"]
   │
   ▼
Step 9: Observability Tracking
   ├─ LangFuse: Logs trace with 4 parallel tool calls
   ├─ Metrics: Latency=2.3s, Token usage=1847
   └─ Behavior monitor: Flags high-complexity query for review
   │
   ▼
Step 10: Response returned with comparison table and citations
```

---

## 📁 Project Structure

```
querybot/
├── agents/
│   ├── langgraph_agent.py      # Main LangGraph workflow & node implementations
│   └── __init__.py
├── api/
│   ├── routes.py               # FastAPI endpoints
│   └── __init__.py
├── cache/
│   ├── hybrid_cache.py         # Parameter-aware Redis + pgvector cache
│   └── __init__.py
├── config/
│   └── settings.py             # Environment configuration
├── data_ingestion/
│   ├── ingestion.py            # Data collection pipelines
│   └── chunking.py             # Document chunking strategies
├── db/
│   ├── models.py               # SQLAlchemy models (DocumentChunk, etc.)
│   └── database.py             # Database connection
├── embeddings/
│   └── embedder.py             # Embedding generation
├── evaluation/
│   └── harness.py              # Test harness + LLM-as-judge
├── guardrails/
│   └── guardrails.py           # Input/output validation + safety
├── models/
│   └── adapters.py             # Model adapter interface (OpenRouter, local)
├── observability/
│   └── monitoring.py           # LangFuse/LangSmith integration
├── tools/
│   └── mcp_tools.py            # MCP tool definitions
├── examples/
│   └── cache_demo.py           # Cache usage examples
└── app.py                      # FastAPI application entry point
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL 14+ with pgvector extension
- Redis 7+
- OpenRouter API key (or local model setup)

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/querybot.git
cd querybot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your credentials

# Initialize database
python -m db.database init-db

# Start Redis (if not running)
redis-server --daemonize yes

# Run the application
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/querybot

# Redis
REDIS_URL=redis://localhost:6379/0

# Model Provider
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_key_here
DEFAULT_MODEL=meta-llama/llama-3-70b-instruct

# Cache
CACHE_ENABLED=true
CACHE_SIMILARITY_THRESHOLD=0.85
CACHE_TTL=3600

# Guardrails
GUARDRAILS_ENABLED=true
MAX_QUERY_LENGTH=1000
RATE_LIMIT_REQUESTS_PER_MINUTE=60

# Observability
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGCHAIN_API_KEY=ls__...
LOG_LEVEL=INFO
```

---

## 🔧 Key Components

### Parameter-Aware Hybrid Cache

The cache system combines **Redis** for fast lookups with **pgvector** for semantic similarity:

```python
from cache.hybrid_cache import get_cache, CacheNamespace

cache = get_cache(similarity_threshold=0.85)

# Store response
cache.set(
    query="What was the turnout in Patna?",
    value={"response": "...", "sources": [...]},
    parameters={"constituency": "Patna", "year": "2020"},
    namespace=CacheNamespace.QUERY_RESPONSE
)

# Retrieve with semantic matching
result = cache.get(
    query="Patna 2020 election participation",
    parameters={"constituency": "Patna"},
    query_vector=embedding  # Optional for semantic search
)

if result.hit_type == "semantic":
    print(f"Similar response found ({result.similarity_score:.2f})")
```

**Cache Hit Types:**
- `exact`: Identical query + parameters
- `semantic`: Similar query (cosine similarity > threshold)
- `parameter`: Matching parameters only
- `miss`: No match found

---

### Model Adapters

Flexible model provider interface supporting OpenRouter and local models:

```python
from models.adapters import get_model_adapter

# Get adapter for OpenRouter
adapter = get_model_adapter("openrouter", model_name="meta-llama/llama-3-70b-instruct")

# Generate response
response = await adapter.generate(
    messages=[{"role": "user", "content": "Analyze this election data..."}],
    temperature=0.7,
    max_tokens=1000
)

# Or use local model
local_adapter = get_model_adapter("local", base_url="http://localhost:11434")
```

**Supported Providers:**
- OpenRouter (100+ models)
- Ollama (local)
- vLLM (self-hosted)
- Custom OpenAI-compatible APIs

---

### Guardrails System

Comprehensive safety and validation layer:

```python
from guardrails.guardrails import GuardrailManager

guardrails = GuardrailManager()

# Validate input
validation = await guardrails.validate_input(
    query=user_query,
    session_id=session_id,
    max_length=1000
)

if not validation.is_valid:
    return {"error": validation.reason}

# Validate output
output_validation = await guardrails.validate_output(
    response=generated_response,
    sources=retrieved_docs,
    min_confidence=0.6
)

# Rate limit check
rate_check = guardrails.check_rate_limit(
    session_id=session_id,
    action_type="database_query"
)
```

**Political Domain Guardrails:**
- Restricted constituencies (sensitive regions)
- Demographic data access controls
- Manifesto comparison balance requirements
- Temporal data freshness rules

---

### Observability Stack

Full tracing and monitoring integration:

```python
from observability.monitoring import ObservabilityManager

obs = ObservabilityManager()

# Start trace
trace = obs.tracer.start_trace(
    name="process_query",
    session_id=session_id,
    metadata={"query_type": "comparative"}
)

# Log metrics
obs.metrics.increment("queries.processed")
obs.metrics.histogram("query.latency", duration_ms)

# Export dashboard
obs.export_dashboard("grafana-dashboard.json")
```

**Tracked Metrics:**
- Query latency (p50, p95, p99)
- Cache hit rates
- Tool usage distribution
- Confidence score distribution
- Error rates by type
- Token usage per model

---

### Evaluation Harness

Automated testing with LLM-as-judge:

```python
from evaluation.harness import EvaluationHarness, create_golden_dataset

# Create test dataset
dataset = create_golden_dataset()

# Run evaluation
harness = EvaluationHarness(model_adapter=adapter)
results = await harness.evaluate(dataset)

# Check for regressions
from evaluation.harness import check_for_regressions
regressions = check_for_regressions(
    current_results=results,
    baseline_version="v1.2.0"
)

print(f"Found {len(regressions)} regressions")
```

**Evaluation Criteria:**
- Relevance (0-5)
- Accuracy (0-5)
- Completeness (0-5)
- Clarity (0-5)
- Source Attribution (0-5)
- Safety Compliance (pass/fail)

---

## 📊 Monitoring Dashboard

Key metrics to monitor in production:

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Query Latency (p95) | < 2s | > 5s |
| Cache Hit Rate | > 60% | < 30% |
| Hallucination Rate | < 2% | > 5% |
| Guardrail Block Rate | < 5% | > 15% |
| Tool Error Rate | < 1% | > 5% |
| Session Retention | > 70% | < 50% |

---

## 🔒 Security Guidelines

1. **Never expose** database credentials or API keys in logs
2. **Validate all** user inputs before processing
3. **Rate limit** autonomous tool executions
4. **Audit log** all high-risk actions (database writes)
5. **Encrypt** sensitive data at rest and in transit
6. **Regularly rotate** API keys and passwords
7. **Monitor** for unusual query patterns (potential abuse)

---

## 🧪 Testing Checklist

Before deployment, verify:

- [ ] All golden test cases pass (evaluation harness)
- [ ] No regressions vs. previous version
- [ ] Cache hit rate > 50% on benchmark queries
- [ ] Guardrails block malicious inputs
- [ ] Observability traces complete end-to-end
- [ ] Model adapter fallback works (primary provider down)
- [ ] Database connection pooling handles load
- [ ] Redis cache eviction works correctly
- [ ] Memory agent preserves context across turns
- [ ] Tool retry logic handles transient failures

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Development Setup:**
```bash
pip install -r requirements-dev.txt
pytest tests/ --cov=querybot
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **LangChain/LangGraph** for the agent orchestration framework
- **pgvector** for efficient vector similarity search in PostgreSQL
- **OpenRouter** for unified model access
- **I-PAC** for the original QueryBot concept and political intelligence domain expertise

---

## 📞 Support

For issues and questions:
- GitHub Issues: [Create an issue](https://github.com/your-org/querybot/issues)
- Documentation: [Wiki](https://github.com/your-org/querybot/wiki)
- Email: support@querybot.ai

---

*Built with ❤️ for democratic transparency and data-driven political analysis.*
