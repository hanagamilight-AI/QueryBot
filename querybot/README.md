# QueryBot - Political Intelligence Conversational AI

## Overview

QueryBot is a sophisticated agent-powered conversational AI system designed for political intelligence analysis. Built with LangGraph, PostgreSQL + pgvector, and FastAPI, it enables natural language querying of election data, survey results, social media sentiment, and manifesto documents.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         API Layer                                │
│                      FastAPI REST Endpoints                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Agent Orchestration                         │
│                    LangGraph State Machine                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Query      │→ │  Retrieval   │→ │  Synthesis   │          │
│  │  Analysis    │  │    Node      │  │    Node      │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         │                                    │                   │
│         ▼                                    ▼                   │
│  ┌──────────────┐                    ┌──────────────┐          │
│  │   Intent     │                    │ Validation   │          │
│  │ Classification│                   │    Node      │          │
│  └──────────────┘                    └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MCP Tool Layer                               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐ │
│  │  Database   │ │   Vector    │ │  External   │ │Conversation│ │
│  │    Tool     │ │   Search    │ │    APIs     │ │   Memory  │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Data Layer                                   │
│              PostgreSQL + pgvector Extension                     │
│  ┌─────────────────────┐  ┌─────────────────────┐               │
│  │   Relational Data   │  │   Vector Embeddings │               │
│  │  (metadata, filters)│  │  (semantic search)  │               │
│  └─────────────────────┘  └─────────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- **Semantic Search**: Vector-based similarity search using pgvector
- **Multi-turn Conversations**: Context-aware chat with persistent memory
- **Source Attribution**: All responses cite original sources
- **Confidence Scoring**: Transparency about answer reliability
- **Hybrid Retrieval**: Combines vector search with structured SQL queries
- **MCP Tools**: Standardized tool interface for extensibility

## Project Structure

```
querybot/
├── agents/
│   └── langgraph_agent.py      # LangGraph agent architecture
├── api/
│   └── main.py                 # FastAPI REST endpoints
├── config/
│   └── settings.py             # Configuration management
├── database/
│   └── models.py               # SQLAlchemy models + pgvector
├── data_ingestion/
│   ├── ingestion.py            # Data pipeline
│   └── chunking.py             # Document chunking strategies
├── embeddings/
│   └── embedder.py             # Embedding service
├── tools/
│   └── mcp_tools.py            # MCP tool implementations
├── requirements.txt
├── README.md
└── .env.example
```

## Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL 14+ with pgvector extension
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

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Run the application**
```bash
# Start the API server
python -m api.main

# Or with uvicorn directly
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

6. **Access the API**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

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

agent = get_agent()

# Single query
result = agent.query("What are the key manifesto promises?")
print(result['response'])

# Chat with context
session_id = "session-123"
response1 = agent.chat("Analyze voter sentiment in urban areas", session_id)
response2 = agent.chat("What about rural areas?", session_id)
```

## Agent Architecture

The LangGraph agent implements a state machine with specialized nodes:

1. **Query Analysis Node**: Classifies intent, extracts entities
2. **Retrieval Node**: Executes vector search + SQL queries
3. **Synthesis Node**: Generates coherent responses
4. **Validation Node**: Checks quality and confidence
5. **Memory Node**: Manages conversation history

### State Flow
```
Query → Analysis → Retrieval → Synthesis → Validation → Memory → Response
```

## MCP Tools

Available tools exposed via Model Context Protocol:

| Tool | Description |
|------|-------------|
| `database_query` | Structured SQL queries with filters |
| `vector_search` | Semantic similarity search |
| `external_api` | Real-time news/social data |
| `conversation_memory` | Session-based memory management |

## Data Ingestion

```python
from data_ingestion.ingestion import DataIngestionPipeline

pipeline = DataIngestionPipeline()
count = pipeline.ingest_from_directory("data/elections", "election")
pipeline.close()
```

Supported formats: JSON, CSV, TXT, PDF

## Configuration

Key environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://...` |
| `PGVECTOR_DIMENSION` | Embedding dimension | `768` |
| `EMBEDDING_MODEL` | Sentence transformer model | `all-MiniLM-L6-v2` |
| `LLM_MODEL` | LLM for synthesis | `gpt-4-turbo-preview` |
| `API_PORT` | Server port | `8000` |

## Deployment on AWS EC2

1. **Launch EC2 instance** (Ubuntu 22.04, t3.medium+)
2. **Install dependencies**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib python3-pip
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
4. **Deploy application**
```bash
git clone <repo>
pip install -r requirements.txt
```
5. **Run with systemd**
```ini
[Unit]
Description=QueryBot API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/querybot
ExecStart=/home/ubuntu/querybot/venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

## Performance Optimization

- Use batch embedding generation
- Implement caching for frequent queries
- Configure connection pooling
- Use async database operations
- Deploy behind load balancer for scale

## License

MIT License

## Contact

For questions and support, contact the development team.
