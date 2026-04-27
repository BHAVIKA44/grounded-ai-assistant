# Grounded AI Assistant

A production-grade Retrieval-Augmented Generation (RAG) system with hybrid retrieval, cross-encoder reranking, and local inference support.

## Features

- **Hybrid Retrieval**: BM25 + Vector search combined
- **Cross-Encoder Reranking**: Optimized result ranking
- **Local Inference**: Ollama integration for private LLM
- **Observability**: LangSmith tracing support
- **Fine-tuning**: LoRA/QLoRA support for custom models
- **Caching**: Redis-powered response caching
- **Production-Ready**: Dockerized with monitoring

## Tech Stack

- **Backend**: FastAPI, Python 3.11+
- **LLM Framework**: LangChain, LangGraph
- **Vector Store**: ChromaDB
- **Keyword Search**: rank_bm25
- **Embeddings**: SentenceTransformers
- **Cache**: Redis
- **Database**: PostgreSQL with SQLAlchemy async
- **Observability**: LangSmith
- **LLM Runtime**: Ollama
- **Fine-Tuning**: PEFT (LoRA/QLoRA)
- **Frontend**: React 18

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Node.js 18+
- PostgreSQL 15+
- Redis 7+

### Docker Compose (Recommended)

```bash
cd infra
docker-compose up -d
```

Services will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- PostgreSQL: localhost:5432
- Redis: localhost:6379
- Ollama: http://localhost:11434

### Manual Setup

1. Clone the repository:
```bash
cd grounded-ai-assistant
```

2. Copy environment file:
```bash
cp .env.example .env
```

3. Start infrastructure:
```bash
cd infra
docker-compose up -d postgres redis ollama
```

4. Install Python dependencies:
```bash
cd backend
pip install -r requirements.txt
```

5. Run the backend:
```bash
uvicorn app.main:app --reload
```

6. Install frontend dependencies:
```bash
cd frontend
npm install
```

7. Run the frontend:
```bash
npm start
```
```

## Architecture

```
backend/
├── app/
│   ├── api/          # FastAPI routes
│   ├── core/         # Config, logging
│   ├── services/     # Business logic
│   ├── retrieval/    # RAG (BM25 + vector + rerank)
│   ├── models/       # LLM + embeddings
│   ├── evaluation/   # RAG evaluation
│   └── schemas/      # Pydantic models
├── tests/
└── requirements.txt

frontend/
├── src/
└── public/

infra/
└── docker-compose.yml
```

## API Endpoints

### Documents

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/documents/upload` | Upload documents |
| GET | `/api/v1/documents` | List all documents |
| GET | `/api/v1/documents/{id}` | Get document by ID |
| DELETE | `/api/v1/documents/{id}` | Delete document |

### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/chat/ask` | Ask a question |

### Evaluation

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/evaluation/evaluate` | Evaluate RAG response |

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/ready` | Readiness check |

## Environment Variables

See `.env.example` for all configuration options.

## Development

Run tests:
```bash
cd backend
pytest tests/ -v --cov
```

## License

MIT