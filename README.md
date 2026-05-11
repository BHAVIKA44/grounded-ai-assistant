# Grounded AI Assistant

Grounded AI Assistant is a FastAPI + React Retrieval-Augmented Generation (RAG) system designed for practical, interview-ready backend architecture. It supports document ingestion, hybrid retrieval, grounded answer generation with citations, evaluation workflows, and operational controls through an admin UI.

## Key Capabilities

- Document ingestion for `PDF`, `DOCX`, and `TXT`
- Parsing, chunking, and indexing pipeline
- Hybrid retrieval (`BM25` + vector search) with optional reranking
- Grounded chat answers with source attribution
- Out-of-context fallback behavior
- Redis response caching
- RAG quality evaluation (single + batch)
- Admin operations for model selection, cache control, system status, and fine-tuning controls
- Multi-provider LLM support:
  - Local: `ollama`
  - Hosted: `groq` (free tier friendly), `openai`

## Architecture

```text
backend/app/
├── api/           # HTTP route layer
├── core/          # config, settings, logging, typed exceptions
├── db/            # SQLAlchemy base/models/session
├── services/      # ingestion, llm, cache, parsing, chunking
├── retrieval/     # bm25, hybrid, reranker
├── evaluation/    # rag evaluator
├── fine_tuning/   # trainer workflow
├── schemas/       # request/response contracts
├── utils/         # retry, security, text helpers
└── main.py        # FastAPI app entrypoint
```

## Local Run (Docker, Recommended)

1. Start services:

```bash
docker compose -f infra/docker-compose.yml up -d --build
```

2. Open:
- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

3. Verify health:

```bash
curl http://localhost:8000/health
```

## LLM Provider Setup

### Local mode (Ollama)

- Keep `OLLAMA_BASE_URL=http://ollama:11434` in Docker environment.
- In Admin tab:
  - Provider: `ollama`
  - Model: one of supported ollama models

### Hosted mode (Groq / OpenAI)

- Set environment variables in backend deployment:
  - Groq:
    - `GROQ_API_KEY`
    - `GROQ_MODEL` (default: `llama-3.1-8b-instant`)
    - `GROQ_BASE_URL` (default set in app)
  - OpenAI:
    - `OPENAI_API_KEY`
    - `OPENAI_MODEL`
    - `OPENAI_BASE_URL` (default set in app)
- In Admin tab:
  - Provider: `groq` or `openai`
  - Select/enter model and update

## API Surface (Core)

- `POST /api/v1/documents` and `POST /api/v1/documents/upload`
- `GET /api/v1/documents`
- `DELETE /api/v1/documents/{document_id}`
- `POST /api/v1/ask`
- `GET /api/v1/retrieval`
- `POST /api/v1/evaluation`
- `POST /api/v1/evaluation/batch`
- `GET /api/v1/admin/status`
- `GET /api/v1/admin/llm`
- `POST /api/v1/admin/llm`
- `POST /api/v1/admin/llm/pull`
- `DELETE /api/v1/admin/cache`
- `POST /api/v1/admin/fine-tuning/prepare`
- `POST /api/v1/admin/fine-tuning/run`
- `GET /api/v1/admin/fine-tuning/status`

## Deployment (Free-Tier Friendly)

Recommended split:
- Frontend: Vercel
- Backend: Render Web Service
- Postgres: Supabase or Render Postgres
- Redis: Upstash
- LLM: Groq API

Use Ollama for local development only; avoid self-hosted Ollama in free cloud hosting.

## Engineering Notes

- Structured logging is enabled across request lifecycle and service boundaries.
- Error responses are standardized with `code` + `message` in key APIs.
- Citation handling is normalized to known source names from retrieved contexts.
- Fine-tuning endpoint behavior depends on runtime:
  - GPU runtime: training path is initialized for LoRA workflow.
  - CPU-only runtime: API completes with a safe fallback artifact at `./models/lora/fine_tune_summary.json` to avoid hard failures.

## Roadmap / TODO

### High-priority

- Implement full fine-tuning execution pipeline in production mode (dataset tokenization, Trainer loop, checkpointing, adapter export) and validate end-to-end on GPU runtime
- Add integration tests for:
  - upload → index → ask
  - retrieval quality checks
  - provider switching (`ollama`/`groq`/`openai`)
- Add migration-driven DB lifecycle (strict Alembic versioning in CI)

### Platform hardening

- Add request correlation IDs across frontend ↔ backend logs
- Add `/metrics` endpoint with latency, cache hit-rate, and retrieval quality stats
- Add circuit-breaker + retry policy for external LLM provider calls
- Improve startup warmup to reduce first-query latency for embedding/reranker models

### Product quality

- Add per-document filtering in chat queries
- Add document-level indexing progress in UI
- Add exportable evaluation reports from RAG Ops

## License

MIT
