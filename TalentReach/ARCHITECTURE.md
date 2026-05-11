# TalentFlow — Autonomous AI Recruitment Platform

> End-to-end candidate discovery, screening, outreach, and interview scheduling — zero human intervention.

---

## Overview

TalentFlow is a production-grade multi-agent AI system that autonomously recruits candidates by scraping LinkedIn profiles, enriching them with multimodal LLMs, matching them against job requirements, generating personalized outreach, and conducting conversational screening — all without human input.

**Stack:** Python · LangGraph · Neo4j · FAISS · Kafka · Airflow · Celery · GPT-4V · Claude 3.5 · Llama 3.1

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATION LAYER                          │
│              Airflow DAGs  ──────────  Celery Workers               │
│         (Scheduling & DAGs)          (Distributed Tasks)            │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│                    LANGGRAPH MULTI-AGENT LAYER                      │
│                                                                     │
│  [ProfileScraperAgent] → [ProfileEnrichmentAgent] → [MatchingAgent] │
│                                          ↓                          │
│                              [OutreachAgent] → [ConversationAgent]  │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  INTELLIGENCE │    │  STORAGE LAYER   │    │   EVENT BUS         │
│    LAYER      │    │                  │    │                     │
│               │    │  PostgreSQL      │    │  Kafka              │
│  GPT-4 Vision │    │  (relational)    │    │  (Avro + CQRS)      │
│  Claude 3.5   │    │                  │    │                     │
│  Llama 3.1 8B │    │  FAISS           │    │  ClickHouse         │
│  Llama 3.3 70B│    │  (vector search) │    │  (analytics OLAP)   │
│  vLLM serving │    │                  │    │                     │
└───────────────┘    │  Neo4j           │    └─────────────────────┘
                     │ (knowledge graph)│
                     │                  │
                     │  Redis           │
                     │  (cache + pubsub)│
                     └──────────────────┘
```

---

## The 5 LangGraph Agents

Each agent is a **LangGraph state machine** — a directed graph of nodes (processing steps) with conditional routing, automatic checkpointing to PostgreSQL/Redis, and built-in retry/fallback logic.

### 1. ProfileScraperAgent

**Job:** Scrape raw LinkedIn profiles using Playwright.

**State flow:**
```
init_scrape → detect_captcha → solve_captcha → extract_profile → validate_data → emit_event
                   ↓ (no captcha)
              extract_profile
```

**Anti-Detection Layer** (sits underneath this agent):
- **PPO RL Agent** — trained with Stable-Baselines3 on 100K episodes. Actions: scroll, click, move mouse, pause. Reward: +100 (successful scrape), -100 (bot detected). Learned Bezier-curve mouse paths and variable scroll speeds (200–600px). Result: 10% → 94% success rate.
- **Fingerprint Spoofer** — injects noise into Canvas/WebGL/Audio APIs and randomizes TLS fingerprints via `curl-cffi` + `uTLS` to impersonate real browsers.
- **CAPTCHA Solver** — YOLOv8 custom model (90% solve rate) with 2Captcha API fallback (100% combined).
- **Residential Proxies** — Bright Data integration; rotates real user IPs to avoid datacenter blocks.
- **Session Persistence** — SQLite-based cookie store with AES-256 encryption across sessions.

---

### 2. ProfileEnrichmentAgent

**Job:** Enrich scraped profiles with multimodal analysis and generate embeddings.

**Processing pipeline:**
- **GPT-4 Vision** — analyzes profile photo for professionalism scoring.
- **Sentence-BERT** — generates 768-dim embeddings; stored in FAISS with HNSW indexing.
- **Neo4j** — maps career trajectory as a graph:
  ```cypher
  (candidate)-[:WORKED_AT]->(company)-[:IN_INDUSTRY]->(industry)
  (candidate)-[:HAS_SKILL]->(skill)-[:RELATED_TO]->(skill)
  (candidate)-[:REFERRED_BY]->(recruiter)
  ```
  This enables graph traversals like: *"candidates who left Google for a startup with Python expertise"* — 100x faster than equivalent SQL JOINs.

---

### 3. MatchingAgent

**Job:** Score candidates against job requirements.

**Two-stage retrieval:**
1. **FAISS HNSW** — retrieves top-100 candidates by embedding similarity in <10ms across 500K+ vectors (95% recall@10).
2. **Llama 3.3 70B LoRA** — fine-tuned on 100K recruitment examples, re-ranks and scores each candidate (0–1). Only candidates scoring >0.7 proceed.

**Why fine-tune instead of GPT-4?**
- 67x cheaper per match ($0.0003 vs $0.02)
- 4x faster (850ms vs 3.5s via vLLM)
- Domain-specific accuracy: 87% offer correlation vs ~75% GPT-4 zero-shot

---

### 4. OutreachAgent

**Job:** Generate and send personalized LinkedIn messages.

- **Claude 3.5 Sonnet** — generates messages tailored to the candidate's background, company, skills, and the specific role. No templates.
- **Result:** 34% response rate vs 12% industry average (+183%).
- Messages validated against compliance rules (no salary mentions in initial outreach, GDPR opt-out included).

---

### 5. ConversationAgent

**Job:** Multi-turn conversational screening + interview scheduling.

**Components:**
- **Llama 3.1 8B QLoRA** — fine-tuned intent classifier (150ms latency, 94% accuracy). Classifies: `question`, `interested`, `not_interested`, `reschedule`, `salary_inquiry`.
- **RAG Pipeline** — on candidate questions, retrieves relevant context (job description, company info) from FAISS, passes to LLM with temperature=0.1. Hallucination prevention: model is instructed to only answer from retrieved context.
- **Memory** — conversation history checkpointed via LangGraph to PostgreSQL; supports multi-turn sessions.
- **Calendar Integration** — Google Calendar API for automated interview slot booking.

---

## LLM Strategy

| Task | Model | Method | Latency | Cost/10K candidates |
|------|-------|--------|---------|---------------------|
| Photo analysis | GPT-4 Vision | API | ~2s | ~$80 |
| Outreach generation | Claude 3.5 Sonnet | API | ~1.5s | ~$40 |
| Candidate matching | Llama 3.3 70B | LoRA fine-tune | 850ms | ~$80 (GPU) |
| Intent classification | Llama 3.1 8B | QLoRA fine-tune | 150ms | ~$20 (GPU) |

**vLLM** serves the fine-tuned models with PagedAttention (virtual memory for KV cache): 24x throughput vs HuggingFace Transformers, 30% less GPU memory for 70B models.

---

## Data Layer

### PostgreSQL
Relational data: candidates, jobs, matches, conversations. Write model for CQRS. PgBouncer for connection pooling.

### FAISS (HNSW)
- 768-dim Sentence-BERT embeddings
- HNSW index: 6ms P50 / 12ms P99 across 500K vectors
- Sub-10ms similarity search at 95% recall@10

### Neo4j Knowledge Graph
Career trajectories, skill relationships, referral networks. Enables complex relationship queries that would require many SQL JOINs.

### Redis
- LLM response caching (reduces redundant API calls)
- Pub/Sub for real-time conversation streaming
- LangGraph state checkpointing for fast resume

### TimescaleDB
Read model for CQRS — time-series analytics for system metrics and recruitment KPIs.

---

## Event-Driven Architecture

**Kafka** (with Avro schema registry) as the central event bus:

```
profile.scraped   →  triggers ProfileEnrichmentAgent
profile.enriched  →  triggers MatchingAgent
match.found       →  triggers OutreachAgent
outreach.sent     →  starts ConversationAgent
interview.scheduled → closes pipeline loop
```

**CQRS Pattern:**
- Write commands go to PostgreSQL (ACID transactions)
- Kafka events propagate changes to TimescaleDB read model
- Enables independent scaling of read and write workloads

**Saga Pattern** for distributed transactions — each pipeline step emits an event; the next step subscribes. Failed steps emit compensation events (e.g., `enrichment.failed` → mark candidate `pending_retry`).

**Circuit Breaker** on all external API calls — opens after 5 consecutive failures, recovers after 60s. Prevents cascading failures across the microservice mesh.

---

## Orchestration

### Airflow DAGs
- `daily_profile_scraping` — kicks off Celery scraping tasks each morning
- `weekly_model_retraining` — fine-tunes intent classifier on new conversation data
- `graph_update` — refreshes Neo4j career graph relationships

### Celery
Distributed task processing with RabbitMQ broker and priority queues. Tasks: `scrape_profile`, `generate_embedding`, `match_candidates`, `send_outreach`. Workers scale horizontally; each worker picks up tasks independently.

---

## Key Performance Numbers

| Metric | Value |
|--------|-------|
| Profiles scraped/day | 2,000+ per worker |
| Matching throughput | 500 matches/min (vLLM) |
| FAISS P50 latency | 6ms (500K vectors) |
| Microservice P99 latency | <150ms |
| End-to-end P99 latency | <3s |
| System uptime | 99.9% |
| Response rate | 34% (vs 12% industry avg) |
| Time-to-interview | 3.2 days (vs 14 days) |
| Cost per hire | $260 (vs $4,000 traditional) |

---

## Tech Stack Summary

| Category | Tools |
|----------|-------|
| Agent framework | LangGraph |
| LLMs | GPT-4 Vision, Claude 3.5 Sonnet, Llama 3.1 8B, Llama 3.3 70B |
| LLM serving | vLLM (PagedAttention) |
| Fine-tuning | QLoRA (8B), LoRA (70B) |
| Vector search | FAISS (HNSW) |
| Knowledge graph | Neo4j |
| Relational DB | PostgreSQL + PgBouncer |
| Cache / pubsub | Redis |
| Analytics DB | TimescaleDB + ClickHouse |
| Event bus | Kafka + Avro schema registry |
| Task queue | Celery + RabbitMQ |
| Workflow orchestration | Apache Airflow |
| Scraping | Playwright + Stealth |
| RL framework | Stable-Baselines3 (PPO) |
| CAPTCHA solving | YOLOv8 + 2Captcha |
| API layer | FastAPI (REST + WebSocket) + gRPC |
| Containerization | Docker Compose (12 services) + Kubernetes |
| Observability | OpenTelemetry + Grafana |
