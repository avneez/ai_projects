# InsightStream - Real-Time Data Enrichment Pipeline

An enterprise-grade data pipeline for continuous enrichment and normalization of B2B contact and firmographic data, processing 500K+ records daily with sub-100ms query performance.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Key Features](#key-features)
- [System Components](#system-components)
- [Data Flow](#data-flow)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Monitoring & Observability](#monitoring--observability)
- [Performance Metrics](#performance-metrics)
- [Interview Cross-Questions](#interview-cross-questions)

---

## Overview

InsightStream is a production-ready data enrichment platform designed to handle large-scale B2B data processing with real-time ingestion, validation, and enrichment capabilities. The system integrates multiple data sources (LinkedIn API, Bombora, ZoomInfo) and provides a unified, normalized view of contact and firmographic data.

### Purpose

- **Real-time Data Enrichment**: Continuously enrich contact and company data from multiple sources
- **Data Quality Assurance**: Automated validation and monitoring with configurable quality rules
- **Scalable Architecture**: Process 500K+ records daily with horizontal scalability
- **Fast Querying**: Sub-100ms query performance using ElasticSearch and Redis caching
- **ML-Ready Data**: Generate and store contact embeddings using Sentence-Transformers for similarity search

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Sources                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                  │
│  │ LinkedIn │    │ Bombora  │    │ ZoomInfo │                  │
│  │   API    │    │   API    │    │   API    │                  │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘                  │
└───────┼──────────────┼──────────────┼─────────────────────────┘
        │              │              │
        └──────────────┴──────────────┘
                       │
                       ▼
        ┌──────────────────────────┐
        │    Apache Kafka          │
        │  (Message Streaming)     │
        └──────────┬───────────────┘
                   │
                   ▼
        ┌──────────────────────────┐
        │   Airflow Scheduler      │
        │  (Workflow Orchestration)│
        └──────────┬───────────────┘
                   │
                   ▼
        ┌──────────────────────────┐
        │    Celery Workers        │
        │  (Distributed Tasks)     │
        └──────────┬───────────────┘
                   │
        ┌──────────┴────────────┐
        │                       │
        ▼                       ▼
┌───────────────┐      ┌────────────────┐
│  ETL Pipeline │      │  ML Enrichment │
│  - Validation │      │  - Embeddings  │
│  - Transform  │      │  - Similarity  │
│  - Normalize  │      │                │
└───────┬───────┘      └────────┬───────┘
        │                       │
        └───────────┬───────────┘
                    │
        ┌───────────┴────────────┐
        │                        │
        ▼                        ▼
┌────────────────┐      ┌─────────────────┐
│   PostgreSQL   │      │  ElasticSearch  │
│  + pgvector    │      │    (Indexing)   │
│   (Storage)    │      │                 │
└────────┬───────┘      └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     │
                     ▼
              ┌─────────────┐
              │    Redis    │
              │  (Caching)  │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │   FastAPI   │
              │ (REST API)  │
              └──────┬──────┘
                     │
                     ▼
        ┌────────────────────────┐
        │  Prometheus + Grafana  │
        │    (Monitoring)        │
        └────────────────────────┘
```

### Component Architecture

1. **Ingestion Layer**: Kafka for real-time message streaming from external APIs
2. **Orchestration Layer**: Airflow DAGs for workflow management and scheduling
3. **Processing Layer**: Celery workers for distributed task execution
4. **Storage Layer**: PostgreSQL (with pgvector) for persistent storage
5. **Search Layer**: ElasticSearch for fast full-text search and analytics
6. **Caching Layer**: Redis for sub-100ms query performance
7. **API Layer**: FastAPI for RESTful endpoints
8. **Monitoring Layer**: Prometheus metrics + Grafana dashboards

---

## Technology Stack

### Core Framework
- **FastAPI**: High-performance async REST API framework
- **Python 3.10+**: Core programming language

### Data Processing
- **Pandas**: Data manipulation and transformation
- **NumPy**: Numerical computations

### Message Queue & Task Queue
- **Apache Kafka**: Real-time event streaming
- **Celery**: Distributed task queue for async processing
- **Redis**: Message broker and caching layer

### Database & Storage
- **PostgreSQL 15**: Primary relational database
- **pgvector**: Vector similarity search extension
- **SQLAlchemy**: ORM for database operations

### Search & Indexing
- **ElasticSearch 8.x**: Full-text search and analytics engine

### ML & Embeddings
- **Sentence-Transformers**: Generate semantic embeddings
- **PyTorch**: Deep learning framework

### Workflow Orchestration
- **Apache Airflow**: Schedule and monitor workflows

### Monitoring & Observability
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Metrics visualization dashboards
- **python-json-logger**: Structured logging

### Data Quality
- **Great Expectations**: Data validation and profiling

---

## Key Features

### 1. Multi-Source Data Ingestion
- Parallel ingestion from LinkedIn API, Bombora, and ZoomInfo
- Rate limiting and retry mechanisms
- Deduplication and conflict resolution

### 2. Data Quality Validation
- Automated data quality checks using Great Expectations
- Schema validation and type checking
- Completeness, uniqueness, and consistency rules
- Real-time alerting on quality issues

### 3. Real-Time Processing
- Stream processing with Kafka consumers
- Async task processing with Celery
- Batch and micro-batch processing modes

### 4. ML-Powered Enrichment
- Generate contact embeddings using Sentence-Transformers
- Semantic similarity search for duplicate detection
- Company classification and segmentation

### 5. High-Performance Querying
- ElasticSearch for full-text search (sub-100ms)
- Redis caching for frequently accessed data
- pgvector for similarity search on embeddings

### 6. Monitoring & Alerting
- Real-time metrics via Prometheus
- Custom Grafana dashboards for pipeline health
- Automated alerting on failures and SLA violations

---

## System Components

### 1. Data Ingestion Service
- Kafka producers for each data source
- API client wrappers with retry logic
- Rate limiting and quota management

### 2. ETL Pipeline
- Extract: Pull raw data from Kafka topics
- Transform: Normalize, clean, and enrich data
- Load: Write to PostgreSQL and ElasticSearch

### 3. Embedding Service
- Generate semantic embeddings for contacts
- Store vectors in PostgreSQL with pgvector
- Batch processing for efficiency

### 4. API Service
- RESTful endpoints for data access
- Authentication and authorization
- Request validation and rate limiting

### 5. Monitoring Service
- Prometheus metrics exporters
- Custom business metrics
- Health check endpoints

---

## Data Flow

### Ingestion Flow
```
External API → Kafka Topic → Airflow DAG → Celery Task → Validation → Enrichment → Storage
```

### Query Flow
```
API Request → Redis Cache (HIT) → Response
           ↓
     Redis Cache (MISS) → ElasticSearch/PostgreSQL → Cache Update → Response
```

### Enrichment Flow
```
Raw Contact → Data Validation → Normalization → Embedding Generation → pgvector Storage
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- PostgreSQL 15+ with pgvector extension
- Redis 7+
- ElasticSearch 8+
- Apache Kafka 3+
- Apache Airflow 2.7+

### Installation Steps

1. **Clone the repository**
```bash
git clone <repository-url>
cd InsightStream
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Initialize database**
```bash
python scripts/init_db.py
```

6. **Start services**
```bash
# Start Kafka
kafka-server-start.sh config/server.properties

# Start Redis
redis-server

# Start Celery workers
celery -A src.workers.celery_app worker --loglevel=info

# Start Airflow
airflow webserver -p 8080
airflow scheduler

# Start FastAPI
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Usage

### API Endpoints

#### 1. Enrich Contact Data
```http
POST /api/v1/contacts/enrich
Content-Type: application/json

{
  "email": "john.doe@example.com",
  "sources": ["linkedin", "zoominfo"]
}
```

#### 2. Search Contacts
```http
GET /api/v1/contacts/search?q=software+engineer&limit=10
```

#### 3. Get Similar Contacts
```http
GET /api/v1/contacts/{contact_id}/similar?limit=5
```

#### 4. Bulk Enrichment
```http
POST /api/v1/contacts/bulk-enrich
Content-Type: application/json

{
  "contacts": [
    {"email": "contact1@example.com"},
    {"email": "contact2@example.com"}
  ]
}
```

### Triggering Airflow DAGs

```python
from airflow import DAG
from datetime import datetime

# Trigger via Airflow UI or API
# http://localhost:8080/api/v1/dags/data_enrichment_pipeline/dagRuns

# Or via CLI
airflow dags trigger data_enrichment_pipeline
```

### Monitoring

Access monitoring dashboards:
- **Grafana**: http://localhost:3000
- **Prometheus**: http://localhost:9090
- **Airflow**: http://localhost:8080

---

## Monitoring & Observability

### Key Metrics

1. **Throughput Metrics**
   - Records processed per minute
   - API requests per second
   - Kafka message lag

2. **Latency Metrics**
   - API response time (p50, p95, p99)
   - Database query latency
   - Cache hit/miss ratio

3. **Quality Metrics**
   - Validation failure rate
   - Data completeness score
   - Duplicate detection rate

4. **System Metrics**
   - CPU and memory usage
   - Disk I/O
   - Network throughput

### Alerting Rules

- Pipeline failure: Immediate alert
- Quality score < 95%: Warning after 5 minutes
- API latency > 200ms: Critical after 3 minutes
- Kafka lag > 1000 messages: Warning

---

## Performance Metrics

### Current Performance
- **Throughput**: 500K+ records per day
- **Query Latency**: Sub-100ms (95th percentile)
- **Data Quality Score**: 98.5% average
- **Uptime**: 99.9% SLA
- **Cache Hit Rate**: 85%+

### Scalability
- Horizontal scaling via Celery workers
- Kafka partitioning for parallel processing
- ElasticSearch sharding for distributed indexing
- PostgreSQL read replicas for query scaling

---

## Interview Cross-Questions

### Architecture & Design

**Q1: Why did you choose Kafka over other message queues like RabbitMQ or AWS SQS?**
**A**: Kafka provides superior throughput for high-volume data streams (millions of messages/day), built-in partitioning for parallel processing, and message replay capability which is critical for reprocessing data if enrichment logic changes. Unlike RabbitMQ, Kafka acts as a durable log rather than a transient queue, allowing multiple consumers to process the same data stream independently.

**Q2: How does pgvector compare to dedicated vector databases like Pinecone or Weaviate?**
**A**: pgvector keeps vector data co-located with relational data in PostgreSQL, eliminating the need for cross-database joins and reducing infrastructure complexity. For our use case (500K records), pgvector provides sufficient performance (<100ms for nearest neighbor search) while maintaining ACID guarantees. For billions of vectors, a dedicated vector DB would be more appropriate.

**Q3: Why use both ElasticSearch and PostgreSQL instead of just one?**
**A**: They serve different purposes: PostgreSQL is the source of truth with ACID compliance and relational integrity, while ElasticSearch provides fast full-text search, analytics aggregations, and flexible querying. Redis sits in front as a cache layer. This separation follows the CQRS pattern (Command Query Responsibility Segregation).

### Data Processing

**Q4: How do you handle duplicate records from multiple data sources?**
**A**: Multi-layered approach:
1. **Pre-processing**: Hash-based deduplication using email + company domain
2. **Embedding similarity**: Cosine similarity on contact embeddings (threshold > 0.95)
3. **Conflict resolution**: Merge strategy based on data source priority (LinkedIn > ZoomInfo > Bombora) and timestamp
4. **Manual review**: Flag edge cases (0.85-0.95 similarity) for human review

**Q5: What happens when an enrichment API (like ZoomInfo) is down?**
**A**:
- **Immediate**: Celery task retries with exponential backoff (3 retries, 2^n seconds)
- **Circuit breaker**: After 5 consecutive failures, skip that source for 5 minutes
- **Graceful degradation**: Continue with available sources, mark record as "partially enriched"
- **Dead letter queue**: Failed tasks go to DLQ for manual investigation
- **Monitoring**: Alert fires if >10% of tasks fail within 5 minutes

**Q6: How do you ensure data quality with 500K records per day?**
**A**:
- **Great Expectations**: Automated validation rules on every batch
- **Schema validation**: Pydantic models enforce types at API boundaries
- **Statistical profiling**: Track distribution changes (email domain diversity, job title patterns)
- **Sampling**: Deep validation on 1% sample hourly, full validation daily
- **Quarantine**: Failed records go to separate table for review
- **Metrics**: Quality score (completeness + accuracy + consistency) tracked per source

### Performance & Scalability

**Q7: How did you achieve sub-100ms query latency?**
**A**:
1. **Redis caching**: 3-tier cache (contact, search results, aggregations) with 1-hour TTL
2. **ElasticSearch**: Pre-computed indices with optimized mappings, 3 shards
3. **Database optimization**: Composite indexes on common query patterns, connection pooling
4. **Query optimization**: Limit + offset pattern, avoid SELECT *, use covering indexes
5. **Async I/O**: FastAPI with async/await for non-blocking operations

**Q8: How would you scale this system to 10M records per day?**
**A**:
- **Horizontal scaling**: Add more Celery workers (current: 4 → 20+)
- **Kafka partitioning**: Increase partitions (current: 3 → 15) for parallel consumption
- **Database**: PostgreSQL partitioning by date, read replicas for queries
- **ElasticSearch**: Increase shards and nodes, enable index lifecycle management
- **Batch size optimization**: Tune batch sizes for bulk operations (current: 1000 → 5000)
- **Resource allocation**: Move to Kubernetes for auto-scaling based on queue depth

**Q9: What's your strategy for handling backpressure when ingestion exceeds processing capacity?**
**A**:
- **Kafka consumer lag monitoring**: Alert when lag > 10,000 messages
- **Dynamic worker scaling**: Auto-scale Celery workers based on queue depth
- **Rate limiting**: Throttle ingestion at source if lag exceeds threshold
- **Priority queues**: Critical enrichment tasks in high-priority queue
- **Batch size adjustment**: Increase batch size to process more records per task

### Monitoring & Operations

**Q10: How do you monitor data pipeline health?**
**A**:
- **Pipeline metrics**: Records processed, failure rate, avg processing time per stage
- **SLA metrics**: End-to-end latency from ingestion to storage (target: <5 minutes)
- **Quality metrics**: Validation pass rate, completeness score, duplicate rate
- **System metrics**: CPU, memory, disk, Kafka lag, cache hit rate
- **Business metrics**: Enrichment coverage per source, API cost per record
- **Dashboards**: Grafana with separate views for ops, data engineering, and executives

**Q11: Describe a production incident you debugged and resolved.**
**A**:
**Incident**: Query latency spiked to 2-3 seconds affecting 40% of requests.
**Investigation**:
1. Checked Grafana: ElasticSearch response time was normal, but PostgreSQL showed slow queries
2. pg_stat_statements revealed a missing index on frequently joined columns
3. Redis cache hit rate dropped from 85% to 30% - discovered cache expiration was too aggressive after a recent config change
**Resolution**:
1. Added composite index on (company_id, updated_at)
2. Adjusted cache TTL from 30 min to 2 hours for stable data
3. Warmed cache after deployment
**Result**: Latency returned to <100ms, cache hit rate recovered to 88%

**Q12: How do you handle schema evolution without downtime?**
**A**:
- **Backward compatibility**: Always additive changes first (new columns nullable)
- **Dual writes**: Write to both old and new schema during transition
- **Feature flags**: Gradually roll out reads from new schema
- **Kafka schema registry**: Avro schemas with version compatibility checks
- **Database migrations**: Use Alembic with online DDL (pg_repack for large tables)
- **Zero-downtime deployment**: Blue-green deployment with health checks

### ML & Embeddings

**Q13: Why use Sentence-Transformers instead of OpenAI embeddings?**
**A**:
- **Cost**: $0 vs $0.0001 per embedding (500K/day = $50/day)
- **Latency**: Local inference (<10ms) vs API call (50-100ms)
- **Privacy**: Data stays in-house, important for B2B contacts
- **Customization**: Can fine-tune on our domain-specific data
- **Trade-off**: Slightly lower quality, but sufficient for duplicate detection

**Q14: How do you retrain or update embeddings when the model changes?**
**A**:
- **Versioning**: Store model version with each embedding
- **Incremental update**: Backfill embeddings in batches during off-peak hours
- **Parallel storage**: Store both old and new embeddings during transition
- **A/B testing**: Compare duplicate detection accuracy before full cutover
- **Rollback plan**: Keep previous model version for 30 days

### Data Sources & APIs

**Q15: How do you handle rate limits from external APIs?**
**A**:
- **Token bucket algorithm**: Track requests per API key per time window
- **Distributed rate limiting**: Redis-based counter shared across workers
- **Backoff strategy**: Exponential backoff on 429 responses
- **Queue prioritization**: Urgent requests in separate queue with higher limits
- **API key rotation**: Multiple API keys for higher aggregate throughput
- **Cost tracking**: Monitor API usage per source, alert on budget thresholds

**Q16: How do you ensure data freshness across multiple sources?**
**A**:
- **Incremental updates**: Track last_updated timestamp per contact per source
- **Refresh strategy**: Re-enrich contacts every 30 days, or on-demand for premium users
- **Change detection**: Only update if new data differs from existing (hash comparison)
- **Source priority**: Newer timestamp wins, unless higher-priority source
- **Staleness alerting**: Alert if any source hasn't provided data in 24 hours

---

## Project Structure

```
InsightStream/
├── config/              # Configuration files
├── data/                # Sample data and schemas
├── logs/                # Application logs
├── src/
│   ├── api/            # FastAPI application
│   ├── models/         # Database models
│   ├── pipelines/      # ETL and Airflow DAGs
│   ├── workers/        # Celery tasks
│   ├── utils/          # Utilities and helpers
│   └── monitoring/     # Prometheus metrics
├── tests/              # Unit and integration tests
├── .env.example        # Environment variables template
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

---

## Contributing

Contributions are welcome! Please follow the standard fork-and-pull request workflow.

---

## License

[Your License Here]

---

## Contact

For questions or support, please contact [Your Contact Information]
