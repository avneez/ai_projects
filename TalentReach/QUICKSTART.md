# TalenReach.ai - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Prerequisites
- Docker + Docker Compose
- NVIDIA GPU (for LLM inference)
- API Keys (OpenAI, Anthropic, Bright Data)

### 1. Clone & Setup

```bash
cd /home/abhishek/Desktop/PERSONAL/Projects/TalenReach

# Set environment variables
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
export BRIGHT_DATA_PROXY="http://user:pass@brd.superproxy.io:22225"
export LINKEDIN_EMAIL="your_linkedin_email"
export LINKEDIN_PASSWORD="your_linkedin_password"
export HUGGINGFACE_TOKEN="your_hf_token"
```

### 2. Start All Services

```bash
cd code-snippets/deployment
docker-compose up -d
```

### 3. Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| **FastAPI (Main API)** | http://localhost:8000/docs | - |
| **Airflow** | http://localhost:8080 | admin/admin |
| **Flower (Celery)** | http://localhost:5555 | - |
| **Grafana** | http://localhost:3000 | admin/admin |
| **Neo4j Browser** | http://localhost:7474 | neo4j/password123 |
| **RabbitMQ** | http://localhost:15672 | admin/password123 |
| **Jaeger (Tracing)** | http://localhost:16686 | - |

### 4. Trigger Your First Scrape

```bash
curl -X POST http://localhost:8000/candidates/scrape \
  -H "Content-Type: application/json" \
  -d '{"linkedin_url": "https://linkedin.com/in/example-profile"}'
```

**Response**:
```json
{
  "task_id": "abc-123",
  "status": "queued",
  "message": "Scraping job submitted"
}
```

### 5. Check Task Status

```bash
curl http://localhost:8000/tasks/abc-123
```

**Response**:
```json
{
  "task_id": "abc-123",
  "status": "SUCCESS",
  "result": {
    "status": "success",
    "candidate_id": "uuid-456",
    "profile_data": {...}
  }
}
```

### 6. Create a Job & Match Candidates

```bash
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Senior ML Engineer",
    "description": "Build production ML systems...",
    "tech_stack": ["PyTorch", "Kubernetes", "Python"],
    "min_years_experience": 5
  }'
```

**Response**:
```json
{
  "job_id": "job-789",
  "status": "created",
  "message": "Matching candidates in background"
}
```

### 7. View Matches

```bash
curl http://localhost:8000/jobs/job-789/matches?min_score=0.7
```

**Response**:
```json
[
  {
    "candidate_id": "uuid-456",
    "candidate_name": "John Doe",
    "match_score": 0.89,
    "reasoning": "Strong technical match, 6 years experience...",
    "green_flags": ["PyTorch expert", "Led team of 4", "Published papers"],
    "red_flags": []
  }
]
```

---

## 📊 Monitoring Dashboards

### Grafana (System Metrics)
1. Go to http://localhost:3000
2. Login: admin/admin
3. Navigate to Dashboards → TalenReach Overview
4. View:
   - Profiles scraped/day
   - Matching latency (P50, P99)
   - GPU utilization
   - Success rates

### Flower (Celery Tasks)
1. Go to http://localhost:5555
2. View:
   - Active tasks
   - Worker status
   - Task execution time
   - Success/failure rates

### Jaeger (Distributed Tracing)
1. Go to http://localhost:16686
2. Search for traces by:
   - Service: `matching-agent`
   - Operation: `match_candidates`
3. View flame graphs showing:
   - FAISS search time
   - LLM inference time
   - Database queries

---

## 🛠️ Development Workflow

### Running Tests

```bash
# Unit tests
pytest code-snippets/agents/test_profile_scraper_agent.py

# Integration tests
pytest code-snippets/agents/test_matching_agent.py --integration

# Load tests
locust -f code-snippets/tests/load_test.py --host=http://localhost:8000
```

### Viewing Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f fastapi
docker-compose logs -f celery-scraper-worker

# Filter for errors
docker-compose logs | grep ERROR
```

### Debugging LangGraph State Machines

```bash
# Check PostgreSQL for saved states
docker exec -it talenreach-postgres psql -U admin -d talenreach

# Query checkpoints
SELECT * FROM langgraph_checkpoints
WHERE thread_id LIKE 'scrape_%'
ORDER BY created_at DESC
LIMIT 10;

# View state transitions
SELECT thread_id, current_step, created_at
FROM langgraph_checkpoints
WHERE thread_id = 'scrape_example-profile';
```

### Inspecting FAISS Index

```python
# Python shell
from storage.vector_store import vectorstore

# Load index
vectorstore.load("faiss_index.bin")

# Check size
print(f"Total vectors: {vectorstore.index.ntotal}")

# Search
results = vectorstore.search_similar_candidates("ML engineer with PyTorch", top_k=10)
print(results)
```

---

## 🔧 Configuration

### Scaling Celery Workers

```yaml
# docker-compose.yml
celery-scraper-worker:
  deploy:
    replicas: 10  # Increase from 1 to 10
```

```bash
docker-compose up -d --scale celery-scraper-worker=10
```

### Adjusting vLLM GPU Allocation

```yaml
# docker-compose.yml
vllm-matching:
  deploy:
    resources:
      reservations:
        devices:
          - count: 4  # Use 4 GPUs for tensor parallelism
```

### Changing Kafka Retention

```yaml
# docker-compose.yml
kafka:
  environment:
    KAFKA_LOG_RETENTION_HOURS: 168  # 7 days (default: 168)
```

---

## 🐛 Troubleshooting

### Issue: "LinkedIn CAPTCHA appearing frequently"

**Solution**:
1. Reduce scraping rate (increase delays in RL agent)
2. Rotate proxy IPs more frequently
3. Check if YOLOv8 model is loaded correctly

```python
# Test CAPTCHA solver
from scraping.captcha_solver import solve_captcha_with_yolo
result = solve_captcha_with_yolo("test_captcha.png")
print(result)
```

### Issue: "vLLM GPU OOM error"

**Solution**:
1. Reduce batch size in matching agent
2. Use 4-bit quantization (QLoRA) instead of LoRA
3. Add more GPUs (tensor parallelism)

```bash
# Check GPU memory
nvidia-smi

# Restart vLLM container
docker-compose restart vllm-matching
```

### Issue: "PostgreSQL connection pool exhausted"

**Solution**:
1. Use PgBouncer for connection pooling
2. Increase max connections in PostgreSQL config

```sql
-- Check current connections
SELECT count(*) FROM pg_stat_activity;

-- Increase max connections (restart required)
ALTER SYSTEM SET max_connections = 200;
```

### Issue: "FAISS search slow (>1s)"

**Solution**:
1. Check index size: `vectorstore.index.ntotal`
2. Rebuild index with higher `efConstruction` (accuracy vs speed trade-off)
3. Migrate to Milvus (distributed) if >10M vectors

```python
# Rebuild FAISS index with better params
import faiss
index = faiss.IndexHNSWFlat(384, 64)  # M=64 (more connections)
index.hnsw.efConstruction = 400  # Higher = slower build, faster search
```

---

## 📈 Performance Tuning

### Optimize Scraping Throughput

```python
# airflow_dags.py
# Increase concurrent scraping tasks
scraping_pool = Pool(pool_name='scraping_pool', slots=50)  # From 10 to 50
```

### Optimize Matching Throughput

```python
# matching_agent.py
# Increase batch size for LLM scoring
BATCH_SIZE = 20  # From 10 to 20
```

### Optimize Database Queries

```sql
-- Create indexes
CREATE INDEX idx_candidates_skills ON candidates USING GIN ((profile_data->'skills'));
CREATE INDEX idx_matches_score ON matches(match_score DESC);

-- Analyze query plans
EXPLAIN ANALYZE SELECT * FROM matches WHERE match_score >= 0.7;
```

---

## 🎯 Next Steps

1. **Add More Profiles**:
   - Bulk upload CSV of LinkedIn URLs
   - Schedule daily scraping jobs in Airflow

2. **Fine-Tune Models**:
   - Collect 10K+ labeled examples
   - Train LoRA adapters (see `docs/llm_finetuning_guide.md`)

3. **Deploy to Production**:
   - Set up Kubernetes cluster
   - Configure auto-scaling (HPA)
   - Set up monitoring alerts (PagerDuty)

4. **Integrate with ATS**:
   - Connect to Greenhouse, Lever, etc.
   - Sync candidates bidirectionally

---

## 📚 Additional Resources

- [README.md](../README.md) - Full system architecture
- [PROJECT_SUMMARY.md](../PROJECT_SUMMARY.md) - High-level overview
- [Interview Prep Guide](docs/interview-prep.md) - 27 Q&A for interviews
- [LLM Fine-Tuning Guide](docs/llm_finetuning_guide.md) - LoRA/QLoRA training
- [Anti-Detection Guide](docs/anti_detection_guide.md) - Scraping best practices

---

## 💬 Support

- **Issues**: https://github.com/yourusername/TalenReach/issues
- **Docs**: https://talenreach.ai/docs
- **Email**: support@talenreach.ai

---

**Happy Recruiting! 🎉**
