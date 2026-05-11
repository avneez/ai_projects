# Comprehensive Interview Preparation Guide

## Table of Contents
1. [Data Engineering Questions](#data-engineering)
2. [NLP & News Processing](#nlp-news)
3. [RAG & LLM](#rag-llm)
4. [Machine Learning Models](#ml-models)
5. [System Design & Architecture](#system-design)
6. [Performance & Optimization](#performance)
7. [Production & Deployment](#production)
8. [Scenario-Based Questions](#scenarios)

---

## Data Engineering Questions {#data-engineering}

### Q: How do you handle multiple data sources with different schemas?
**A:** Use a common data model (standardized schema):
- Twitter, Telegram, LinkedIn → All converted to unified `NewsArticle` schema
- Fields: `text`, `tickers`, `timestamp`, `source`, `credibility_score`
- Transform at ingestion layer (adapter pattern)
- Benefits: Downstream services agnostic to source

### Q: How do you ensure data quality from social media?
**A:** Multi-layer validation:
1. **Ingestion**: Deduplication (SHA256 hashing)
2. **Validation**: Spam filters, credibility scoring
3. **Storage**: Only validated data enters vector DB
4. **Monitoring**: Track rejection rate, alert if >50%

### Q: What happens if one data source goes offline?
**A:** System continues with remaining sources:
- Each ingestion service independent (microservice)
- Prediction model trained to work with partial data
- Alert monitoring team to investigate
- Historical backfill when source recovers

### Q: How do you handle data deduplication across sources?
**A:** Content-based hashing:
- Normalize text (lowercase, remove URLs/mentions)
- SHA256 hash
- Store in Redis with 30-day TTL
- Cross-source deduplication (same news from Twitter + Telegram)

### Q: Explain your data pipeline's fault tolerance
**A:**
- **Kafka**: Message persistence, replay capability
- **Redis**: Publish-subscribe (fire-and-forget, but fast)
- **TimescaleDB**: PostgreSQL ACID guarantees
- **Milvus**: Replication and backup
- Trade-off: Strong vs eventual consistency

### Q: How do you clean and normalize financial text data?
**A:**
1. Remove HTML tags, URLs (keep domain)
2. Normalize whitespace
3. Handle special characters ($, %, numbers)
4. Ticker normalization (AAPL vs $AAPL vs Apple Inc)
5. Lowercase (except tickers)

### Q: Why TimescaleDB over InfluxDB for time-series?
**A:**
- **TimescaleDB**: PostgreSQL extension, SQL familiarity, joins, ACID
- **InfluxDB**: Purpose-built, faster writes, less flexible queries
- **Choice**: TimescaleDB (team knows SQL, need joins with relational data)

---

## NLP & News Processing {#nlp-news}

### Q: How do you handle news validation at scale?
**A:** Two-tier approach:
1. **Fast path** (80% filtered): Rule-based (regex, keyword matching) <1ms
2. **ML path** (20%): FinBERT classification ~50ms
3. Throughput: 1000+ articles/minute

### Q: Explain your approach to ticker extraction (NER)
**A:**
- **Primary**: Regex for cashtags ($AAPL) - 90% coverage, instant
- **Fallback**: spaCy custom NER model for "Apple Inc" → AAPL
- **Validation**: Check against NYSE/NASDAQ symbol list (daily update)
- **Context filtering**: "I ate an apple" → rejected (food context)

### Q: How do you measure and improve NER accuracy?
**A:**
- **Baseline**: Manual labeling of 1000 samples
- **Metrics**: Precision (94%), Recall (87%), F1 (90%)
- **Improvement**: Active learning (label edge cases), retrain monthly
- **A/B testing**: Shadow mode for new models

### Q: Why FinBERT over generic BERT?
**A:**
- **Pre-training**: Financial texts (10-Q, earnings calls, news)
- **Domain vocabulary**: Understands "bull", "bear", "resistance"
- **Performance**: 15% better accuracy on financial sentiment
- **Alternatives**: RoBERTa-base (general), DistilBERT (faster/smaller)

### Q: How do you handle multilingual news?
**A:** Current: English only (US market focus)
- **Detection**: `langdetect` library
- **Filter**: Keep English, discard others
- **Future**: Multilingual BERT (mBERT) for global expansion

### Q: What's your strategy for handling news bias?
**A:**
- **Source diversity**: Twitter (retail), LinkedIn (professional), financial APIs
- **Credibility scoring**: Weight by source reliability
- **LLM instruction**: "Maintain objectivity, cite multiple perspectives"
- **Monitoring**: Detect systematic bias (always bullish/bearish)

---

## RAG & LLM {#rag-llm}

### Q: Explain your RAG pipeline in detail
**A:**
1. **Query**: User/system asks about ticker
2. **Embedding**: Encode query (sentence-transformers)
3. **Retrieval**: Milvus vector search (top-k=10, cosine similarity)
4. **Filtering**: Time window, credibility threshold
5. **Context assembly**: Format articles for LLM
6. **Generation**: vLLM inference with structured prompt
7. **Parsing**: Extract JSON sentiment scores
8. **Validation**: Range checking, fallback if invalid

### Q: How do you prevent LLM hallucinations?
**A:**
1. **Grounding**: Instruct to use ONLY provided articles
2. **Low temperature**: 0.3 (vs 0.7-1.0 for creative tasks)
3. **Structured output**: JSON schema (easier to validate)
4. **Citation tracking**: Ask LLM to reference article numbers
5. **Fact-checking layer**: Cross-verify claims against retrieved docs
6. **Monitoring**: Sample outputs, calculate hallucination rate (<5% target)

### Q: Why vLLM over standard Transformers inference?
**A:**
- **Speed**: 24x throughput (240 vs 10 req/sec)
- **Latency**: 10x faster (200ms vs 2s)
- **Memory**: 25% less GPU memory (PagedAttention)
- **Features**: Continuous batching, optimized CUDA kernels
- **Cost**: Fewer GPUs needed

### Q: Llama 3.1 8B vs 70B vs GPT-4 - trade-offs?
**A:**
| Model | Quality | Speed | Cost | Choice |
|-------|---------|-------|------|--------|
| GPT-4 | Highest | Slow | $0.03/req | ✗ Expensive |
| Llama 70B | High | Medium | $0.001/req | ✗ Slower |
| **Llama 8B** | **Good** | **Fast** | **$0.0003/req** | ✓ **Best balance** |
| Phi-3 | Fair | Fastest | Free | For non-critical |

### Q: How do you version and update embeddings?
**A:**
- **Model versioning**: Lock to `all-MiniLM-L6-v2:v1.0.0`
- **Collection versioning**: `financial_news_v1`, `_v2` (separate)
- **A/B testing**: Route 10% traffic to new collection
- **Gradual migration**: If better, migrate 100% over 1 week
- **Rollback**: Keep old collection for 30 days

### Q: How do you handle context window limits (e.g., 4K tokens)?
**A:**
- **Truncation**: Top-k=10 articles ≈ 2-3K tokens (safe)
- **Summarization**: If needed, use extractive summary (first 2 sentences)
- **Chunking**: Split long articles, embed separately
- **Model choice**: Llama 3.1 has 8K context (comfortable margin)

### Q: RAG vs Fine-tuning - when to use each?
**A:**
| Aspect | RAG | Fine-tuning |
|--------|-----|-------------|
| **Data freshness** | Real-time | Static (snapshot) |
| **Update cost** | Low (add to DB) | High (retrain) |
| **Interpretability** | High (see sources) | Low (black box) |
| **Latency** | Higher (+retrieval) | Lower |
| **Use case** | News (daily changes) | Domain adaptation |

**Choice**: RAG for our use case (news changes constantly)

---

## Machine Learning Models {#ml-models}

### Q: Walk me through your model training pipeline
**A:**
1. **Data collection**: 2 years historical OHLCV + news
2. **Feature engineering**: 70 features (technical + sentiment)
3. **Train/val/test split**: Walk-forward (avoid look-ahead bias)
4. **Model training**: XGBoost (200 trees) + LSTM (2-layer)
5. **Hyperparameter tuning**: Optuna (100 trials)
6. **Evaluation**: Sharpe ratio, win rate, max drawdown
7. **Serialization**: Save to MLflow registry
8. **Backtesting**: Historical replay (validate on unseen data)

### Q: How do you prevent overfitting?
**A:**
1. **Walk-forward validation**: Train on Month 1-12, test on Month 13 (rolling)
2. **Regularization**: XGBoost (max_depth=6, min_child_weight), LSTM (dropout=0.3)
3. **Early stopping**: Stop if validation loss plateaus (20 rounds)
4. **Feature selection**: Remove low-importance features (SHAP analysis)
5. **Ensemble**: Combines XGBoost + LSTM (averages reduce overfitting)
6. **Cross-validation**: 5-fold time-series CV

### Q: Explain your ensemble logic
**A:**
```
XGBoost prediction: [0.68, 0.22, 0.10] (BUY/HOLD/SELL)
LSTM prediction:    [0.75, 0.15, 0.10]

Weighted average (60% XGB, 40% LSTM):
Final: [0.71, 0.19, 0.10]

LLM sentiment adjustment (+0.65):
Boost BUY by 0.065: [0.775, 0.19, 0.10]

Normalize: [0.79, 0.19, 0.02]

Decision: BUY (0.79 > 0.65 threshold)
```

**Why these weights?**
- XGBoost more reliable on tabular data (validation accuracy)
- LSTM captures momentum (complements XGBoost)
- LLM adds external context (10% influence)

### Q: How do you handle class imbalance?
**A:**
- **Data**: BUY (25%), HOLD (50%), SELL (25%)
- **Solutions**:
  1. SMOTE oversampling (minority classes)
  2. Class weights in loss function (inverse frequency)
  3. Stratified sampling (maintain distribution)
  4. Adjust decision threshold (lower for SELL if needed)

### Q: What features are most important?
**A:** (From SHAP analysis)
1. **RSI** (Relative Strength Index) - 18% importance
2. **LLM sentiment score** - 15%
3. **Volume ratio** - 12%
4. **MACD** - 10%
5. **Bollinger Band position** - 8%
6. Rest: 37% distributed

**Interpretation**: Momentum indicators + sentiment drive decisions

### Q: How do you evaluate model performance?
**A:**
- **Classification**: Accuracy (62%), Precision (70%), Recall (55%), F1 (0.61)
- **Trading**: Sharpe ratio (1.6), max drawdown (-12%), win rate (56%)
- **Financial**: Total return (45% over 1 year backtest), profit factor (1.8)
- **Benchmark**: Compare vs buy-and-hold (S&P 500: 15% same period) → Outperform

### Q: How often do you retrain models?
**A:**
- **Scheduled**: Monthly with latest data
- **Triggered**: If Sharpe ratio < 1.0 for 30 days
- **Drift detection**: Monitor feature distributions (KS test)
- **A/B test**: New model in shadow mode for 1 week before deployment

---

## System Design & Architecture {#system-design}

### Q: Explain the overall system architecture
**A:** Event-driven microservices:
1. **Ingestion**: Twitter/Telegram/LinkedIn → Kafka
2. **Processing**: Validation → Embedding → Milvus
3. **Storage**: TimescaleDB (OHLCV), Milvus (news vectors), PostgreSQL (trades)
4. **Trigger**: Price movement detector (Redis pub/sub)
5. **Prediction**: Feature engineering + ML + LLM → Signal
6. **Execution**: Pub/sub → Trade executors (horizontally scaled)
7. **Monitoring**: Prometheus + Grafana

### Q: Why microservices over monolith?
**A:**
- **Independence**: Twitter goes down, Telegram continues
- **Scaling**: Scale executors independently from ingestion
- **Deployment**: Update news validator without touching prediction model
- **Fault isolation**: One service crash doesn't kill system
- **Trade-off**: More complex (service discovery, inter-service communication)

### Q: How do you ensure low latency (<3s end-to-end)?
**A:**
- **Parallel processing**: Fetch features + RAG in parallel (multiprocessing)
- **Caching**: Redis for frequent queries (40-60% hit rate)
- **Batching**: GPU inference (32 samples/batch)
- **Optimized storage**: TimescaleDB indices, Milvus HNSW index
- **Profiling**: Identify bottlenecks (LLM inference = 2s, optimize first)

### Q: How does your system scale to 1000+ stocks?
**A:**
- **Horizontal scaling**:
  - Ingestion: Partition by source (3 services)
  - Price detector: Multiprocessing (10 workers, 100 stocks each)
  - Executors: 10 instances (distributed lock prevents duplicates)
  - vLLM: 4 GPU instances (load balanced)
- **Data partitioning**: Milvus partitions by date, TimescaleDB chunks
- **Auto-scaling**: Kubernetes HPA based on CPU/queue depth

### Q: What are the single points of failure (SPOFs)?
**A:**
1. **Redis**: Clustered mode (3 nodes) + persistent storage
2. **TimescaleDB**: Primary-replica setup, auto-failover
3. **Milvus**: Distributed deployment (3 nodes)
4. **vLLM**: Multiple instances behind load balancer
5. **Trade broker API**: Retry logic + circuit breaker

### Q: How do you handle message ordering?
**A:**
- **Kafka**: Partitions maintain order per key (symbol)
- **Redis pub/sub**: No ordering guarantee (acceptable for our use case)
- **TimescaleDB**: Timestamp-based querying ensures chronological order

---

## Performance & Optimization {#performance}

### Q: What's your system's throughput?
**A:**
- **Ingestion**: 10,000+ tweets/hour, 100 Telegram messages/minute
- **Validation**: 1,000 articles/minute
- **Prediction**: 10-20 predictions/minute (limited by LLM)
- **Execution**: 20 orders/minute (broker API limit)

### Q: Where are the bottlenecks?
**A:**
1. **LLM inference**: 2s (dominates latency)
2. **Milvus search**: 50-100ms
3. **TimescaleDB query**: 20-50ms
4. **Broker API**: 200ms (network)

**Mitigation**: Caching, batching, async processing

### Q: How do you optimize GPU utilization?
**A:**
- **Batching**: 32 requests/batch (80% GPU utilization)
- **Mixed precision**: FP16 (2x speedup)
- **Quantization**: INT8 (4x throughput, 4% accuracy loss)
- **Model selection**: 8B model (fits on single GPU vs 70B needing 4 GPUs)
- **Monitoring**: Track GPU memory, utilization (Prometheus)

### Q: How do you reduce costs?
**A:**
- **vLLM vs GPT-4 API**: 150x cheaper ($0.0003 vs $0.045/request)
- **GPU selection**: L4 ($0.70/hr) vs A100 ($3/hr) for 8B model
- **Spot instances**: 70% discount (acceptable for non-critical services)
- **Caching**: 40% cache hit rate = 40% fewer LLM calls
- **Serverless**: Shut down non-trading hours (save 16 hours/day)

---

## Production & Deployment {#production}

### Q: How do you deploy new models without downtime?
**A:** Blue-green deployment:
1. Deploy new model (green) alongside old (blue)
2. Route 10% traffic to green (shadow mode)
3. Compare predictions: green vs blue
4. If green better: Gradually shift 100% traffic
5. Keep blue running 24h (rollback buffer)
6. Decommission blue

### Q: What monitoring do you have in place?
**A:**
- **Metrics** (Prometheus): Latency (p50, p95, p99), error rates, throughput
- **Dashboards** (Grafana): Real-time P&L, win rate, signal distribution
- **Alerts** (PagerDuty): Executor down, API errors >5%, drawdown >10%
- **Logs** (ELK): Centralized logging with trace IDs
- **APM** (DataDog): Application performance monitoring

### Q: How do you test the system?
**A:**
1. **Unit tests**: pytest (>80% coverage)
2. **Integration tests**: Test end-to-end pipeline with mock data
3. **Backtesting**: Historical replay (validate strategy)
4. **Paper trading**: 2 weeks with fake money
5. **Shadow mode**: Run in parallel with production (compare signals)
6. **Gradual rollout**: 1% → 10% → 100% over 1 week

### Q: What's your disaster recovery plan?
**A:**
- **Backups**: Daily snapshots (TimescaleDB, PostgreSQL, Milvus)
- **Multi-region**: Primary (us-east-1), backup (us-west-2)
- **RTO**: 15 minutes (Recovery Time Objective)
- **RPO**: 1 hour (Recovery Point Objective - max data loss)
- **Runbook**: Step-by-step recovery procedures

---

## Scenario-Based Questions {#scenarios}

### Q: A model starts performing badly in production. How do you debug?
**A:**
1. **Check data drift**: Feature distributions changed?
2. **Validate inputs**: Are features being calculated correctly?
3. **Review predictions**: Sample 100 recent predictions, analyze patterns
4. **Check market regime**: Volatility spike? New market conditions?
5. **Compare vs baseline**: Simple strategy (buy-and-hold) outperforming?
6. **Rollback if needed**: Switch to previous model version
7. **Post-mortem**: Root cause analysis, improve monitoring

### Q: You notice LLM hallucination rate increased to 15%. What do you do?
**A:**
1. **Immediate**: Lower temperature (0.3 → 0.1), more explicit instructions
2. **Sample outputs**: Identify common hallucination patterns
3. **Check retrieval quality**: Is RAG returning irrelevant articles?
4. **Prompt engineering**: Add "Cite sources" instruction
5. **Model update**: Try different model (Llama → Mistral)
6. **Human-in-loop**: Flag low-confidence outputs for review

### Q: Your trade executor is lagging, orders taking 10+ seconds. Debug process?
**A:**
1. **Check metrics**: CPU, memory, network latency
2. **Broker API**: Is the API slow? (check their status page)
3. **Queue depth**: Backlog of pending signals?
4. **Database**: Slow writes to trade log?
5. **Distributed lock**: Lock contention between executors?
6. **Solution**: Scale executors, optimize DB writes (batching), contact broker

### Q: You need to add a new data source (e.g., Bloomberg). How?
**A:**
1. **Design adapter**: Bloomberg API → Unified NewsArticle schema
2. **Authentication**: API key management (Vault)
3. **Rate limiting**: Respect Bloomberg API limits
4. **Testing**: Unit tests + integration tests
5. **Shadow mode**: Ingest but don't use for predictions (1 week)
6. **Validation**: Compare quality vs existing sources
7. **Gradual rollout**: Include in prediction pipeline at 10% weight
8. **Monitor**: Track Bloomberg-specific metrics

---

## Key Takeaways for Interviews

### Always Mention:
1. **Trade-offs**: "I chose X because Y, but Z would work if..."
2. **Metrics**: Quantify everything (latency, throughput, accuracy)
3. **Alternatives**: Show you considered multiple approaches
4. **Production concerns**: Monitoring, scaling, fault tolerance
5. **Business impact**: How does this improve trading performance?

### Avoid:
1. **"I don't know"**: Say "I haven't worked with X, but I'd approach it like..."
2. **Overconfidence**: Acknowledge limitations
3. **Theory only**: Mention practical experience when possible
4. **Ignoring trade-offs**: Every decision has pros/cons

### Practice Format:
**Situation** → **Task** → **Action** → **Result** (STAR method)

Example: "When our LLM hallucination rate spiked (S), I needed to reduce it below 5% (T). I adjusted temperature, improved prompts, and added citation tracking (A). Result: 4% hallucination rate, 92% user satisfaction (R)."
