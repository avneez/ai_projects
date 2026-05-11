# RAG Service (Retrieval-Augmented Generation)

## Purpose
Retrieve relevant news/financial reports from Milvus and provide context to LLM for generating market sentiment scores.

## Libraries Used
- **LangChain** - RAG orchestration framework
- **pymilvus** - Vector database client
- **sentence-transformers** - Query embedding
- **Redis** - Query caching

## Architecture & Logic

### RAG Pipeline Flow
```
User Query: "Find news about AAPL price movement"
  ↓
Query Embedding (sentence-transformers)
  ↓
Vector Search in Milvus (top-k=10)
  ↓
Retrieve news articles + metadata
  ↓
Rerank by relevance (optional)
  ↓
Format context for LLM
  ↓
Send to vLLM inference service
  ↓
Parse LLM output (sentiment scores)
  ↓
Return structured response
```

## Key Components

### 1. Query Construction
```python
# Logic
def build_query(ticker, time_window="24h"):
    query_text = f"News and analysis about {ticker} stock price movement, earnings, and market sentiment"

    filters = {
        "tickers": ticker,
        "timestamp": last_24_hours(),
        "credibility_score": > 0.5
    }

    return query_text, filters
```

### 2. Vector Search in Milvus
```
Search Parameters:
- metric_type: "IP" (Inner Product, for cosine similarity)
- top_k: 10 (retrieve 10 most similar)
- nprobe: 16 (HNSW search parameter)
- ef: 64 (search quality vs speed)

Output: List of (text, score, metadata) tuples
```

**Why top-k=10?**
- Balances context richness vs LLM token limit
- 10 articles ≈ 2000-3000 tokens
- More articles = diminishing returns

### 3. Reranking (Optional)
```
Initial Results (10 articles)
  ↓
Cross-Encoder Reranking
  ↓
Final Top-5 (highest relevance)
```

**Cross-Encoder Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- More accurate than bi-encoder
- Slower (pairwise comparison)
- Use only for top candidates

**When to rerank?**
- User queries (high importance, low volume)
- Skip for automated pipeline (too slow)

### 4. Context Formatting
```python
# Pseudo-code for context assembly
context = ""
for i, article in enumerate(top_k_results):
    context += f"""
    Article {i+1} (Source: {article.source}, Score: {article.credibility_score})
    Published: {article.timestamp}
    Tickers: {', '.join(article.tickers)}

    {article.text}

    ---
    """
```

### 5. LLM Prompt Template
```
System: You are a financial analyst AI specializing in market sentiment analysis.

Context:
{retrieved_articles}

Task: Analyze the above news articles and provide the following scores for {ticker}:

1. market_sentiment_score: -1 (very bearish) to +1 (very bullish)
2. fear_greed_score: 0 (extreme fear) to 100 (extreme greed)
3. upside_catalyst_rating: 0-10 (likelihood of positive catalysts)
4. downside_risk_rating: 0-10 (likelihood of negative events)
5. event_importance_score: 0-10 (significance of recent events)
6. sector_impact: 0-10 (broader sector implications)

Output as JSON only.
```

## LangChain Integration

### Chain Structure
```
QueryEmbedding
  ↓
MilvusRetriever (custom LangChain retriever)
  ↓
ContextFormatter
  ↓
LLMChain (vLLM backend)
  ↓
OutputParser (JSON parsing)
```

### Custom Retriever Class
```python
# Conceptual implementation
class MilvusRetriever(BaseRetriever):
    def get_relevant_documents(self, query):
        # Embed query
        query_vector = embedder.encode(query)

        # Search Milvus
        results = milvus_collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "IP", "nprobe": 16},
            limit=10,
            filter=filters
        )

        # Convert to LangChain Document format
        return [Document(page_content=r.text, metadata=r.metadata) for r in results]
```

## Performance Optimization

### 1. Query Caching
```
Cache Key: hash(ticker + time_window + top_k)
Cache TTL: 5 minutes (for real-time), 1 hour (for historical)

Hit rate: 40-60% (repeated queries for same stocks)
Speedup: 100x (Redis lookup vs Milvus search + LLM inference)
```

### 2. Parallel Retrieval
```python
# For multiple tickers
tickers = ["AAPL", "TSLA", "NVDA"]

with ThreadPoolExecutor(max_workers=3) as executor:
    results = executor.map(retrieve_and_analyze, tickers)
```

### 3. Milvus Connection Pooling
- Pool size: 10 connections
- Timeout: 10 seconds
- Retry: 3 attempts

## Hallucination Prevention

### Techniques Used

1. **Grounding in Retrieved Context**
   - LLM instruction: "Base your analysis ONLY on the provided articles"
   - Detect hallucinations: Check if LLM mentions facts not in context

2. **Low Temperature**
   - temperature=0.3 (vs 0.7-1.0 for creative tasks)
   - Reduces randomness, more factual outputs

3. **Structured Output**
   - JSON schema enforcement
   - Easier to validate (check ranges, data types)

4. **Fact-Checking Layer (Optional)**
   ```
   LLM Output: "AAPL earnings increased 50%"
     ↓
   Check against retrieved articles
     ↓
   If not found: Flag as potential hallucination
   ```

5. **Citation Tracking**
   - Ask LLM to cite article numbers
   - Example: "Sentiment is bullish [Article 2, 5]"

## Failure Handling

### No Relevant Results Found
```
If Milvus returns 0 results:
  1. Expand time window (24h → 7 days)
  2. Broaden query (ticker + sector news)
  3. Fallback: Return neutral scores
```

### LLM Inference Failure
```
If vLLM service down:
  1. Retry with exponential backoff (3 attempts)
  2. Fallback to cached results (if available)
  3. Return error status to prediction model
  4. Prediction model proceeds without LLM scores
```

### Milvus Connection Issues
```
Circuit Breaker Pattern:
- After 5 failures in 1 minute → Open circuit
- Return cached/default scores for 30 seconds
- Try one request (half-open)
- If success → Close circuit, resume normal operation
```

## Evaluation Metrics

### Retrieval Quality
- **Precision@10**: % of retrieved articles actually relevant
- **Recall@10**: % of relevant articles retrieved
- **NDCG@10**: Normalized Discounted Cumulative Gain (ranking quality)

**Target Metrics:**
- Precision: >80%
- Recall: >70%
- NDCG: >0.85

### End-to-End Quality
- **Sentiment Accuracy**: Compare LLM scores vs human labels
- **Latency**: p95 < 2 seconds (retrieval + inference)
- **Consistency**: Same query → similar scores (low variance)

## Interview Q&A

**Q: Why LangChain over custom implementation?**
A: LangChain provides:
- Pre-built components (retrievers, chains, memory)
- Easy integration with multiple LLMs/vector DBs
- Community support and best practices
- Faster prototyping

Alternative: Custom implementation for full control (more complex).

**Q: How do you handle conflicting information in retrieved articles?**
A:
1. LLM sees all articles (doesn't filter)
2. Instruction: "Weigh articles by credibility_score"
3. LLM naturally synthesizes conflicting views
4. Output includes confidence/uncertainty

**Q: What if retrieved articles are outdated?**
A: Milvus filter: `timestamp > (now - time_window)`. Default 24h, expandable to 7 days if no results.

**Q: How do you prevent prompt injection attacks?**
A:
1. Sanitize user input (remove special tokens)
2. Fixed prompt template (user input in designated slots)
3. Validate LLM output format (JSON schema)
4. No eval() or exec() on LLM output

**Q: RAG vs Fine-tuning?**
A:
- **RAG**: Fresh data, no retraining, interpretable sources
- **Fine-tuning**: Bakes knowledge into model, expensive to update
- Choice: RAG for real-time news (data changes daily)

**Q: How to measure hallucination rate?**
A:
1. Sample 100 LLM outputs
2. Human review: Mark hallucinations
3. Calculate: hallucinations / total claims
4. Target: <5% hallucination rate

**Q: What if Milvus vector DB grows too large?**
A:
1. Partition by date
2. Drop old partitions (>90 days)
3. Archive to cheaper storage
4. Tiered search: recent (fast) + archive (slower)

**Q: Can you explain the top-k vs similarity threshold trade-off?**
A:
- **top-k=10**: Always get 10 results (even if low quality)
- **threshold=0.7**: Only high-quality matches (might get 0-20 results)
- Choice: Use top-k + post-filter by threshold

**Q: How do you version RAG pipelines?**
A:
1. Version embedding model (affects all queries)
2. Version Milvus collection (separate collections for A/B test)
3. Version prompt templates (Git + semantic versioning)
4. Shadow mode: Run new version in parallel, compare outputs
