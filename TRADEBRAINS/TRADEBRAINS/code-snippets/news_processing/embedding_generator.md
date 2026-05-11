# News Embedding Generator

## Purpose
Convert validated news text into dense vector embeddings for storage in Milvus vector database. Enables semantic search for RAG pipeline.

## Libraries Used
- **sentence-transformers** - State-of-the-art text embeddings
- **pymilvus** - Milvus Python SDK
- **torch** - PyTorch for model inference
- **Kafka** - Message queue integration

## Model Selection

### Chosen Model: `all-MiniLM-L6-v2`
- **Embedding dimension**: 384
- **Max sequence length**: 256 tokens
- **Model size**: 80MB
- **Inference speed**: 50 texts/sec (GPU), 15 texts/sec (CPU)

**Why this model?**
- Good balance of quality vs speed
- Small enough for production deployment
- Pre-trained on 1B+ sentence pairs
- Strong performance on semantic similarity tasks

### Alternative Models Considered

| Model | Dim | Size | Speed | Use Case |
|-------|-----|------|-------|----------|
| `all-mpnet-base-v2` | 768 | 420MB | Slower | Highest quality |
| `all-MiniLM-L6-v2` | 384 | 80MB | Fast | **Production** ✓ |
| `paraphrase-MiniLM-L3-v2` | 384 | 60MB | Fastest | High-volume |

## Architecture & Logic

### 1. Text Preprocessing
```
Raw news text
  ↓
Clean HTML tags, URLs
  ↓
Normalize whitespace
  ↓
Truncate to 256 tokens (model limit)
  ↓
Ready for embedding
```

**Preprocessing steps:**
- Remove HTML: `BeautifulSoup` or regex
- Remove URLs: Keep only domain for context
- Normalize: Multiple spaces → single space
- Truncation: Keep first 256 tokens (most important info)

### 2. Batch Embedding Generation
```
Input: List of 32 texts (batch)
  ↓
Tokenization (SentenceTransformer handles internally)
  ↓
Model inference (GPU accelerated)
  ↓
Output: 32 x 384 dimensional vectors
  ↓
Normalize (L2 normalization for cosine similarity)
```

**Why batching?**
- GPU utilization: 80% vs 20% (single text)
- Throughput: 50 texts/sec (batched) vs 10 texts/sec (individual)
- Memory efficient: Shared tokenization overhead

### 3. Milvus Storage
```
Embedding vector (384-dim)
+ Metadata (ticker, timestamp, source, validation_scores)
  ↓
Insert into Milvus collection
  ↓
Indexed for fast similarity search
```

## Milvus Collection Schema

```python
# Conceptual schema
{
  "id": "primary_key (auto-generated)",
  "embedding": "float_vector[384]",
  "text": "varchar(5000)",
  "tickers": "array<varchar>",
  "timestamp": "int64 (Unix timestamp)",
  "source": "varchar (twitter/telegram/linkedin)",
  "credibility_score": "float",
  "relevance_score": "float",
  "content_hash": "varchar (for deduplication)"
}
```

### Index Configuration
- **Index type**: HNSW (Hierarchical Navigable Small World)
- **Metric**: Cosine similarity (IP after L2 normalization)
- **Parameters**: `M=16, efConstruction=200`

**Why HNSW?**
- Best accuracy/speed trade-off
- Sub-millisecond search on millions of vectors
- Good for high-dimensional data (384-dim)

**Alternative indices:**
- IVF_FLAT: Faster build, slower search
- ANNOY: Memory-efficient, less accurate

## Data Flow Pipeline

```
Kafka Consumer (validated_news topic)
  ↓
Batch Accumulator (collect 32 messages)
  ↓
Text Preprocessing (parallel with multiprocessing)
  ↓
Embedding Generation (GPU batch inference)
  ↓
Milvus Insert (batch insert for performance)
  ↓
Acknowledge Kafka offset
```

## Performance Optimization

### 1. Batching Strategy
- **Batch size**: 32 texts (GPU memory sweet spot)
- **Timeout**: 5 seconds (don't wait forever)
- **Adaptive**: Smaller batches during low traffic

### 2. GPU Optimization
- **Mixed precision**: FP16 inference (2x faster)
- **Model quantization**: INT8 (4x smaller, slight accuracy loss)
- **Multiple GPUs**: Split batches across GPUs

### 3. Multiprocessing
```
Main Process
  ↓
Fork 4 Worker Processes
  ├─ Worker 1: Preprocessing
  ├─ Worker 2: Preprocessing
  ├─ Worker 3: Embedding (GPU 0)
  └─ Worker 4: Embedding (GPU 1)
```

**Why multiprocessing?**
- Python GIL: Blocks multi-threading for CPU work
- CPU preprocessing + GPU inference in parallel
- Throughput: 200+ texts/sec (multi-GPU setup)

## Embedding Quality Assurance

### 1. Similarity Validation
```
Test: "Apple reports strong earnings"
Similar: "AAPL quarterly results beat expectations" (score: 0.87)
Dissimilar: "I ate an apple for lunch" (score: 0.23)
```

### 2. Semantic Clustering
- Related news should cluster together
- Different topics should separate
- Visualize with t-SNE/UMAP for validation

### 3. Retrieval Quality
```
Query: "Tesla earnings news"
Top results should be:
  1. Tesla Q3 earnings
  2. TSLA stock reaction to earnings
  3. Musk comments on earnings

NOT:
  - General EV news (unless Tesla-specific)
  - Unrelated Tesla news (e.g., factory opening)
```

## Storage Considerations

### Milvus Storage Costs
```
1 million news articles
× 384 dimensions
× 4 bytes (float32)
= 1.46 GB (vectors only)

With metadata: ~3 GB total
```

**Optimization:**
- Use float16: Half the size (768 MB)
- Compression: Milvus auto-compression for historical data
- Partitioning: By date (drop old partitions)

### Retention Policy
- **Hot data**: Last 7 days (fast SSD)
- **Warm data**: 8-90 days (regular disk)
- **Cold data**: >90 days (archive or delete)

## Integration Points

### Input
- **Source**: Kafka topic `validated_news`
- **Format**: JSON with text, tickers, metadata
- **Rate**: 100-1000 messages/minute

### Output
- **Destination**: Milvus collection `financial_news`
- **Indexed**: HNSW for fast search
- **Accessible**: RAG service queries this collection

## Error Handling

### Model Inference Failures
- **Retry**: 3 attempts with exponential backoff
- **Fallback**: Skip embedding, log for manual review
- **Alert**: Notify if >5% failure rate

### Milvus Connection Issues
- **Buffer**: Accumulate embeddings in memory (max 10k)
- **Persistence**: Write to disk if memory full
- **Reconnect**: Auto-reconnect with circuit breaker

## Interview Q&A

**Q: Why sentence-transformers over Word2Vec or GloVe?**
A: Word2Vec/GloVe are word-level (average to sentence = poor quality). Sentence-transformers are trained end-to-end for sentence similarity. Better semantic understanding.

**Q: How do you handle very long texts (>256 tokens)?**
A:
1. Truncation: Keep first 256 tokens (headline + intro)
2. Chunking: Split into multiple embeddings (for detailed analysis)
3. Summarization: Use LLM to summarize first (adds latency)

**Q: What if embedding quality degrades over time (domain drift)?**
A:
1. Monitor retrieval quality metrics
2. Fine-tune model on recent financial texts
3. A/B test new model vs production
4. Gradual rollout

**Q: How do you ensure embedding consistency?**
A:
1. Lock model version (avoid automatic updates)
2. Deterministic inference (disable dropout)
3. Normalize embeddings (L2 norm)
4. Version collection schema

**Q: Cosine similarity vs Euclidean distance?**
A: Cosine similarity (normalized vectors → inner product).
- Invariant to vector magnitude
- Better for text (direction matters more than length)
- Range: [-1, 1] (easier to interpret)

**Q: How to handle multilingual news?**
A:
1. Language detection → separate collections per language
2. Use `paraphrase-multilingual-MiniLM-L12-v2` (supports 50+ languages)
3. Translate to English first (adds latency)

**Q: What's the cold start problem?**
A: Empty vector DB at launch.
Solution: Backfill historical news (24-48 hours of data) before going live.

**Q: How do you debug bad retrievals?**
A:
1. Log query embedding
2. Inspect top-k results with similarity scores
3. Visualize embeddings (t-SNE)
4. Check for outliers/noise in training data

**Q: Can you update embeddings after ingestion?**
A: Yes, but expensive. Better to:
1. Recompute if model changes (batch job)
2. Keep old embeddings if text hasn't changed
3. Use versioned collections for major updates
