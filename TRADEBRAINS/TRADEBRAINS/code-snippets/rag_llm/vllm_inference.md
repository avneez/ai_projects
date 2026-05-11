# vLLM Inference Service

## Purpose
Fast, efficient LLM inference server for generating sentiment scores from financial news context. Optimized for production throughput.

## Libraries Used
- **vLLM** - High-throughput LLM inference engine
- **FastAPI** - REST API server
- **Ray** - Distributed serving (optional)
- **Prometheus** - Metrics and monitoring

## Why vLLM over Standard Transformers?

### Performance Comparison
| Method | Throughput | Latency (p95) | GPU Memory |
|--------|-----------|---------------|------------|
| Transformers | 10 req/sec | 2s | 16GB |
| vLLM | 240 req/sec | 200ms | 12GB |
| **Speedup** | **24x** | **10x faster** | **25% less** |

### vLLM Optimizations
1. **PagedAttention**: Efficient KV cache management (inspired by OS paging)
2. **Continuous Batching**: Dynamic batching of requests
3. **Optimized CUDA kernels**: Fused operations
4. **Quantization support**: INT8/FP16

## Model Selection

### Chosen Model: Llama 3.1 (8B)
- **Parameters**: 8 billion
- **Context length**: 8K tokens
- **Quantization**: FP16 (12GB VRAM)
- **License**: Open source (Meta)

**Why Llama 3.1?**
- Strong reasoning capabilities
- Instruction-following fine-tuned
- Open-source (no API costs)
- Good JSON output formatting

### Alternative Models

| Model | Size | Quality | Speed | Cost |
|-------|------|---------|-------|------|
| GPT-4 | - | Highest | Slow | $$$ API |
| Llama 3.1 70B | 70B | High | Medium | $ |
| **Llama 3.1 8B** | **8B** | **Good** | **Fast** | **Free** ✓ |
| Mistral 7B | 7B | Good | Fast | Free |
| Phi-3 Mini | 3.8B | Fair | Fastest | Free |

## vLLM Server Architecture

### Deployment Configuration
```bash
# Start vLLM server
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --dtype float16 \
  --port 8000
```

### Configuration Parameters Explained

**`--tensor-parallel-size 1`**
- Split model across GPUs (for large models)
- 1 = Single GPU
- 2 = Split across 2 GPUs (for 70B model)

**`--gpu-memory-utilization 0.9`**
- Use 90% of GPU memory
- More memory = more concurrent requests
- Leave 10% buffer for safety

**`--max-model-len 4096`**
- Maximum context + output tokens
- Lower = more throughput (less KV cache per request)

**`--dtype float16`**
- Half-precision (vs float32)
- 2x faster, 2x less memory
- Negligible quality loss for inference

## API Endpoints

### 1. Generate Sentiment Scores
```http
POST /v1/completions
Content-Type: application/json

{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "prompt": "{system_prompt}\n\n{context}\n\n{task}",
  "max_tokens": 300,
  "temperature": 0.3,
  "top_p": 0.9,
  "stop": ["</s>", "END"]
}

Response:
{
  "id": "cmpl-xxx",
  "choices": [{
    "text": "{\n  \"market_sentiment_score\": 0.7,\n  \"fear_greed_score\": 65,\n  ...\n}",
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 1200,
    "completion_tokens": 85,
    "total_tokens": 1285
  }
}
```

### 2. Health Check
```http
GET /health

Response: {"status": "healthy"}
```

### 3. Model Info
```http
GET /v1/models

Response: {
  "data": [{
    "id": "meta-llama/Llama-3.1-8B-Instruct",
    "max_model_len": 4096
  }]
}
```

## Inference Parameters

### Temperature (0.0 - 2.0)
- **Low (0.1-0.3)**: Deterministic, factual (our use case)
- **Medium (0.7)**: Balanced creativity
- **High (1.5+)**: Creative, diverse outputs

**Why 0.3 for sentiment?**
- Need consistent scoring
- Avoid random variations
- Still some flexibility for nuance

### Top-p (Nucleus Sampling)
- **0.9**: Consider top 90% probability mass
- Prevents unlikely/nonsensical tokens
- More stable than temperature alone

### Max Tokens
- **300 tokens**: Enough for JSON output (~200 tokens typical)
- Prevents runaway generation
- Faster inference (less computation)

## Performance Optimization

### 1. Continuous Batching
```
Traditional Batching:
Request 1-32 → Wait for all → Process batch → Respond
Downside: Fast requests wait for slow ones

Continuous Batching (vLLM):
Requests arrive → Process immediately
Batch dynamically as tokens generate
Requests finish independently
```

**Benefits:**
- Lower latency for individual requests
- Higher throughput overall
- No wasted GPU cycles

### 2. Quantization (INT8/AWQ)
```
FP16 (baseline): 16 GB VRAM, 50 tokens/sec
INT8: 8 GB VRAM, 90 tokens/sec (4% quality loss)
AWQ (4-bit): 4 GB VRAM, 120 tokens/sec (6% quality loss)
```

**When to use:**
- FP16: Best quality (production default)
- INT8: 2x GPU capacity, minimal loss
- AWQ: Extreme throughput, acceptable loss

### 3. Speculative Decoding
```
Draft model (small, fast) generates tokens
Verification model (large, accurate) validates
Accept correct tokens, reject wrong ones
```

**Speedup:** 2-3x for long sequences
**Tradeoff:** More complex setup

## Prompt Engineering

### System Prompt (Critical for Quality)
```
You are an expert financial analyst AI. Your role is to analyze
news articles and provide objective sentiment scores.

Rules:
1. Base analysis ONLY on provided articles
2. Maintain objectivity (no personal opinions)
3. Output valid JSON only
4. Use provided scoring scales exactly
5. If uncertain, use neutral scores (0 or 5)
```

### Few-Shot Examples (Optional)
```
Example 1:
Context: [positive earnings article]
Output: {"market_sentiment_score": 0.8, ...}

Example 2:
Context: [negative regulatory article]
Output: {"market_sentiment_score": -0.6, ...}

Now analyze this:
Context: {actual_context}
```

**Improves:**
- Consistency (follows examples)
- Accuracy (understands scoring scale)
- Format adherence (JSON structure)

## Output Parsing

### JSON Extraction
```python
# Pseudo-code
def parse_llm_output(text):
    # Extract JSON from response
    json_match = re.search(r'\{.*\}', text, re.DOTALL)

    if not json_match:
        raise ParseError("No JSON found")

    json_str = json_match.group()
    data = json.loads(json_str)

    # Validate schema
    required_fields = [
        "market_sentiment_score",
        "fear_greed_score",
        "upside_catalyst_rating",
        "downside_risk_rating",
        "event_importance_score",
        "sector_impact"
    ]

    for field in required_fields:
        if field not in data:
            raise ValidationError(f"Missing field: {field}")

        # Validate ranges
        if field == "market_sentiment_score":
            assert -1 <= data[field] <= 1
        else:
            assert 0 <= data[field] <= (100 if "greed" in field else 10)

    return data
```

## Scaling Strategies

### Horizontal Scaling
```
Load Balancer (Nginx)
  ↓
├─ vLLM Server 1 (GPU 0)
├─ vLLM Server 2 (GPU 1)
├─ vLLM Server 3 (GPU 2)
└─ vLLM Server 4 (GPU 3)
```

**Benefits:**
- 4x throughput (4 GPUs)
- Fault tolerance (1 GPU fails, others continue)
- Rolling updates (update one at a time)

### Multi-GPU on Single Instance
```bash
# Tensor parallelism (split single model)
vllm serve Llama-3.1-70B \
  --tensor-parallel-size 4  # 4 GPUs

# vs

# Pipeline parallelism (multiple models)
4 separate vLLM instances with load balancer
```

**When to use each:**
- Tensor parallel: Large model (70B) that doesn't fit on 1 GPU
- Multiple instances: Smaller models (8B), better GPU utilization

## Monitoring & Metrics

### Key Metrics (Prometheus)
```
# Throughput
vllm_requests_per_second

# Latency
vllm_request_duration_seconds{quantile="0.95"}

# GPU Utilization
vllm_gpu_memory_used_bytes
vllm_gpu_utilization_percent

# Errors
vllm_request_failures_total

# Queue Depth
vllm_queue_size
```

### Alerting Thresholds
- Latency p95 > 2s → Alert
- GPU memory > 95% → Warning
- Error rate > 5% → Critical
- Queue depth > 100 → Scale up

## Cost Optimization

### GPU Selection
| GPU | VRAM | Cost/hr | Model Size | Throughput |
|-----|------|---------|------------|------------|
| T4 | 16GB | $0.35 | 8B (INT8) | 50 req/s |
| L4 | 24GB | $0.70 | 8B (FP16) | 100 req/s |
| A10G | 24GB | $1.00 | 8B (FP16) | 120 req/s |
| **A100 40GB** | **40GB** | **$3.00** | **70B (INT8)** | **80 req/s** |
| A100 80GB | 80GB | $4.00 | 70B (FP16) | 100 req/s |

**Recommendation:** L4 for 8B model (best price/performance)

### Inference Cost per Request
```
L4 GPU: $0.70/hour = $0.000194/second
Average request: 1.5 seconds
Cost per request: $0.0003 (0.03 cents)

vs

GPT-4 API: $0.03 per 1K tokens
Average: 1.5K tokens/request
Cost per request: $0.045 (4.5 cents)

Savings: 150x cheaper
```

## Interview Q&A

**Q: Why vLLM over Text Generation Inference (TGI) or TensorRT-LLM?**
A:
- **vLLM**: Best for throughput, easy setup, Python-native
- **TGI**: Hugging Face ecosystem, good integration
- **TensorRT-LLM**: Fastest (NVIDIA-optimized), harder to set up

**Q: How do you handle model updates without downtime?**
A:
1. Blue-green deployment: New vLLM instance (green)
2. Test green instance with shadow traffic
3. Switch load balancer from blue → green
4. Keep blue running for 24h (rollback if needed)

**Q: What if LLM generates invalid JSON?**
A:
1. Retry with "strictly output valid JSON" instruction (up to 2 times)
2. If still fails: Log error, return neutral scores
3. Alert if failure rate > 5%

**Q: How to prevent model bias?**
A:
1. Use objective instruction ("analyze facts, not opinions")
2. A/B test outputs vs human labels
3. Monitor for systematic bias (always bullish/bearish)
4. Fine-tune on balanced dataset if needed

**Q: Can you switch models at runtime?**
A: Not easily. vLLM loads model at startup.
Solution: Run multiple vLLM instances with different models, route traffic via load balancer.

**Q: How do you debug slow inference?**
A:
1. Check GPU utilization (should be >80%)
2. Monitor queue depth (high = bottleneck)
3. Profile with vLLM metrics (prompt tokens, generation tokens)
4. Reduce max_tokens if too high

**Q: What's the ROI of vLLM vs API (e.g., GPT-4)?**
A:
Setup cost: $500 (engineering time) + $700/month (GPU)
API cost: $10k/month (at scale)
Break-even: 1 month
Annual savings: $100k+

**Q: How do you ensure output consistency?**
A:
1. Low temperature (0.3)
2. Seed parameter (deterministic sampling, if supported)
3. Cache identical prompts
4. Monitor output variance over time
