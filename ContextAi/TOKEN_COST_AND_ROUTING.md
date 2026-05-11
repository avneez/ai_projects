# Token Calculation, Cost Tracking & Intelligent LLM Routing

> **Interview Preparation Guide**: Deep dive into token counting, cost calculation, and intelligent routing strategies for multi-LLM platforms

---

## Table of Contents

1. [Token Calculation](#token-calculation)
2. [Cost Calculation & Tracking](#cost-calculation--tracking)
3. [Intelligent LLM Routing](#intelligent-llm-routing)
4. [Interview Questions & Answers](#interview-questions--answers)

---

## Token Calculation

### Why Token Counting Matters

**Interview Question**: *"Why do we need accurate token counting?"*

**Answer**:
1. **Billing Accuracy**: LLM providers charge per token, not per character
2. **Cost Prediction**: Estimate costs BEFORE making API calls
3. **Quota Management**: Track tenant usage against their limits
4. **Performance Optimization**: Shorter prompts = faster responses + lower costs
5. **Provider Differences**: Each provider tokenizes text differently

---

### How Different Providers Tokenize

**Key Concept**: Different LLMs use different tokenization algorithms.

| Provider | Tokenizer | Example |
|----------|-----------|---------|
| **OpenAI** | `tiktoken` (BPE) | "Hello World" → `['Hello', ' World']` (2 tokens) |
| **Anthropic** | Custom BPE | "Hello World" → `['Hello', ' World']` (2 tokens, similar) |
| **Hugging Face** | Model-specific | Varies by model (BERT, GPT-2, etc.) |
| **Rule of Thumb** | N/A | 1 token ≈ 4 characters (English) |

**Why This Matters**:
- OpenAI's "Hello, how are you?" = ~6 tokens
- Same text in Anthropic ≈ 5-7 tokens (slightly different)
- Inaccurate counting → billing errors (could lose 10-15% in revenue)

---

### Token Counting Implementation

```python
import tiktoken
from anthropic import Anthropic
from transformers import AutoTokenizer

class TokenCounter:
    """
    Provider-specific token counting for billing accuracy
    """

    def __init__(self):
        # OpenAI uses tiktoken library
        self.openai_encoders = {
            "gpt-3.5-turbo": tiktoken.encoding_for_model("gpt-3.5-turbo"),
            "gpt-4": tiktoken.encoding_for_model("gpt-4"),
            "gpt-4-turbo": tiktoken.encoding_for_model("gpt-4-turbo")
        }

        # Anthropic client
        self.anthropic_client = Anthropic()

        # Hugging Face tokenizers cache
        self.hf_tokenizers = {}

    def count_tokens(self, text: str, provider: str, model: str) -> int:
        """
        Count tokens using provider-specific tokenizer
        """
        if provider == "openai":
            encoder = self.openai_encoders.get(model)
            tokens = encoder.encode(text)
            return len(tokens)

        elif provider == "anthropic":
            # Anthropic provides count_tokens API
            return self.anthropic_client.count_tokens(text)

        elif provider == "huggingface":
            # Load model-specific tokenizer
            if model not in self.hf_tokenizers:
                self.hf_tokenizers[model] = AutoTokenizer.from_pretrained(model)

            tokens = self.hf_tokenizers[model].encode(text)
            return len(tokens)

        else:
            # Fallback: approximate
            return len(text) // 4  # 1 token ≈ 4 chars
```

**Interview Tip**: Mention that you **verify token counts** against provider responses:

```python
# After API call, compare our count vs provider's reported count
our_count = token_counter.count_tokens(prompt, provider, model)
provider_count = response["usage"]["prompt_tokens"]

if abs(our_count - provider_count) > 5:  # 5 token tolerance
    logger.warning(f"Token mismatch: ours={our_count}, theirs={provider_count}")
    # Use provider's count for billing (always trust the source)
```

---

## Cost Calculation & Tracking

### Pricing Structure (Real-World Example)

**Interview Question**: *"How do you calculate costs for LLM requests?"*

**Answer**: Costs depend on:
1. **Provider pricing** (varies by model)
2. **Token type** (prompt tokens vs completion tokens have different prices)
3. **Platform markup** (our 30% fee)
4. **Infrastructure costs** (fixed cost per request)

**Actual Pricing (as of Jan 2025)**:

| Provider | Model | Prompt Price | Completion Price |
|----------|-------|--------------|------------------|
| OpenAI | GPT-3.5-Turbo | $0.0015 / 1K tokens | $0.002 / 1K tokens |
| OpenAI | GPT-4 | $0.03 / 1K tokens | $0.06 / 1K tokens |
| OpenAI | GPT-4-Turbo | $0.01 / 1K tokens | $0.03 / 1K tokens |
| Anthropic | Claude 3 Opus | $0.015 / 1K tokens | $0.075 / 1K tokens |
| Anthropic | Claude 3 Sonnet | $0.003 / 1K tokens | $0.015 / 1K tokens |
| Anthropic | Claude 3 Haiku | $0.00025 / 1K tokens | $0.00125 / 1K tokens |

**Why Different Prices**:
- **Prompt tokens**: Input processing
- **Completion tokens**: Output generation (more compute-intensive)
- Completion tokens typically **2-5x more expensive** than prompt tokens

---

### Cost Calculation Engine

```python
from decimal import Decimal

class CostCalculator:
    """
    Calculate exact costs with breakdown
    """

    # Pricing table (updated monthly)
    PRICING = {
        "openai": {
            "gpt-3.5-turbo": {
                "prompt": Decimal("0.0015"),
                "completion": Decimal("0.002")
            },
            "gpt-4": {
                "prompt": Decimal("0.03"),
                "completion": Decimal("0.06")
            }
        },
        "anthropic": {
            "claude-3-sonnet": {
                "prompt": Decimal("0.003"),
                "completion": Decimal("0.015")
            }
        }
    }

    # ContextAI fees
    PLATFORM_MARKUP = Decimal("0.30")  # 30%
    INFRASTRUCTURE_COST = Decimal("0.0001")  # $0.0001 per request

    def calculate_cost(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> dict:
        """
        Calculate total cost with breakdown

        Example:
            Input: 50 prompt tokens, 100 completion tokens (GPT-3.5)

            Calculation:
            - Prompt cost = (50 / 1000) × $0.0015 = $0.000075
            - Completion cost = (100 / 1000) × $0.002 = $0.0002
            - Provider total = $0.000275
            - Platform markup = $0.000275 × 0.30 = $0.0000825
            - Infrastructure = $0.0001
            - Total = $0.0003575 (~$0.00036)
        """
        pricing = self.PRICING[provider][model]

        # Step 1: Calculate provider costs
        prompt_cost = (Decimal(prompt_tokens) / 1000) * pricing["prompt"]
        completion_cost = (Decimal(completion_tokens) / 1000) * pricing["completion"]
        provider_cost = prompt_cost + completion_cost

        # Step 2: Add platform markup
        platform_markup = provider_cost * self.PLATFORM_MARKUP

        # Step 3: Add infrastructure cost
        infrastructure_cost = self.INFRASTRUCTURE_COST

        # Step 4: Calculate total
        total_cost = provider_cost + platform_markup + infrastructure_cost

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "provider_cost_usd": float(provider_cost),
            "platform_markup_usd": float(platform_markup),
            "infrastructure_cost_usd": float(infrastructure_cost),
            "total_cost_usd": float(total_cost),
            "breakdown": {
                "prompt_cost": float(prompt_cost),
                "completion_cost": float(completion_cost),
                "rate_per_1k_prompt": float(pricing["prompt"]),
                "rate_per_1k_completion": float(pricing["completion"])
            }
        }
```

**Example Calculation**:

```python
# Example: GPT-3.5-Turbo request
cost = calculator.calculate_cost(
    provider="openai",
    model="gpt-3.5-turbo",
    prompt_tokens=150,
    completion_tokens=300
)

# Output:
{
    "prompt_tokens": 150,
    "completion_tokens": 300,
    "provider_cost_usd": 0.000825,      # Base cost
    "platform_markup_usd": 0.0002475,   # 30% markup
    "infrastructure_cost_usd": 0.0001,  # Fixed fee
    "total_cost_usd": 0.0011725,        # Total charged to customer
    "breakdown": {
        "prompt_cost": 0.000225,        # 150 tokens × $0.0015
        "completion_cost": 0.0006,      # 300 tokens × $0.002
        "rate_per_1k_prompt": 0.0015,
        "rate_per_1k_completion": 0.002
    }
}
```

---

### Real-Time Usage Tracking

**Interview Question**: *"How do you track usage in real-time for thousands of concurrent requests?"*

**Answer**: **Dual-storage architecture**:
1. **Redis** → Fast, in-memory (for real-time quota checks)
2. **PostgreSQL** → Durable, queryable (for billing and analytics)

```python
class UsageTracker:
    """
    Track usage with dual write pattern
    """

    async def track_usage(
        self,
        tenant_id: str,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float
    ):
        """
        Write to both Redis and PostgreSQL
        """
        total_tokens = prompt_tokens + completion_tokens

        # 1. UPDATE REDIS (real-time aggregates)
        pipe = redis_client.pipeline()

        # Daily counters (key format: usage:{tenant}:metric:daily:{date})
        date_key = datetime.utcnow().strftime("%Y-%m-%d")
        pipe.incrbyfloat(f"usage:{tenant_id}:cost:daily:{date_key}", cost_usd)
        pipe.incrby(f"usage:{tenant_id}:tokens:daily:{date_key}", total_tokens)
        pipe.incrby(f"usage:{tenant_id}:requests:daily:{date_key}", 1)

        # Set TTL (auto-expire after 7 days)
        pipe.expire(f"usage:{tenant_id}:cost:daily:{date_key}", 604800)

        # Monthly counters (for billing)
        month_key = datetime.utcnow().strftime("%Y-%m")
        pipe.incrbyfloat(f"usage:{tenant_id}:cost:monthly:{month_key}", cost_usd)

        # Execute atomically
        await pipe.execute()

        # 2. WRITE TO POSTGRESQL (durable storage)
        await db.execute("""
            INSERT INTO usage_logs (
                tenant_id, timestamp, provider, model,
                prompt_tokens, completion_tokens, total_tokens,
                cost_usd
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """,
            tenant_id, datetime.utcnow(), provider, model,
            prompt_tokens, completion_tokens, total_tokens, cost_usd
        )
```

**Why This Architecture?**

| Need | Solution | Reason |
|------|----------|--------|
| **Fast quota check** | Redis `GET` | <5ms response time |
| **Billing accuracy** | PostgreSQL | ACID compliance, no data loss |
| **Real-time dashboard** | Redis aggregates | No DB load |
| **Historical analytics** | PostgreSQL queries | Complex aggregations, joins |

**Interview Example**:

```
Interviewer: "What if Redis crashes during high load?"

You: "Great question! We handle this with:

1. Redis Persistence: AOF (Append-Only File) writes every operation to disk
2. Redis Cluster: 3 masters + 3 replicas (high availability)
3. Graceful Degradation: If Redis is down, we:
   - Continue processing requests
   - Write directly to PostgreSQL
   - Use in-memory caching temporarily
   - Alert operations team
4. Recovery: When Redis returns, we rebuild counters from PostgreSQL:

   SELECT tenant_id, SUM(cost_usd)
   FROM usage_logs
   WHERE DATE(timestamp) = CURRENT_DATE
   GROUP BY tenant_id
```

---

### Quota Enforcement

```python
async def check_quota(tenant_id: str) -> bool:
    """
    Check if tenant has exceeded their quota
    Returns True if within quota, raises exception if exceeded
    """
    # Get current usage from Redis (fast)
    date_key = datetime.utcnow().strftime("%Y-%m-%d")
    daily_tokens = int(redis_client.get(f"usage:{tenant_id}:tokens:daily:{date_key}") or 0)
    daily_cost = float(redis_client.get(f"usage:{tenant_id}:cost:daily:{date_key}") or 0)

    # Get tenant's plan and quota
    tenant = await db.fetchrow("SELECT plan, quota_tokens FROM tenants WHERE id = $1", tenant_id)

    # Define quotas per plan
    QUOTAS = {
        "free": {"tokens": 100_000, "cost": 0},  # $0 spend, hard limit
        "pro": {"tokens": 5_000_000, "cost": 99},  # $99/month base
        "enterprise": {"tokens": None, "cost": None}  # Unlimited
    }

    quota = QUOTAS[tenant["plan"]]

    # Check token limit
    if quota["tokens"] and daily_tokens >= quota["tokens"]:
        raise HTTPException(
            status_code=402,  # Payment Required
            detail=f"Daily quota exceeded: {daily_tokens:,} / {quota['tokens']:,} tokens"
        )

    # Warn if approaching limit (90%)
    if quota["tokens"] and daily_tokens >= quota["tokens"] * 0.9:
        logger.warning(f"Tenant {tenant_id} at 90% quota: {daily_tokens:,} tokens")
        # Send alert email/webhook

    return True
```

---

## Intelligent LLM Routing

### The Routing Problem

**Interview Question**: *"You have 4 different LLM providers (OpenAI, Anthropic, Cohere, Hugging Face). How do you decide which one to use for a given request?"*

**Answer**: We use **intelligent routing** based on multiple factors:

1. **User Preference** (priority-based)
2. **Cost Optimization** (cheapest that meets quality bar)
3. **Latency Optimization** (fastest response)
4. **Quality Optimization** (best model for task type)
5. **Smart Routing** (ML-based multi-factor optimization)

---

### Routing Strategy 1: Priority-Based (Simplest)

**Use Case**: Tenant prefers OpenAI, but wants Anthropic as backup

```python
# Tenant configuration
tenant_providers = [
    {"provider": "openai", "model": "gpt-4", "priority": 1},
    {"provider": "anthropic", "model": "claude-3-sonnet", "priority": 2}
]

async def priority_routing(providers):
    """
    Try providers in priority order
    """
    # Sort by priority (highest first)
    sorted_providers = sorted(providers, key=lambda p: p["priority"], reverse=True)

    for provider in sorted_providers:
        # Check if provider is healthy (circuit breaker)
        if is_healthy(provider["provider"]):
            return provider

    raise Exception("No healthy providers available")
```

**Flow Diagram**:
```
Request → Check Priority 1 (OpenAI) → Healthy?
                                         ↓ Yes
                                      Use OpenAI
                                         ↓ No (503 error)
                                    Check Priority 2 (Anthropic) → Use Anthropic
```

---

### Routing Strategy 2: Cost-Optimized

**Use Case**: Tenant wants cheapest option that meets minimum quality threshold

**Key Insight**: Different models have vastly different pricing:
- GPT-4: $0.03/1K prompt tokens
- GPT-3.5-Turbo: $0.0015/1K prompt tokens
- Claude Haiku: $0.00025/1K prompt tokens

**20x price difference!**

```python
async def cost_optimized_routing(providers, prompt, max_tokens):
    """
    Select cheapest provider that meets quality requirements
    """
    MIN_QUALITY_SCORE = 0.75  # Don't use terrible models

    candidates = []

    for provider_config in providers:
        provider = provider_config["provider"]
        model = provider_config["model"]

        # Skip unhealthy providers
        if not is_healthy(provider):
            continue

        # Estimate cost BEFORE calling API
        cost = estimate_cost(provider, model, prompt, max_tokens)

        # Get quality score from historical data
        quality = get_quality_score(provider, model)

        # Only consider if quality is acceptable
        if quality >= MIN_QUALITY_SCORE:
            candidates.append({
                "provider": provider,
                "model": model,
                "cost": cost,
                "quality": quality
            })

    # Select cheapest
    if candidates:
        return min(candidates, key=lambda c: c["cost"])
    else:
        raise Exception("No providers meet quality threshold")
```

**Example**:

```python
# Prompt: "What is 2+2?"
# Candidates:
# - GPT-4: $0.003, quality: 0.95
# - GPT-3.5: $0.0002, quality: 0.90
# - Claude Haiku: $0.00005, quality: 0.85

# Result: Claude Haiku selected (60x cheaper, quality still acceptable)
```

**Interview Insight**:
- For **simple queries** (math, facts), use cheaper models
- For **complex reasoning** (code review, analysis), use GPT-4
- **Cost savings**: 40-60% on average

---

### Routing Strategy 3: Latency-Optimized

**Use Case**: Real-time chatbot needs <500ms responses

**Challenge**: Different providers have different latencies:
- OpenAI GPT-3.5: ~800ms (p95)
- Anthropic Claude Haiku: ~600ms (p95)
- Hugging Face (self-hosted): ~300ms (p95)

```python
async def latency_optimized_routing(providers):
    """
    Select fastest provider based on historical p95 latency
    """
    provider_latencies = []

    for config in providers:
        provider = config["provider"]
        model = config["model"]

        if not is_healthy(provider):
            continue

        # Query Prometheus for p95 latency (last 5 minutes)
        latency_p95 = await get_latency_from_prometheus(provider, model)

        provider_latencies.append({
            "provider": provider,
            "model": model,
            "latency_ms": latency_p95
        })

    # Select fastest
    return min(provider_latencies, key=lambda p: p["latency_ms"])
```

**Prometheus Query**:
```promql
histogram_quantile(0.95,
  rate(llm_request_duration_seconds_bucket{
    provider="openai",
    model="gpt-3.5-turbo"
  }[5m])
) * 1000  # Convert to milliseconds
```

---

### Routing Strategy 4: Quality-Optimized

**Use Case**: Critical business task needs best possible output

**Task-Specific Quality Scores**:

| Model | Coding | Analysis | Creative Writing | Math |
|-------|--------|----------|------------------|------|
| GPT-4 | 0.95 | 0.90 | 0.85 | 0.92 |
| Claude Opus | 0.88 | 0.95 | 0.90 | 0.85 |
| GPT-3.5 | 0.75 | 0.70 | 0.72 | 0.80 |

```python
async def quality_optimized_routing(providers, prompt):
    """
    Select best model based on task type
    """
    # Analyze prompt to detect task type
    task_type = analyze_task_type(prompt)

    # Task type detection
    # "Write a Python function" → task_type = "coding"
    # "Analyze this financial report" → task_type = "analysis"
    # "Write a poem" → task_type = "creative"

    # Quality scores (from benchmarks + our historical data)
    QUALITY_SCORES = {
        "coding": {
            "gpt-4": 0.95,
            "claude-opus": 0.88,
            "gpt-3.5": 0.75
        },
        "analysis": {
            "claude-opus": 0.95,
            "gpt-4": 0.90,
            "gpt-3.5": 0.70
        }
    }

    # Score each provider
    scored_providers = []
    for config in providers:
        model_key = config["model"]
        score = QUALITY_SCORES.get(task_type, {}).get(model_key, 0.5)

        scored_providers.append({
            "provider": config["provider"],
            "model": config["model"],
            "quality_score": score
        })

    # Select highest quality
    return max(scored_providers, key=lambda p: p["quality_score"])
```

**Task Detection Logic**:

```python
def analyze_task_type(prompt: str) -> str:
    """
    Detect task type from prompt keywords
    """
    prompt_lower = prompt.lower()

    # Coding keywords
    if any(kw in prompt_lower for kw in ["code", "function", "python", "javascript", "debug"]):
        return "coding"

    # Analysis keywords
    if any(kw in prompt_lower for kw in ["analyze", "explain", "summarize", "review"]):
        return "analysis"

    # Creative keywords
    if any(kw in prompt_lower for kw in ["write", "create", "story", "poem", "creative"]):
        return "creative"

    # Math keywords
    if any(kw in prompt_lower for kw in ["calculate", "solve", "equation", "math"]):
        return "math"

    return "general"
```

---

### Routing Strategy 5: Smart Routing (ML-Based)

**Use Case**: Balance cost, latency, AND quality

**Algorithm**: Weighted scoring system

```python
async def smart_routing(providers, prompt, user_weights):
    """
    Multi-factor optimization using weighted scoring

    User can specify preferences:
    - "I care 50% about cost, 30% about quality, 20% about latency"
    """

    # Default weights if not specified
    weights = {
        "cost": user_weights.get("cost_weight", 0.3),      # 30%
        "latency": user_weights.get("latency_weight", 0.3), # 30%
        "quality": user_weights.get("quality_weight", 0.4)  # 40%
    }

    candidates = []

    for config in providers:
        provider = config["provider"]
        model = config["model"]

        if not is_healthy(provider):
            continue

        # Gather metrics
        cost = estimate_cost(provider, model, prompt, max_tokens=1000)
        latency_ms = get_latency_p95(provider, model)
        quality = get_quality_score(provider, model)

        # Normalize scores to 0-1 (higher is better)
        # For cost and latency, lower is better, so invert

        cost_score = 1 / (1 + cost * 1000)  # Normalize cost
        latency_score = 1 / (1 + latency_ms / 1000)  # Normalize latency
        quality_score = quality  # Already 0-1

        # Calculate weighted total
        total_score = (
            weights["cost"] * cost_score +
            weights["latency"] * latency_score +
            weights["quality"] * quality_score
        )

        candidates.append({
            "provider": provider,
            "model": model,
            "total_score": total_score,
            "breakdown": {
                "cost": cost,
                "latency_ms": latency_ms,
                "quality": quality
            }
        })

    # Select highest scoring option
    return max(candidates, key=lambda c: c["total_score"])
```

**Example Calculation**:

```python
# Prompt: "Explain quantum computing"
# User weights: cost=0.5, latency=0.2, quality=0.3

# Candidate 1: GPT-4
# - Cost: $0.05 → cost_score = 1/(1+0.05*1000) = 0.0196
# - Latency: 1200ms → latency_score = 1/(1+1200/1000) = 0.4545
# - Quality: 0.95 → quality_score = 0.95
# Total = 0.5*0.0196 + 0.2*0.4545 + 0.3*0.95 = 0.0098 + 0.0909 + 0.285 = 0.3857

# Candidate 2: Claude Sonnet
# - Cost: $0.01 → cost_score = 0.0909
# - Latency: 900ms → latency_score = 0.5263
# - Quality: 0.85 → quality_score = 0.85
# Total = 0.5*0.0909 + 0.2*0.5263 + 0.3*0.85 = 0.0455 + 0.1053 + 0.255 = 0.4058

# Result: Claude Sonnet wins! (0.4058 > 0.3857)
```

---

### Prompt Analysis for Better Routing

```python
def analyze_prompt(prompt: str) -> dict:
    """
    Analyze prompt characteristics to inform routing
    """
    word_count = len(prompt.split())
    char_count = len(prompt)

    # Estimate complexity
    complexity = "simple"
    if word_count > 100:
        complexity = "medium"
    if word_count > 500:
        complexity = "complex"

    # Detect task type (as shown before)
    task_type = analyze_task_type(prompt)

    # Estimate required tokens
    estimated_tokens = char_count // 4

    return {
        "word_count": word_count,
        "char_count": char_count,
        "complexity": complexity,
        "task_type": task_type,
        "estimated_tokens": estimated_tokens
    }
```

**Routing Decision Based on Analysis**:

```python
prompt_analysis = analyze_prompt(prompt)

if prompt_analysis["complexity"] == "simple":
    # Use cheapest model (GPT-3.5 or Claude Haiku)
    routing_strategy = "cost_optimized"

elif prompt_analysis["task_type"] == "coding":
    # Use best coding model (GPT-4)
    routing_strategy = "quality_optimized"

elif prompt_analysis["estimated_tokens"] > 5000:
    # Long prompt, use model with large context window
    routing_strategy = "context_window_optimized"

else:
    # Default: balanced approach
    routing_strategy = "smart_routing"
```

---

### Circuit Breaker Pattern

**Problem**: What if OpenAI goes down?

**Solution**: Circuit breaker automatically stops sending requests to failing providers

```python
class CircuitBreaker:
    """
    Prevent sending requests to failing providers

    States:
    - CLOSED: Normal operation
    - OPEN: Provider failing, don't send requests
    - HALF_OPEN: Testing if provider recovered
    """

    def __init__(self):
        self.failure_threshold = 5  # Open after 5 failures
        self.recovery_timeout = 60  # Try again after 60 seconds

    async def call_with_circuit_breaker(self, provider: str, func):
        """
        Execute function with circuit breaker protection
        """
        state = await redis_client.get(f"circuit:{provider}:state") or "CLOSED"

        if state == "OPEN":
            # Check if recovery timeout expired
            opened_at = await redis_client.get(f"circuit:{provider}:opened_at")
            if time.time() - float(opened_at) > self.recovery_timeout:
                # Try half-open (test recovery)
                await redis_client.set(f"circuit:{provider}:state", "HALF_OPEN")
            else:
                # Still in timeout, skip this provider
                raise CircuitOpenException(f"{provider} circuit is open")

        try:
            # Execute the function
            result = await func()

            # Success! Reset failure count
            await redis_client.delete(f"circuit:{provider}:failures")
            await redis_client.set(f"circuit:{provider}:state", "CLOSED")

            return result

        except Exception as e:
            # Record failure
            failures = await redis_client.incr(f"circuit:{provider}:failures")

            # Open circuit if threshold exceeded
            if failures >= self.failure_threshold:
                await redis_client.set(f"circuit:{provider}:state", "OPEN")
                await redis_client.set(f"circuit:{provider}:opened_at", time.time())
                logger.error(f"Circuit opened for {provider} after {failures} failures")

            raise e
```

**Flow**:
```
Request 1 → OpenAI → Success
Request 2 → OpenAI → Success
Request 3 → OpenAI → Fail (1/5 failures)
Request 4 → OpenAI → Fail (2/5 failures)
Request 5 → OpenAI → Fail (3/5 failures)
Request 6 → OpenAI → Fail (4/5 failures)
Request 7 → OpenAI → Fail (5/5 failures) → CIRCUIT OPENS
Request 8 → OpenAI → SKIP (circuit open) → Failover to Anthropic
...
After 60 seconds → Try OpenAI again (HALF_OPEN)
```

---

### Complete Routing Flow

```python
async def route_request(
    tenant_id: str,
    prompt: str,
    user_preferences: dict
) -> dict:
    """
    Complete routing logic with all strategies
    """

    # 1. Get tenant's configured providers
    providers = await get_tenant_providers(tenant_id)

    # 2. Analyze prompt
    prompt_analysis = analyze_prompt(prompt)

    # 3. Determine routing strategy
    strategy = user_preferences.get("routing_strategy", "smart_routing")

    # 4. Apply strategy
    if strategy == "priority":
        selected = await priority_routing(providers)
    elif strategy == "cost_optimized":
        selected = await cost_optimized_routing(providers, prompt, user_preferences)
    elif strategy == "latency_optimized":
        selected = await latency_optimized_routing(providers)
    elif strategy == "quality_optimized":
        selected = await quality_optimized_routing(providers, prompt_analysis)
    else:  # smart_routing
        selected = await smart_routing(providers, prompt, user_preferences)

    # 5. Execute with circuit breaker
    try:
        result = await circuit_breaker.call_with_circuit_breaker(
            selected["provider"],
            lambda: call_llm_api(selected["provider"], selected["model"], prompt)
        )

        # 6. Track usage and cost
        cost = calculate_cost(
            selected["provider"],
            selected["model"],
            result["usage"]["prompt_tokens"],
            result["usage"]["completion_tokens"]
        )

        await track_usage(tenant_id, selected["provider"], selected["model"], cost)

        return {
            "response": result,
            "provider_used": selected["provider"],
            "model_used": selected["model"],
            "cost": cost,
            "routing_reasoning": selected.get("reasoning")
        }

    except CircuitOpenException:
        # Circuit is open, try next provider
        return await route_request_with_fallback(tenant_id, prompt, providers)
```

---

## Interview Questions & Answers

### Q1: How do you ensure billing accuracy when different providers count tokens differently?

**Answer**:

1. **Use provider-specific tokenizers** (tiktoken for OpenAI, Anthropic's API for Claude)
2. **Always verify against provider response**:
   ```python
   our_count = token_counter.count_tokens(prompt, provider, model)
   actual_count = response["usage"]["prompt_tokens"]

   # Trust provider's count, log if mismatch
   if abs(our_count - actual_count) > 5:
       logger.warning(f"Token mismatch: {our_count} vs {actual_count}")

   # Use actual count for billing
   bill_for_tokens(actual_count)
   ```
3. **Test tokenization accuracy** in CI/CD:
   ```python
   def test_token_counting():
       test_cases = [
           ("Hello world", "openai", "gpt-4", 2),
           ("The quick brown fox", "anthropic", "claude-3", 4)
       ]
       for text, provider, model, expected in test_cases:
           actual = token_counter.count_tokens(text, provider, model)
           assert abs(actual - expected) <= 1  # Allow 1 token tolerance
   ```

---

### Q2: How do you handle cost spikes when a tenant suddenly uses GPT-4 instead of GPT-3.5?

**Answer**:

1. **Real-time quota alerts**:
   ```python
   async def check_quota(tenant_id):
       daily_cost = get_daily_cost(tenant_id)
       quota = get_tenant_quota(tenant_id)

       if daily_cost > quota * 0.8:  # 80% warning
           send_alert(tenant_id, f"80% of daily quota used: ${daily_cost:.2f}")

       if daily_cost > quota:  # Hard limit
           raise QuotaExceededException()
   ```

2. **Model restrictions by plan**:
   ```python
   ALLOWED_MODELS = {
       "free": ["gpt-3.5-turbo", "claude-haiku"],
       "pro": ["gpt-3.5-turbo", "gpt-4", "claude-sonnet"],
       "enterprise": ["*"]  # All models
   }
   ```

3. **Cost prediction before request**:
   ```python
   estimated_cost = estimate_cost(provider, model, prompt, max_tokens)

   if estimated_cost > tenant_daily_budget_remaining:
       raise HTTPException(
           402,
           detail=f"Request would cost ${estimated_cost:.4f}, "
                  f"but only ${budget_remaining:.4f} remaining today"
       )
   ```

---

### Q3: Walk me through how you'd route a request: "Write a Python function to sort a list"

**Answer**:

```
Step 1: Analyze Prompt
- Task type: "coding" (detected from "Python function")
- Complexity: "simple" (short prompt, basic task)
- Estimated tokens: 50 (prompt) + 200 (completion) = 250 total

Step 2: Get Tenant Providers
- Available: [OpenAI GPT-4, OpenAI GPT-3.5, Anthropic Claude Sonnet]

Step 3: Apply Routing Strategy (assume "smart_routing")

Candidate 1: GPT-4
- Cost: 250 tokens × ($0.03 + $0.06) / 1000 = $0.0225
- Latency: 1200ms (p95)
- Quality: 0.95 (excellent for coding)
- Score: (weights: cost=0.3, latency=0.2, quality=0.5)
  = 0.3 * (1/23.5) + 0.2 * (1/2.2) + 0.5 * 0.95
  = 0.013 + 0.091 + 0.475 = 0.579

Candidate 2: GPT-3.5-Turbo
- Cost: 250 tokens × ($0.0015 + $0.002) / 1000 = $0.000875
- Latency: 800ms (p95)
- Quality: 0.80 (good enough for simple coding)
- Score: 0.3 * (1/1.875) + 0.2 * (1/1.8) + 0.5 * 0.80
  = 0.160 + 0.111 + 0.400 = 0.671

Candidate 3: Claude Sonnet
- Cost: 250 tokens × ($0.003 + $0.015) / 1000 = $0.0045
- Latency: 900ms (p95)
- Quality: 0.82
- Score: 0.3 * (1/5.5) + 0.2 * (1/1.9) + 0.5 * 0.82
  = 0.055 + 0.105 + 0.410 = 0.570

Step 4: Decision
WINNER: GPT-3.5-Turbo (highest score: 0.671)

Reasoning:
- Task is simple enough for GPT-3.5
- 25x cheaper than GPT-4
- Faster than GPT-4
- Quality adequate for basic coding task

Step 5: Execute
- Call OpenAI API with GPT-3.5-Turbo
- Track tokens: 50 prompt + 180 completion = 230 total
- Calculate actual cost: $0.00081
- Store in PostgreSQL and Redis
- Return response to user
```

---

### Q4: How do you prevent one tenant from using all your OpenAI API quota?

**Answer**:

1. **Per-tenant rate limiting**:
   ```python
   # Redis sliding window
   current_requests = redis.incr(f"rate:{tenant_id}:minute")
   if current_requests > TIER_LIMITS[tenant_plan]:
       raise RateLimitException()
   ```

2. **Per-tenant cost budgets**:
   ```python
   # Daily budget enforcement
   daily_spend = get_daily_spend(tenant_id)
   if daily_spend > tenant_budget:
       raise BudgetExceededException()
   ```

3. **Fair queuing** (if using request queue):
   ```python
   # Separate queues per tenant
   queue_name = f"llm_requests:{tenant_id}"
   celery.send_task("process_llm", queue=queue_name)

   # Worker pool allocation: 70% shared, 30% per-tenant
   ```

4. **Circuit breakers per tenant**:
   ```python
   # If tenant causes too many errors, throttle them
   error_rate = get_tenant_error_rate(tenant_id)
   if error_rate > 0.5:  # 50% errors
       apply_throttling(tenant_id, duration=300)  # 5 min penalty
   ```

---

### Q5: How do you optimize costs for a tenant making 1 million requests per month?

**Answer**:

**Step 1: Analyze Usage Patterns**
```sql
SELECT
    model,
    COUNT(*) as request_count,
    AVG(prompt_tokens) as avg_prompt_tokens,
    AVG(completion_tokens) as avg_completion_tokens,
    SUM(cost_usd) as total_cost
FROM usage_logs
WHERE tenant_id = 'tenant_xyz'
  AND timestamp > NOW() - INTERVAL '30 days'
GROUP BY model
ORDER BY total_cost DESC;
```

**Step 2: Identify Optimization Opportunities**

1. **Caching**: Detect repeated prompts
   ```python
   # 35% of prompts are duplicates
   # Implement caching → Save 35% of costs
   cache_key = hashlib.sha256(prompt.encode()).hexdigest()
   cached_response = redis.get(f"llm_cache:{cache_key}")
   ```

2. **Model downgrading**: Use cheaper models for simple tasks
   ```python
   # 60% of requests are <100 tokens (simple queries)
   # Switch from GPT-4 ($0.03) → GPT-3.5 ($0.0015)
   # Savings: 60% × 95% = 57% cost reduction
   ```

3. **Prompt compression**: Reduce token count
   ```python
   # Remove unnecessary words, use abbreviations
   # "Please write a Python function that takes a list as input" (12 tokens)
   # → "Python function: input list" (5 tokens)
   # Savings: 58% token reduction
   ```

4. **Batch processing**: Combine multiple requests
   ```python
   # Instead of 10 separate API calls, batch into 1
   # Save on per-request overhead
   ```

**Expected Results**:
- Caching: 35% cost reduction
- Model optimization: 25% cost reduction
- Prompt compression: 15% cost reduction
- **Total: 60-70% cost savings**

---

### Q6: Your system routes to OpenAI, but their API goes down. Walk me through what happens.

**Answer**:

**Timeline**:

```
14:30:00 - User makes request
14:30:01 - Router selects OpenAI (priority 1)
14:30:02 - Call OpenAI API → 503 Service Unavailable
14:30:02 - Record failure (1/5)
14:30:03 - Failover to Anthropic (priority 2)
14:30:05 - Anthropic succeeds → Return to user
14:30:05 - Log incident, increment metrics

14:31:00 - Another request arrives
14:31:01 - Try OpenAI again → 503 (2/5 failures)
14:31:02 - Failover to Anthropic → Success

... (3 more failures) ...

14:32:00 - 5th failure → Circuit breaker OPENS
14:32:00 - Emit alert: "OpenAI circuit open"
14:32:01 - All future requests skip OpenAI, go directly to Anthropic

14:35:00 - Circuit breaker timeout (60s) → HALF_OPEN
14:35:01 - Test request to OpenAI → Still failing → Re-open circuit

14:45:00 - OpenAI recovers
14:45:01 - Test request → Success!
14:45:02 - Circuit closes, resume normal operation
```

**Code Flow**:
```python
try:
    # Try primary provider
    result = await call_openai(prompt)
except Exception as e:
    # Record failure
    failures = await redis.incr("circuit:openai:failures")

    if failures >= 5:
        # Open circuit
        await redis.set("circuit:openai:state", "OPEN")
        await redis.set("circuit:openai:opened_at", time.time())

        # Alert operations
        await send_alert("OpenAI circuit opened")

    # Failover to next provider
    result = await call_anthropic(prompt)
```

**Monitoring**:
```python
# Prometheus alert
- alert: ProviderCircuitOpen
  expr: circuit_breaker_state{provider="openai"} == 1
  for: 1m
  annotations:
    summary: "OpenAI circuit breaker is open"

# Grafana dashboard shows:
# - Request success rate by provider
# - Circuit breaker state (green/red)
# - Failover count
```

---

### Q7: How would you implement A/B testing for different models?

**Answer**:

```python
async def ab_test_routing(tenant_id: str, prompt: str):
    """
    A/B test: 50% GPT-4, 50% Claude Opus
    Compare quality, latency, cost
    """

    # Consistent assignment (same user always gets same variant)
    variant = hashlib.md5(tenant_id.encode()).hexdigest()[:1]

    if int(variant, 16) % 2 == 0:
        # Variant A: GPT-4
        provider = "openai"
        model = "gpt-4"
        variant_name = "gpt4"
    else:
        # Variant B: Claude Opus
        provider = "anthropic"
        model = "claude-3-opus"
        variant_name = "claude_opus"

    # Track which variant was used
    await db.execute("""
        INSERT INTO ab_test_assignments (tenant_id, variant, timestamp)
        VALUES ($1, $2, NOW())
    """, tenant_id, variant_name)

    # Make request and track metrics
    start = time.time()
    result = await call_llm(provider, model, prompt)
    latency = time.time() - start

    # Store results for analysis
    await db.execute("""
        INSERT INTO ab_test_results (
            tenant_id, variant, latency_ms, cost_usd, tokens_used
        ) VALUES ($1, $2, $3, $4, $5)
    """, tenant_id, variant_name, latency * 1000, result["cost"], result["tokens"])

    return result

# Analysis query
"""
SELECT
    variant,
    COUNT(*) as requests,
    AVG(latency_ms) as avg_latency,
    AVG(cost_usd) as avg_cost,
    AVG(user_satisfaction_score) as satisfaction
FROM ab_test_results
GROUP BY variant;

Results:
| variant      | requests | avg_latency | avg_cost | satisfaction |
|--------------|----------|-------------|----------|--------------|
| gpt4         | 10,000   | 1200ms      | $0.025   | 4.2/5        |
| claude_opus  | 10,000   | 950ms       | $0.030   | 4.5/5        |

Conclusion: Claude Opus has higher satisfaction despite higher cost
Decision: Switch default to Claude Opus
"""
```

---

## Summary Cheat Sheet

### Token Counting
- **OpenAI**: Use `tiktoken` library
- **Anthropic**: Use `count_tokens` API
- **Rule of thumb**: 1 token ≈ 4 characters
- **Always verify**: Compare your count vs provider's response

### Cost Calculation
```
Total Cost = Provider Cost + Platform Markup + Infrastructure Cost

Provider Cost = (prompt_tokens / 1000) × prompt_rate +
                (completion_tokens / 1000) × completion_rate

Platform Markup = Provider Cost × 30%
Infrastructure = $0.0001 per request
```

### Routing Strategies
1. **Priority**: User-configured order (simplest)
2. **Cost-Optimized**: Cheapest model meeting quality threshold
3. **Latency-Optimized**: Fastest model based on p95 latency
4. **Quality-Optimized**: Best model for task type
5. **Smart Routing**: Weighted multi-factor optimization

### Circuit Breaker
- **Threshold**: 5 consecutive failures
- **Recovery timeout**: 60 seconds
- **States**: CLOSED → OPEN → HALF_OPEN → CLOSED

### Cost Optimization Tactics
- **Caching**: 35-45% cost savings
- **Model downgrading**: 25-60% savings (simple tasks)
- **Prompt compression**: 10-20% savings
- **Batch processing**: 5-10% savings

---

## Practice Interview Questions

1. **How do you ensure token counting accuracy across different providers?**
2. **Walk me through the cost calculation for a GPT-4 request with 100 prompt tokens and 200 completion tokens.**
3. **How would you route a request if the user says "I want the fastest response possible"?**
4. **What happens if Redis goes down during a high-traffic period?**
5. **How do you prevent a tenant from accidentally spending $10,000 in one day?**
6. **Explain the circuit breaker pattern and why it's important.**
7. **How would you optimize costs for a chatbot making 1 million requests per month?**
8. **What metrics would you track to improve routing decisions over time?**

---

**Good luck with your interview! 🚀**
