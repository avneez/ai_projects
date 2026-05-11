# Token Pricing & Tier System

## Token Economics

### Token-to-Dollar Conversion

```
1 Token = $0.01 USD

Token Packages:
- 100 tokens = $1.00
- 500 tokens = $4.50 (10% discount)
- 1,000 tokens = $8.00 (20% discount)
- 5,000 tokens = $35.00 (30% discount)
- 10,000 tokens = $60.00 (40% discount)
```

---

## Subscription Tiers

### FREE Tier

```yaml
Monthly Cost: $0
Monthly Token Quota: 100 tokens
Rollover: No (resets monthly)

Permissions:
  - generate_image: ✅ (10-30 tokens each)
  - generate_audio: ✅ (15-50 tokens each)
  - generate_video: ❌
  - caption_video: ❌
  - generate_avatar: ❌
  - train_lora: ❌
  - api_access: ❌

Rate Limits:
  - Requests per hour: 10
  - Concurrent jobs: 1
  - Max queue depth: 3
  - Priority: Low

Restrictions:
  - Watermark on images: Yes
  - Max image resolution: 1024x1024
  - Max audio duration: 30 seconds
  - Download quality: Compressed

What can you generate:
  - ~3-10 images per month (depending on settings)
  - ~2-6 audio clips per month
```

### PRO Tier

```yaml
Monthly Cost: $20/month
Monthly Token Quota: 1,000 tokens
Bonus: 200 tokens on signup
Rollover: Up to 500 tokens (max accumulation: 1,500)

Permissions:
  - generate_image: ✅ (10-30 tokens each)
  - generate_video: ✅ (100-500 tokens each)
  - generate_audio: ✅ (15-50 tokens each)
  - caption_video: ✅ (20-40 tokens each)
  - generate_avatar: ✅ 2D only (150 tokens each)
  - train_lora: ✅ (500 tokens per training)
  - api_access: ❌

Rate Limits:
  - Requests per hour: 100
  - Concurrent jobs: 5
  - Max queue depth: 20
  - Priority: Normal

Restrictions:
  - Watermark: Optional (can disable)
  - Max image resolution: 2048x2048
  - Max video duration: 30 seconds
  - Max audio duration: 120 seconds (2 min)
  - Download quality: High quality

What can you generate:
  - ~33-100 images per month
  - ~2-10 videos per month
  - ~20-66 audio clips per month
  - Mix and match based on needs
```

### ENTERPRISE Tier

```yaml
Monthly Cost: $200/month
Monthly Token Quota: 10,000 tokens
Bonus: 2,000 tokens on signup
Rollover: Up to 5,000 tokens (max accumulation: 15,000)

Permissions:
  - generate_image: ✅ (10-30 tokens each)
  - generate_video: ✅ (100-500 tokens each)
  - generate_audio: ✅ (15-50 tokens each)
  - caption_video: ✅ (20-40 tokens each)
  - generate_avatar: ✅ 2D and 3D (150-600 tokens each)
  - train_lora: ✅ (500 tokens per training)
  - api_access: ✅ (programmatic access)
  - batch_processing: ✅ (bulk operations)
  - custom_models: ✅ (use your fine-tuned models)
  - priority_support: ✅

Rate Limits:
  - Requests per hour: 1,000
  - Concurrent jobs: 20
  - Max queue depth: 100
  - Priority: High

Restrictions:
  - Watermark: No
  - Max image resolution: 4096x4096
  - Max video duration: 120 seconds (2 min)
  - Max audio duration: 600 seconds (10 min)
  - Download quality: Lossless/Uncompressed

What can you generate:
  - ~300-1,000 images per month
  - ~20-100 videos per month
  - ~200-666 audio clips per month
  - 6-20 custom LoRA trainings per month
  - API integration for automation
```

---

## Detailed Token Cost Table

### Text-to-Image

| Configuration | Tokens | Cost | Example Use Case |
|---------------|--------|------|------------------|
| **Quick Draft** | 5-10 | $0.05-0.10 | 512x512, SD 2.1, 20 steps |
| **Standard** | 15-30 | $0.15-0.30 | 1024x1024, SD XL, 30 steps |
| **High Quality** | 40-60 | $0.40-0.60 | 1536x1536, SD XL, 50 steps |
| **Ultra HD** | 80-120 | $0.80-1.20 | 2048x2048, SD XL Refiner, 50 steps |

**Formula:**
```python
tokens = 10 × (resolution_mult) × (steps_mult) × (model_mult) × (priority_mult)

# Example: 1024x1024, 30 steps, SD XL, normal priority
tokens = 10 × 2.0 × 1.0 × 1.5 × 1.0 = 30 tokens
```

### Text-to-Video

| Configuration | Tokens | Cost | Example Use Case |
|---------------|--------|------|------------------|
| **Short Clip (3s, 720p)** | 80-120 | $0.80-1.20 | Social media teaser |
| **Standard (5s, 720p)** | 150-200 | $1.50-2.00 | Instagram/TikTok |
| **Medium (10s, 720p)** | 250-350 | $2.50-3.50 | YouTube short |
| **HD (10s, 1080p)** | 400-500 | $4.00-5.00 | Professional content |
| **Long HD (30s, 1080p)** | 1000-1500 | $10-15 | Commercial/ad |

**Formula:**
```python
tokens = 100 × (resolution_mult) × (1 + duration × 2) × (model_mult) × (priority_mult)

# Example: 5 seconds, 720p, SD XL, normal priority
tokens = 100 × 2.0 × (1 + 5 × 2) × 1.5 × 1.0 = 100 × 2.0 × 11 × 1.5 = 330 tokens
```

### Text-to-Audio

| Configuration | Tokens | Cost | Example Use Case |
|---------------|--------|------|------------------|
| **Sound Effect (5s)** | 15-25 | $0.15-0.25 | UI sounds, alerts |
| **Short Music (15s)** | 30-50 | $0.30-0.50 | Jingles, loops |
| **Standard (30s)** | 50-80 | $0.50-0.80 | Background music |
| **Long (60s)** | 100-150 | $1.00-1.50 | Full track |
| **Extended (120s)** | 180-250 | $1.80-2.50 | Podcast intro/outro |

**Formula:**
```python
tokens = 15 × (1 + duration × 2) × (quality_mult) × (model_mult) × (priority_mult)

# Example: 30 seconds, standard quality, MusicGen Large, normal priority
tokens = 15 × (1 + 30 × 2) × 1.0 × 2.0 × 1.0 = 15 × 61 × 2.0 = 1830...
# Wait, this seems high. Let me recalculate:
# Base: 15, Duration multiplier: (duration / 10), so 30s = 3.0x
tokens = 15 × 3.0 × 1.0 × 2.0 × 1.0 = 90 tokens (more reasonable)
```

### Video Captioning

| Configuration | Tokens | Cost | Example Use Case |
|---------------|--------|------|------------------|
| **Short (30s video)** | 20-30 | $0.20-0.30 | Social media clip |
| **Medium (60s video)** | 35-50 | $0.35-0.50 | YouTube video |
| **Long (120s video)** | 60-80 | $0.60-0.80 | Tutorial/review |
| **With transcription** | +20 | +$0.20 | Audio to text |

**Formula:**
```python
tokens = 20 × (duration / 30) × (transcription_mult)

# Example: 60 second video with audio transcription
tokens = 20 × (60 / 30) × 1.5 = 20 × 2.0 × 1.5 = 60 tokens
```

### Virtual Avatar Generation

| Configuration | Tokens | Cost | Example Use Case |
|---------------|--------|------|------------------|
| **2D Portrait** | 150-200 | $1.50-2.00 | Profile picture |
| **2D Full Body** | 250-350 | $2.50-3.50 | Character design |
| **3D Head** | 400-500 | $4.00-5.00 | VR/AR avatar |
| **3D Full Body** | 600-800 | $6.00-8.00 | Game character |
| **3D Rigged** | 800-1200 | $8.00-12.00 | Animated character |
| **3D Rigged + Animated** | 1500-2000 | $15-20 | Ready-to-use avatar |

**Formula:**
```python
tokens = 150 × (dimension_mult) × (rig_mult) × (animation_mult) × (priority_mult)

# Example: 3D rigged avatar
tokens = 150 × 2.0 × 1.5 × 1.0 × 1.0 = 450 tokens
```

### LoRA Training

| Configuration | Tokens | Cost | Example Use Case |
|---------------|--------|------|------------------|
| **Quick Training (10 images)** | 300-400 | $3.00-4.00 | Face/style |
| **Standard (20 images)** | 500-600 | $5.00-6.00 | Character/object |
| **Advanced (50 images)** | 800-1000 | $8.00-10.00 | Complex style |

---

## Token Purchase Options

### One-Time Purchases (No Expiry)

```python
PURCHASE_PACKAGES = {
    "starter": {
        "tokens": 100,
        "price": 1.00,
        "discount": 0,
        "best_for": "Try before tier"
    },
    "basic": {
        "tokens": 500,
        "price": 4.50,
        "discount": 10,
        "best_for": "Occasional users"
    },
    "standard": {
        "tokens": 1000,
        "price": 8.00,
        "discount": 20,
        "best_for": "Regular users"
    },
    "premium": {
        "tokens": 5000,
        "price": 35.00,
        "discount": 30,
        "best_for": "Power users"
    },
    "ultimate": {
        "tokens": 10000,
        "price": 60.00,
        "discount": 40,
        "best_for": "Professionals"
    }
}
```

### How Purchased Tokens Work

```
User has PRO tier:
- Monthly quota: 1,000 tokens (resets monthly)
- Purchased tokens: 500 (permanent, no expiry)

Usage order:
1. First use monthly quota (1,000 tokens)
2. When quota exhausted, use purchased tokens (500 tokens)
3. When all tokens exhausted, purchase more or wait for monthly reset

Example month:
Day 1-15: Used 1,000 monthly quota tokens
Day 16-30: Using purchased tokens (500 remaining → 200 remaining)
Next month: Get fresh 1,000 quota + still have 200 purchased tokens
```

---

## Token Refund Policy

### Automatic Refunds

```yaml
Job Failed (Technical Error):
  Refund: 100% of tokens
  Timing: Immediate
  Example: GPU crashed, model loading error

Job Cancelled by User:
  Before Processing: 100% refund
  During Processing: 50% refund (resources already used)
  After Completion: No refund

Quality Issues (NSFW/Corrupted):
  Refund: 100% of tokens
  Timing: Immediate
  Review: Automatic for NSFW, manual review for quality

Duplicate Generation (Same prompt within 1 hour):
  Refund: 100% on duplicates
  Cache: Return cached result instead
```

### Manual Refund Requests

```python
# Support can issue manual refunds
async def issue_manual_refund(
    user_id: str,
    tokens: int,
    reason: str,
    approved_by: str
):
    """Issue manual refund (support/admin only)"""
    await TokenService.add_tokens(
        user_id=user_id,
        tokens_to_add=tokens,
        source=f"refund:{reason}"
    )

    # Log refund
    await db.add(ManualRefundLog(
        user_id=user_id,
        tokens=tokens,
        reason=reason,
        approved_by=approved_by,
        timestamp=datetime.utcnow()
    ))
```

---

## Token Monitoring Dashboard

### User Dashboard

```
┌─────────────────────────────────────────────────────┐
│           TOKEN BALANCE DASHBOARD                   │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Current Balance:    547 tokens   ($5.47)          │
│                                                     │
│  This Month:                                        │
│  ├─ Monthly Quota:   1,000 tokens                   │
│  ├─ Used:            453 tokens (45%)               │
│  └─ Remaining:       547 tokens                     │
│                                                     │
│  Purchased Tokens:   0 tokens                       │
│  Rollover Tokens:    0 tokens                       │
│                                                     │
│  Next Reset:         Dec 31, 2025 (10 days)         │
│                                                     │
│  ┌──────────────────────────────────────┐          │
│  │  [  Purchase More Tokens  ]          │          │
│  │  [  Upgrade to Enterprise ]          │          │
│  └──────────────────────────────────────┘          │
│                                                     │
├─────────────────────────────────────────────────────┤
│           USAGE BREAKDOWN (This Month)              │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Images:     320 tokens (15 images)    [████░░]    │
│  Videos:     100 tokens (1 video)      [█░░░░░]    │
│  Audio:      33 tokens (2 clips)       [░░░░░░]    │
│  Captions:   0 tokens                  [░░░░░░]    │
│                                                     │
├─────────────────────────────────────────────────────┤
│           RECENT ACTIVITY                           │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Dec 21, 10:30 AM  │  Image Generation  │  -30 ⚡   │
│  Dec 21, 09:15 AM  │  Video Generation  │ -100 ⚡   │
│  Dec 20, 04:20 PM  │  Image Generation  │  -25 ⚡   │
│  Dec 20, 02:10 PM  │  Token Purchase    │ +500 ⚡   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### API Endpoint for Token Info

```python
@router.get("/tokens/balance")
async def get_token_balance(user: User = Depends(get_current_user)):
    """Get user's current token balance and usage"""
    tokens = await TokenService.get_user_tokens(user.id)

    # Get usage breakdown for current month
    usage_breakdown = await db.query(
        TokenUsageLog.pipeline,
        func.sum(TokenUsageLog.tokens_used).label('total'),
        func.count(TokenUsageLog.id).label('count')
    ).filter(
        TokenUsageLog.user_id == user.id,
        TokenUsageLog.timestamp >= datetime.utcnow().replace(day=1)
    ).group_by(TokenUsageLog.pipeline).all()

    return {
        "balance": tokens["balance"],
        "monthly_quota": tokens["monthly_quota"],
        "quota_used": tokens["quota_used"],
        "quota_remaining": tokens["quota_remaining"],
        "reset_date": tokens["reset_date"],
        "tier": user.tier.value,
        "usage_breakdown": [
            {
                "pipeline": row.pipeline,
                "tokens_used": row.total,
                "generations": row.count
            }
            for row in usage_breakdown
        ]
    }
```

---

## Cost Estimation Before Generation

### Pre-Generation Cost Preview

```python
@router.post("/estimate-cost")
async def estimate_cost(
    request: EstimateCostRequest,
    user: User = Depends(get_current_user)
):
    """Estimate token cost before generation"""

    if request.pipeline == "text_to_image":
        tokens = TokenCalculator.calculate_image_cost(
            resolution=request.resolution,
            steps=request.steps,
            model=request.model,
            priority=request.priority
        )
    elif request.pipeline == "text_to_video":
        tokens = TokenCalculator.calculate_video_cost(
            resolution=request.resolution,
            duration=request.duration,
            fps=request.fps,
            model=request.model,
            priority=request.priority
        )
    # ... other pipelines

    # Check if user has enough tokens
    user_tokens = await TokenService.get_user_tokens(user.id)
    has_enough = user_tokens["balance"] >= tokens

    return {
        "tokens_required": tokens,
        "cost_usd": tokens * 0.01,
        "user_balance": user_tokens["balance"],
        "has_enough_tokens": has_enough,
        "estimated_time": get_estimated_time(request.pipeline),
        "breakdown": {
            "base_cost": TokenCostConfig.BASE_COSTS[request.pipeline],
            "multipliers": {
                "resolution": get_resolution_mult(request.resolution),
                "model": get_model_mult(request.model),
                "priority": get_priority_mult(request.priority)
            }
        }
    }
```

---

## Token Gifting & Promotions

### Referral Program

```python
REFERRAL_REWARDS = {
    "referee": 100,  # New user gets 100 tokens
    "referrer": 50   # Existing user gets 50 tokens
}

async def process_referral(referrer_id: str, referee_id: str):
    """Process referral rewards"""
    # Give tokens to new user
    await TokenService.add_tokens(
        user_id=referee_id,
        tokens_to_add=REFERRAL_REWARDS["referee"],
        source="referral_signup"
    )

    # Give tokens to referrer
    await TokenService.add_tokens(
        user_id=referrer_id,
        tokens_to_add=REFERRAL_REWARDS["referrer"],
        source="referral_reward"
    )
```

### Promotional Events

```python
# Holiday promotion: 2x tokens on all purchases
@router.post("/purchase-tokens")
async def purchase_tokens(
    package: str,
    user: User = Depends(get_current_user)
):
    package_info = PURCHASE_PACKAGES[package]
    tokens = package_info["tokens"]

    # Check for active promotions
    active_promo = await get_active_promotion()
    if active_promo and active_promo.type == "multiplier":
        tokens = int(tokens * active_promo.multiplier)
        bonus_tokens = tokens - package_info["tokens"]

    # Process payment (Stripe, etc.)
    payment = await process_payment(user.id, package_info["price"])

    if payment.status == "succeeded":
        # Add tokens
        new_balance = await TokenService.add_tokens(
            user_id=user.id,
            tokens_to_add=tokens,
            source=f"purchase:{package}"
        )

        return {
            "success": True,
            "tokens_purchased": package_info["tokens"],
            "bonus_tokens": bonus_tokens if active_promo else 0,
            "total_tokens_added": tokens,
            "new_balance": new_balance
        }
```

---

## Tier Comparison Table

| Feature | FREE | PRO ($20/mo) | ENTERPRISE ($200/mo) |
|---------|------|--------------|----------------------|
| **Monthly Tokens** | 100 | 1,000 | 10,000 |
| **Token Rollover** | ❌ | ✅ (500 max) | ✅ (5,000 max) |
| **Cost per Token** | - | $0.02 | $0.02 |
| **Signup Bonus** | - | 200 tokens | 2,000 tokens |
| **Images (~25 tokens)** | ~4/mo | ~40/mo | ~400/mo |
| **Videos (~200 tokens)** | ❌ | ~5/mo | ~50/mo |
| **Audio (~50 tokens)** | ~2/mo | ~20/mo | ~200/mo |
| **Watermark** | ✅ Yes | 🔧 Optional | ❌ No |
| **Max Resolution** | 1024x1024 | 2048x2048 | 4096x4096 |
| **API Access** | ❌ | ❌ | ✅ |
| **Priority Queue** | Low | Normal | High |
| **Concurrent Jobs** | 1 | 5 | 20 |
| **Support** | Community | Email | Priority + Phone |

---

This token-based system provides:
✅ **Fair pricing** based on actual resource usage
✅ **Flexible consumption** - mix and match different generations
✅ **Clear costs** - users know exactly what they'll pay
✅ **Automatic refunds** - failed jobs don't waste tokens
✅ **Scalable** - from hobbyists to enterprises
