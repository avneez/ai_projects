# Authentication & Token System - Overview

This folder contains complete documentation for TXT2CREATE's dual-token authentication and usage tracking system.

---

## 📋 Quick Summary

TXT2CREATE uses **two types of tokens**:

1. **JWT Tokens** - For user authentication (who you are)
2. **Usage Tokens** - For resource tracking and billing (what you can do)

---

## 📁 Documentation Files

### 1. [Authentication Flow](./authentication-flow.md)
**Complete user authentication system with JWT**

**What's covered:**
- User registration and email verification
- Login flow with JWT generation
- Access tokens (15 min) + Refresh tokens (7 days)
- Token verification and protected endpoints
- Session management with Redis
- Logout and token revocation

**Key Code Sections:**
- User registration with password hashing
- JWT creation and validation
- Permission-based access control (RBAC)
- Token refresh mechanism
- Session storage and management

**Example Flow:**
```
User registers → Email verification → Login
→ Receive JWT tokens → Make API calls with JWT
→ Token expires → Refresh with refresh token
→ Continue using app
```

---

### 2. [Token Pricing & Tiers](./token-pricing-tiers.md)
**Usage token system for fair resource allocation**

**What's covered:**
- Token economics (1 token = $0.01)
- Subscription tiers (FREE, PRO, ENTERPRISE)
- Detailed token cost formulas for all pipelines
- Token purchase packages
- Automatic refund policies
- Token monitoring dashboard
- Cost estimation before generation

**Pricing Tiers:**

| Tier | Monthly Cost | Tokens | Key Features |
|------|--------------|--------|--------------|
| **FREE** | $0 | 100 | Images + Audio, watermarked |
| **PRO** | $20 | 1,000 | All features, no watermark |
| **ENTERPRISE** | $200 | 10,000 | API access, priority support |

**Token Costs (Examples):**
- Image (1024×1024, SD XL, 30 steps): **30 tokens** ($0.30)
- Video (10s, 720p): **350 tokens** ($3.50)
- Audio (30s music): **50 tokens** ($0.50)
- Avatar (3D rigged): **450 tokens** ($4.50)

---

## 🔑 How It Works

### Part 1: User Authentication (JWT)

```
┌─────────────────────────────────────────────────────┐
│ Step 1: User Login                                  │
├─────────────────────────────────────────────────────┤
│ POST /api/v1/auth/login                             │
│ {                                                   │
│   "email": "user@example.com",                      │
│   "password": "SecurePass123!"                      │
│ }                                                   │
│                                                     │
│ ↓                                                   │
│                                                     │
│ Auth Service:                                       │
│ 1. Verify password (bcrypt)                         │
│ 2. Generate access token (JWT, 15 min)              │
│ 3. Generate refresh token (JWT, 7 days)             │
│ 4. Store session in Redis                           │
│                                                     │
│ ↓                                                   │
│                                                     │
│ Response:                                           │
│ {                                                   │
│   "access_token": "eyJhbGci...",                    │
│   "refresh_token": "eyJhbGci...",                   │
│   "expires_in": 900                                 │
│ }                                                   │
└─────────────────────────────────────────────────────┘
```

**JWT Access Token Contains:**
```json
{
  "sub": "user-id-550e8400-...",
  "email": "user@example.com",
  "tier": "PRO",
  "permissions": ["generate_image", "generate_video", ...],
  "exp": 1703174400,
  "type": "access"
}
```

### Part 2: Usage Token Deduction

```
┌─────────────────────────────────────────────────────┐
│ Step 2: User Generates Video                       │
├─────────────────────────────────────────────────────┤
│ POST /api/v1/generate/video                         │
│ Headers:                                            │
│   Authorization: Bearer eyJhbGci...                 │
│ Body:                                               │
│ {                                                   │
│   "prompt": "Sunset over mountains",                │
│   "duration": 10,                                   │
│   "resolution": "720p"                              │
│ }                                                   │
│                                                     │
│ ↓                                                   │
│                                                     │
│ FastAPI Service:                                    │
│ 1. Verify JWT → Extract user_id                     │
│ 2. Calculate cost:                                  │
│    tokens = 100 + (10 × 20) × 2.0 × 1.5 = 700      │
│ 3. Check permission: "generate_video" ✓             │
│                                                     │
│ ↓                                                   │
│                                                     │
│ Token Service (PostgreSQL Transaction):            │
│ BEGIN;                                              │
│   SELECT balance FROM user_tokens                   │
│   WHERE user_id = '...' FOR UPDATE;                 │
│                                                     │
│   -- balance = 1,000                                │
│   -- required = 700                                 │
│   -- enough? YES ✓                                  │
│                                                     │
│   UPDATE user_tokens                                │
│   SET balance = balance - 700,                      │
│       quota_used = quota_used + 700                 │
│   WHERE user_id = '...';                            │
│                                                     │
│   INSERT INTO token_usage_log ...;                  │
│ COMMIT;                                             │
│                                                     │
│ ↓                                                   │
│                                                     │
│ Create job → Queue Celery task                      │
│                                                     │
│ ↓                                                   │
│                                                     │
│ Response:                                           │
│ {                                                   │
│   "job_id": "abc-123",                              │
│   "status": "queued",                               │
│   "tokens_used": 700,                               │
│   "new_balance": 300,                               │
│   "estimated_time": "3-5 minutes"                   │
│ }                                                   │
└─────────────────────────────────────────────────────┘
```

### Part 3: Automatic Refund on Failure

```
┌─────────────────────────────────────────────────────┐
│ Step 3: Video Processing                           │
├─────────────────────────────────────────────────────┤
│ Celery Worker:                                      │
│                                                     │
│ try:                                                │
│   generate_video(prompt, duration, resolution)      │
│   upload_to_s3(video)                               │
│   mark_job_complete()                               │
│   # SUCCESS - tokens already deducted, keep them    │
│                                                     │
│ except Exception as e:                              │
│   # FAILURE - refund tokens automatically           │
│   refund_tokens(user_id, job_id, 700)               │
│   mark_job_failed(error=str(e))                     │
│   # User gets 700 tokens back                       │
│                                                     │
│ Refund Transaction:                                 │
│ BEGIN;                                              │
│   UPDATE user_tokens                                │
│   SET balance = balance + 700,                      │
│       quota_used = quota_used - 700                 │
│   WHERE user_id = '...';                            │
│                                                     │
│   INSERT INTO token_refund_log (                    │
│     tokens_refunded: 700,                           │
│     reason: "GPU error: CUDA out of memory"         │
│   );                                                │
│ COMMIT;                                             │
└─────────────────────────────────────────────────────┘
```

---

## 🔐 Security Features

### JWT Security
- **HS256 signing** with secret key
- **Short-lived access tokens** (15 min) - minimize damage if stolen
- **Long-lived refresh tokens** (7 days) - for user convenience
- **Unique token IDs (jti)** - for revocation
- **Session tracking** in Redis - can invalidate all sessions

### Token Transaction Security
- **Atomic operations** - PostgreSQL transactions
- **Row-level locking** - prevent race conditions
- **Optimistic concurrency** - version-based updates
- **Audit trail** - all token usage logged
- **Automatic refunds** - failed jobs don't waste tokens

---

## 💰 Token Economics

### Token Pricing Strategy

```
Base principle: 1 token = $0.01 USD

Why token-based?
✓ Fair - pay for what you use
✓ Flexible - mix and match different generations
✓ Predictable - users know costs upfront
✓ Scalable - from hobbyists to enterprises

Cost Structure:
- Text-to-Image: 10-120 tokens ($0.10 - $1.20)
- Text-to-Video: 80-1500 tokens ($0.80 - $15.00)
- Text-to-Audio: 15-250 tokens ($0.15 - $2.50)
- Video Captioning: 20-80 tokens ($0.20 - $0.80)
- Avatar Generation: 150-2000 tokens ($1.50 - $20.00)
```

### Subscription vs Pay-As-You-Go

**Subscriptions (Monthly Quota):**
- FREE: 100 tokens/month (resets monthly)
- PRO: 1,000 tokens/month + rollover (up to 500)
- ENTERPRISE: 10,000 tokens/month + rollover (up to 5,000)

**One-Time Purchases (No Expiry):**
- 100 tokens = $1.00
- 500 tokens = $4.50 (10% off)
- 1,000 tokens = $8.00 (20% off)
- 5,000 tokens = $35.00 (30% off)
- 10,000 tokens = $60.00 (40% off)

**Usage Order:**
1. Use monthly quota first
2. Then use purchased tokens
3. Then prompt to purchase more

---

## 📊 Database Schema

### Users Table
```sql
CREATE TABLE users (
  id UUID PRIMARY KEY,
  email VARCHAR UNIQUE NOT NULL,
  password_hash VARCHAR NOT NULL,
  name VARCHAR,
  tier VARCHAR DEFAULT 'FREE',  -- FREE, PRO, ENTERPRISE
  email_verified BOOLEAN DEFAULT false,
  created_at TIMESTAMP,
  last_login TIMESTAMP
);
```

### User Tokens Table
```sql
CREATE TABLE user_tokens (
  user_id UUID PRIMARY KEY REFERENCES users(id),
  balance INTEGER DEFAULT 0,           -- Current token balance
  monthly_quota INTEGER DEFAULT 100,   -- Monthly allocation
  quota_used INTEGER DEFAULT 0,        -- Used this month
  reset_date TIMESTAMP,                -- When quota resets
  created_at TIMESTAMP,
  last_used TIMESTAMP
);
```

### Token Usage Log
```sql
CREATE TABLE token_usage_log (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  job_id UUID REFERENCES jobs(id),
  pipeline VARCHAR,                    -- text_to_image, text_to_video, etc.
  tokens_used INTEGER,
  timestamp TIMESTAMP,
  INDEX idx_user_timestamp (user_id, timestamp)
);
```

### Token Refund Log
```sql
CREATE TABLE token_refund_log (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  job_id UUID REFERENCES jobs(id),
  tokens_refunded INTEGER,
  reason VARCHAR,
  timestamp TIMESTAMP
);
```

---

## 🔄 API Endpoints

### Authentication Endpoints

```python
POST   /api/v1/auth/register          # Create new account
POST   /api/v1/auth/login              # Get JWT tokens
POST   /api/v1/auth/refresh            # Refresh access token
POST   /api/v1/auth/logout             # Revoke session
GET    /api/v1/auth/verify-email       # Verify email address
POST   /api/v1/auth/forgot-password    # Password reset
```

### Token Management Endpoints

```python
GET    /api/v1/tokens/balance          # Get current balance
GET    /api/v1/tokens/usage            # Usage history
POST   /api/v1/tokens/purchase         # Buy tokens
POST   /api/v1/tokens/estimate-cost    # Estimate before generation
GET    /api/v1/tokens/pricing          # Get pricing info
```

### Protected Generation Endpoints

```python
# All require: Authorization: Bearer <access_token>

POST   /api/v1/generate/image          # Generate image (10-120 tokens)
POST   /api/v1/generate/video          # Generate video (80-1500 tokens)
POST   /api/v1/generate/audio          # Generate audio (15-250 tokens)
POST   /api/v1/caption/video           # Caption video (20-80 tokens)
POST   /api/v1/generate/avatar         # Generate avatar (150-2000 tokens)
```

---

## 🚨 Error Handling

### Authentication Errors

```json
// 401 Unauthorized - Invalid or expired token
{
  "error": "Invalid token",
  "code": "TOKEN_INVALID",
  "message": "Please login again"
}

// 403 Forbidden - No permission
{
  "error": "Permission denied",
  "code": "PERMISSION_DENIED",
  "required_permission": "generate_video",
  "current_tier": "FREE",
  "message": "Upgrade to PRO to access video generation"
}
```

### Token Errors

```json
// 402 Payment Required - Insufficient tokens
{
  "error": "Insufficient tokens",
  "code": "INSUFFICIENT_TOKENS",
  "tokens_required": 700,
  "tokens_available": 300,
  "tokens_needed": 400,
  "purchase_url": "/api/v1/tokens/purchase",
  "suggestions": {
    "basic_package": {
      "tokens": 500,
      "price": 4.50,
      "enough": true
    },
    "pro_upgrade": {
      "tokens": 1000,
      "price": 20.00,
      "monthly": true
    }
  }
}
```

---

## 🎯 Best Practices

### For Frontend Developers

1. **Store tokens securely**
   ```javascript
   // Access token in localStorage (short-lived, acceptable)
   localStorage.setItem('access_token', token);

   // Refresh token in httpOnly cookie (more secure)
   // Set by backend, not accessible to JavaScript
   ```

2. **Automatic token refresh**
   ```javascript
   // Intercept 401 errors and refresh token
   axios.interceptors.response.use(
     response => response,
     async error => {
       if (error.response.status === 401) {
         const newToken = await refreshAccessToken();
         // Retry original request with new token
         return axios(originalRequest);
       }
     }
   );
   ```

3. **Show token balance**
   ```javascript
   // Always display current balance before generation
   const balance = await getTokenBalance();
   const cost = await estimateCost(generationParams);

   if (balance.tokens < cost.tokens_required) {
     showPurchaseDialog(cost.tokens_needed);
   }
   ```

### For Backend Developers

1. **Use transactions for token deduction**
   ```python
   async with db.begin():  # Always use transactions
       tokens = await db.query(...).with_for_update()
       if tokens.balance >= required:
           tokens.balance -= required
       else:
           raise InsufficientTokensError()
   ```

2. **Always refund on failure**
   ```python
   try:
       result = generate_image(prompt)
   except Exception as e:
       await refund_tokens(user_id, job_id, tokens_used)
       raise
   ```

3. **Log everything**
   ```python
   # Log all token operations for audit trail
   await db.add(TokenUsageLog(...))
   await db.add(TokenRefundLog(...))
   ```

---

## 📈 Monitoring & Analytics

### Key Metrics to Track

1. **Token Economics**
   - Tokens purchased per day
   - Tokens used per day
   - Tokens refunded per day
   - Average tokens per user
   - Conversion rate (free → paid)

2. **Authentication**
   - Login success rate
   - Token refresh rate
   - Session duration
   - Failed login attempts

3. **Usage Patterns**
   - Most popular pipelines
   - Peak usage times
   - Average cost per generation
   - User tier distribution

---

## 🔗 Related Documentation

- [System Architecture](../diagrams/system-architecture.md) - How auth fits in overall system
- [Data Flow Overview](../pipelines/data-flow-overview.md) - Request flow through pipelines
- [Tech Stack](../tech-stack/complete-stack.md) - Technologies used
- [Q&A - Security](../qa/comprehensive-qa.md#7-security--privacy) - Security deep dive

---

## 🚀 Quick Start for Developers

### 1. Test Authentication

```bash
# Register
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "SecurePass123!",
    "name": "Test User"
  }'

# Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "SecurePass123!"
  }'

# Response: {"access_token": "eyJ...", "refresh_token": "eyJ..."}
```

### 2. Make Authenticated Request

```bash
# Generate image (requires tokens)
curl -X POST http://localhost:8000/api/v1/generate/image \
  -H "Authorization: Bearer eyJhbGci..." \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat in a garden",
    "resolution": "1024x1024",
    "steps": 30
  }'

# Response: {"job_id": "abc-123", "tokens_used": 30, "new_balance": 70}
```

### 3. Check Token Balance

```bash
curl -X GET http://localhost:8000/api/v1/tokens/balance \
  -H "Authorization: Bearer eyJhbGci..."

# Response:
# {
#   "balance": 70,
#   "monthly_quota": 100,
#   "quota_used": 30,
#   "reset_date": "2026-01-21T00:00:00Z"
# }
```

---

**Last Updated**: 2025-12-21
**Version**: 1.0
**Status**: Complete Architecture Design
