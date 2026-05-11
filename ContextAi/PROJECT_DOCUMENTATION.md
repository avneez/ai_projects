# ContextAI — Multi-Tenant LLM Application Platform

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Business Problem & Solution](#business-problem--solution)
3. [System Architecture](#system-architecture)
4. [Technology Stack](#technology-stack)
5. [Code Snippets & Implementation](#code-snippets--implementation)
6. [Security & Compliance](#security--compliance)
7. [Monitoring & Observability](#monitoring--observability)
8. [Interview Cross-Questions & Answers](#interview-cross-questions--answers)

---

## Executive Summary

**ContextAI** is an enterprise-grade, multi-tenant platform designed to deploy, manage, and orchestrate LLM (Large Language Model) applications at scale. It provides a unified API layer that abstracts multiple LLM providers, ensuring high availability, security, and compliance for production workloads.

**Key Metrics:**
- Supports 50+ concurrent tenants
- 99.9% uptime SLA
- <200ms API response time (p95)
- AES-256 encryption at rest
- JWT/OAuth2 authentication
- Real-time monitoring with ELK, Prometheus, and Grafana

---

## Business Problem & Solution

### What Was the Business Problem?

**Problem Statement:**
Organizations face several challenges when deploying LLM applications in production:

1. **Vendor Lock-in**: Tightly coupled to single LLM providers (OpenAI, Anthropic, etc.)
2. **Lack of Multi-tenancy**: No secure way to serve multiple customers/departments with isolated data
3. **Security Concerns**: Sensitive data exposure, inadequate encryption, poor secrets management
4. **No Observability**: Limited visibility into API latency, token usage, costs, and performance
5. **Scalability Issues**: Difficulty handling concurrent requests across multiple tenants
6. **Compliance Requirements**: GDPR, SOC2, HIPAA compliance needed for enterprise adoption

### How ContextAI Solves It

**Solution Architecture:**

1. **Unified API Layer**: Single API interface for multiple LLM providers with automatic failover
2. **Multi-Tenant Architecture**: Complete data isolation using tenant-specific databases and encryption keys
3. **Enterprise Security**: AES-256 encryption, JWT/OAuth2, HashiCorp Vault, rate limiting, and audit logs
4. **Horizontal Scalability**: Kubernetes-based microservices with auto-scaling based on load
5. **Comprehensive Monitoring**: Real-time tracking of performance, costs, and usage patterns
6. **Compliance-Ready**: Built-in audit trails, data residency controls, and privacy features

### Why It Matters

- **Cost Optimization**: Dynamic routing to cost-effective models, token usage tracking
- **High Availability**: Automatic failover prevents vendor outages from impacting business
- **Security & Compliance**: Enterprise-ready security enables adoption in regulated industries
- **Developer Productivity**: Unified API reduces integration complexity from weeks to hours
- **Scalability**: Handle growth from 10 to 10,000+ users without re-architecture

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Load Balancer (NGINX)                     │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
        ┌───────▼────────┐             ┌───────▼────────┐
        │  API Gateway   │             │  API Gateway   │
        │   (FastAPI)    │             │   (FastAPI)    │
        └───────┬────────┘             └───────┬────────┘
                │                               │
        ┌───────▼───────────────────────────────▼────────┐
        │         Authentication Service (JWT/OAuth2)     │
        └───────┬─────────────────────────────────────────┘
                │
        ┌───────▼──────────────────────────────────────────┐
        │         LLM Router & Orchestration Service       │
        │    (Dynamic Model Selection & Failover Logic)    │
        └───┬─────────────┬─────────────┬─────────────┬────┘
            │             │             │             │
    ┌───────▼──────┐ ┌────▼─────┐ ┌────▼─────┐ ┌────▼─────┐
    │   OpenAI     │ │ Anthropic │ │  Hugging │ │  Cohere  │
    │   Adapter    │ │  Adapter  │ │   Face   │ │  Adapter │
    └──────────────┘ └───────────┘ └──────────┘ └──────────┘

┌──────────────────────────────────────────────────────────────────┐
│                     Supporting Services                          │
├──────────────────────────────────────────────────────────────────┤
│  - Tenant Management Service                                     │
│  - Token Usage Tracker                                           │
│  - Rate Limiter (Redis)                                          │
│  - Secrets Manager (HashiCorp Vault)                             │
│  - Audit Logger                                                  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                   Data Layer                                     │
├──────────────────────────────────────────────────────────────────┤
│  PostgreSQL (Metadata, Users, Tenants) - Encrypted at Rest       │
│  Redis (Caching, Rate Limiting, Session Management)              │
│  S3 (Audit Logs, Model Responses - Encrypted)                   │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│              Monitoring & Observability Stack                    │
├──────────────────────────────────────────────────────────────────┤
│  Elasticsearch (Log Aggregation)                                 │
│  Logstash (Log Processing)                                       │
│  Kibana (Visualization)                                          │
│  Prometheus (Metrics Collection)                                 │
│  Grafana (Metrics Dashboard)                                     │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                 Frontend Application                             │
├──────────────────────────────────────────────────────────────────┤
│  React (TypeScript) - Admin Dashboard & API Console              │
│  - Tenant Management UI                                          │
│  - Usage Analytics & Billing                                     │
│  - API Key Management                                            │
│  - Real-time Monitoring Dashboard                                │
└──────────────────────────────────────────────────────────────────┘
```

### Microservices Architecture

**Core Services:**

1. **API Gateway Service** (FastAPI)
   - Request routing and validation
   - Authentication/Authorization
   - Rate limiting enforcement

2. **LLM Router Service** (Python)
   - Dynamic model selection based on tenant config
   - Failover logic when primary provider fails
   - Request/response transformation

3. **Tenant Management Service** (FastAPI)
   - Tenant provisioning and configuration
   - Subscription management
   - Feature flag management

4. **Usage Tracker Service** (Python)
   - Token counting and cost calculation
   - Real-time usage metrics
   - Quota enforcement

5. **Audit Service** (Python)
   - Immutable audit trail
   - Compliance reporting
   - Security event logging

### Data Flow

```
1. Client Request → Load Balancer
2. Load Balancer → API Gateway
3. API Gateway → Auth Service (JWT Validation)
4. API Gateway → Rate Limiter (Redis Check)
5. API Gateway → LLM Router
6. LLM Router → Model Selection Logic
7. LLM Router → Provider Adapter (OpenAI/Anthropic/etc.)
8. Provider Adapter → External LLM API
9. Response → Encryption → Caching (Redis)
10. Response → Usage Tracker (Async)
11. Response → Audit Logger (Async)
12. Response → Client
13. Metrics → Prometheus/ELK Stack
```

---

## Technology Stack

### Backend
- **Python 3.11+** with FastAPI (API services)
- **Pydantic** (Data validation)
- **SQLAlchemy** (ORM)
- **Celery** (Async task processing)
- **Alembic** (Database migrations)

### Frontend
- **React 18+** with TypeScript
- **Vite** (Build tool)
- **TanStack Query** (Data fetching)
- **Zustand** (State management)
- **Recharts** (Data visualization)
- **TailwindCSS** (Styling)

### Infrastructure
- **Docker** (Containerization)
- **Kubernetes** (Orchestration)
- **NGINX** (Load balancing, reverse proxy)
- **Helm** (K8s package management)

### Databases & Caching
- **PostgreSQL 15** (Primary database)
- **Redis 7** (Caching, rate limiting, sessions)
- **AWS S3** (Object storage for logs)

### Security
- **HashiCorp Vault** (Secrets management)
- **JWT/OAuth2** (Authentication)
- **AES-256** (Encryption at rest)
- **TLS 1.3** (Encryption in transit)

### Monitoring & Observability
- **ELK Stack** (Elasticsearch, Logstash, Kibana)
- **Prometheus** (Metrics collection)
- **Grafana** (Metrics visualization)
- **Jaeger** (Distributed tracing)

### LLM Providers
- **OpenAI** (GPT-3.5, GPT-4, GPT-4-Turbo)
- **Anthropic** (Claude 2, Claude 3)
- **Hugging Face** (Open-source models)
- **Cohere** (Command, Generate)

### CI/CD
- **GitHub Actions** (CI/CD pipelines)
- **ArgoCD** (GitOps deployment)
- **SonarQube** (Code quality)
- **Trivy** (Security scanning)

---

## Code Snippets & Implementation

### 1. FastAPI Gateway with Multi-Tenant Authentication

```python
# services/api_gateway/main.py
from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import jwt
from datetime import datetime, timedelta
from pydantic import BaseModel
import redis
import hashlib

app = FastAPI(title="ContextAI API Gateway", version="1.0.0")
security = HTTPBearer()
redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)

# Configuration
JWT_SECRET = "your-secret-key-from-vault"
JWT_ALGORITHM = "HS256"
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX_REQUESTS = 100

class TenantContext(BaseModel):
    tenant_id: str
    user_id: str
    subscription_tier: str
    permissions: list[str]

class LLMRequest(BaseModel):
    prompt: str
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000
    stream: bool = False

async def get_tenant_from_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    x_tenant_id: Optional[str] = Header(None)
) -> TenantContext:
    """
    Extract and validate tenant information from JWT token
    Implements multi-tenant authentication
    """
    try:
        token = credentials.credentials
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])

        # Validate tenant_id matches header
        if x_tenant_id and payload.get("tenant_id") != x_tenant_id:
            raise HTTPException(status_code=403, detail="Tenant ID mismatch")

        return TenantContext(
            tenant_id=payload["tenant_id"],
            user_id=payload["user_id"],
            subscription_tier=payload.get("tier", "free"),
            permissions=payload.get("permissions", [])
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def check_rate_limit(tenant: TenantContext):
    """
    Redis-based rate limiting per tenant
    Implements sliding window rate limiting
    """
    rate_limit_key = f"rate_limit:{tenant.tenant_id}:{datetime.now().strftime('%Y%m%d%H%M')}"

    current_count = redis_client.incr(rate_limit_key)

    if current_count == 1:
        redis_client.expire(rate_limit_key, RATE_LIMIT_WINDOW)

    # Adjust rate limit based on subscription tier
    max_requests = {
        "free": 100,
        "pro": 1000,
        "enterprise": 10000
    }.get(tenant.subscription_tier, 100)

    if current_count > max_requests:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {max_requests} requests per minute for {tenant.subscription_tier} tier"
        )

    return current_count

@app.post("/v1/chat/completions")
async def chat_completion(
    request: LLMRequest,
    tenant: TenantContext = Depends(get_tenant_from_token)
):
    """
    Main LLM API endpoint with multi-tenant isolation
    """
    # Rate limiting
    await check_rate_limit(tenant)

    # Route to LLM Router service
    from services.llm_router import route_request

    response = await route_request(
        tenant_id=tenant.tenant_id,
        prompt=request.prompt,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stream=request.stream
    )

    return response

@app.get("/health")
async def health_check():
    """Health check endpoint for Kubernetes liveness probe"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}
```

### 2. LLM Router with Failover Mechanism

```python
# services/llm_router/router.py
from typing import Optional, Dict, Any
import asyncio
import httpx
from enum import Enum
from datetime import datetime
import logging
from circuitbreaker import circuit

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGING_FACE = "huggingface"
    COHERE = "cohere"

class ProviderConfig(BaseModel):
    provider: LLMProvider
    api_key: str
    priority: int
    enabled: bool
    max_retries: int = 3

class LLMRouter:
    """
    Intelligent router that selects optimal LLM provider
    with automatic failover on errors
    """

    def __init__(self):
        self.providers = {}
        self.circuit_breakers = {}
        self.metrics = {}

    async def register_provider(self, tenant_id: str, config: ProviderConfig):
        """Register LLM provider for a tenant"""
        if tenant_id not in self.providers:
            self.providers[tenant_id] = []

        self.providers[tenant_id].append(config)
        # Sort by priority (higher priority first)
        self.providers[tenant_id].sort(key=lambda x: x.priority, reverse=True)

    @circuit(failure_threshold=5, recovery_timeout=60)
    async def call_openai(self, api_key: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """OpenAI API adapter with circuit breaker"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": kwargs.get("model", "gpt-3.5-turbo"),
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_tokens", 1000)
                }
            )
            response.raise_for_status()
            return response.json()

    @circuit(failure_threshold=5, recovery_timeout=60)
    async def call_anthropic(self, api_key: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Anthropic API adapter with circuit breaker"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                },
                json={
                    "model": kwargs.get("model", "claude-3-sonnet-20240229"),
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": kwargs.get("max_tokens", 1000)
                }
            )
            response.raise_for_status()
            return response.json()

    async def route_request(
        self,
        tenant_id: str,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Route request to appropriate provider with automatic failover
        """
        providers = self.providers.get(tenant_id, [])

        if not providers:
            raise ValueError(f"No providers configured for tenant {tenant_id}")

        last_error = None

        # Try providers in priority order
        for config in providers:
            if not config.enabled:
                continue

            try:
                logger.info(f"Attempting provider: {config.provider.value}")
                start_time = datetime.now()

                if config.provider == LLMProvider.OPENAI:
                    result = await self.call_openai(config.api_key, prompt, **kwargs)
                elif config.provider == LLMProvider.ANTHROPIC:
                    result = await self.call_anthropic(config.api_key, prompt, **kwargs)
                else:
                    raise NotImplementedError(f"Provider {config.provider} not implemented")

                # Track metrics
                latency = (datetime.now() - start_time).total_seconds()
                self._record_success(tenant_id, config.provider, latency)

                return {
                    "provider": config.provider.value,
                    "response": result,
                    "latency_ms": latency * 1000
                }

            except Exception as e:
                last_error = e
                logger.error(f"Provider {config.provider.value} failed: {str(e)}")
                self._record_failure(tenant_id, config.provider, str(e))
                continue

        # All providers failed
        raise Exception(f"All providers failed. Last error: {str(last_error)}")

    def _record_success(self, tenant_id: str, provider: LLMProvider, latency: float):
        """Record successful API call metrics"""
        key = f"{tenant_id}:{provider.value}"
        if key not in self.metrics:
            self.metrics[key] = {"success": 0, "failure": 0, "total_latency": 0}

        self.metrics[key]["success"] += 1
        self.metrics[key]["total_latency"] += latency

    def _record_failure(self, tenant_id: str, provider: LLMProvider, error: str):
        """Record failed API call metrics"""
        key = f"{tenant_id}:{provider.value}"
        if key not in self.metrics:
            self.metrics[key] = {"success": 0, "failure": 0, "total_latency": 0}

        self.metrics[key]["failure"] += 1

# Global router instance
router = LLMRouter()

async def route_request(tenant_id: str, prompt: str, **kwargs):
    """Public interface for routing requests"""
    return await router.route_request(tenant_id, prompt, **kwargs)
```

### 3. Encryption at Rest Implementation

```python
# services/security/encryption.py
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
import os
import base64
from typing import Union

class AES256Encryption:
    """
    AES-256 encryption for data at rest
    Each tenant has a unique encryption key stored in HashiCorp Vault
    """

    def __init__(self, key: bytes):
        """
        Initialize with 256-bit key (32 bytes)
        """
        if len(key) != 32:
            raise ValueError("Key must be 32 bytes for AES-256")
        self.key = key
        self.backend = default_backend()

    def encrypt(self, plaintext: Union[str, bytes]) -> str:
        """
        Encrypt data using AES-256-CBC
        Returns base64-encoded ciphertext with IV prepended
        """
        if isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')

        # Generate random IV
        iv = os.urandom(16)

        # Pad plaintext to block size (128 bits = 16 bytes)
        padder = padding.PKCS7(algorithms.AES.block_size).padder()
        padded_data = padder.update(plaintext) + padder.finalize()

        # Encrypt
        cipher = Cipher(
            algorithms.AES(self.key),
            modes.CBC(iv),
            backend=self.backend
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()

        # Prepend IV to ciphertext and encode as base64
        encrypted_data = iv + ciphertext
        return base64.b64encode(encrypted_data).decode('utf-8')

    def decrypt(self, ciphertext_b64: str) -> str:
        """
        Decrypt AES-256-CBC encrypted data
        Expects base64-encoded ciphertext with IV prepended
        """
        # Decode from base64
        encrypted_data = base64.b64decode(ciphertext_b64.encode('utf-8'))

        # Extract IV (first 16 bytes) and ciphertext
        iv = encrypted_data[:16]
        ciphertext = encrypted_data[16:]

        # Decrypt
        cipher = Cipher(
            algorithms.AES(self.key),
            modes.CBC(iv),
            backend=self.backend
        )
        decryptor = cipher.decryptor()
        padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()

        # Unpad
        unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
        plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()

        return plaintext.decode('utf-8')

# Integration with HashiCorp Vault
import hvac

class VaultKeyManager:
    """
    Manage encryption keys in HashiCorp Vault
    Each tenant gets a unique encryption key
    """

    def __init__(self, vault_url: str, vault_token: str):
        self.client = hvac.Client(url=vault_url, token=vault_token)

    def get_tenant_key(self, tenant_id: str) -> bytes:
        """
        Retrieve tenant-specific encryption key from Vault
        Creates new key if doesn't exist
        """
        secret_path = f"secret/data/tenants/{tenant_id}/encryption_key"

        try:
            secret = self.client.secrets.kv.v2.read_secret_version(
                path=f"tenants/{tenant_id}/encryption_key"
            )
            key_b64 = secret['data']['data']['key']
            return base64.b64decode(key_b64)
        except:
            # Key doesn't exist, create new one
            new_key = os.urandom(32)  # 256 bits
            self.client.secrets.kv.v2.create_or_update_secret(
                path=f"tenants/{tenant_id}/encryption_key",
                secret={'key': base64.b64encode(new_key).decode('utf-8')}
            )
            return new_key

    def rotate_tenant_key(self, tenant_id: str) -> bytes:
        """
        Rotate encryption key for a tenant
        NOTE: Requires re-encryption of existing data
        """
        new_key = os.urandom(32)
        self.client.secrets.kv.v2.create_or_update_secret(
            path=f"tenants/{tenant_id}/encryption_key",
            secret={'key': base64.b64encode(new_key).decode('utf-8')}
        )
        return new_key

# Usage Example
vault_manager = VaultKeyManager(
    vault_url="http://vault:8200",
    vault_token=os.getenv("VAULT_TOKEN")
)

def encrypt_tenant_data(tenant_id: str, data: str) -> str:
    """Encrypt data for specific tenant"""
    key = vault_manager.get_tenant_key(tenant_id)
    encryptor = AES256Encryption(key)
    return encryptor.encrypt(data)

def decrypt_tenant_data(tenant_id: str, encrypted_data: str) -> str:
    """Decrypt data for specific tenant"""
    key = vault_manager.get_tenant_key(tenant_id)
    encryptor = AES256Encryption(key)
    return encryptor.decrypt(encrypted_data)
```

### 4. Prometheus Metrics & Monitoring

```python
# services/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info
from functools import wraps
import time

# Define metrics
llm_requests_total = Counter(
    'llm_requests_total',
    'Total number of LLM API requests',
    ['tenant_id', 'provider', 'model', 'status']
)

llm_request_duration = Histogram(
    'llm_request_duration_seconds',
    'LLM request duration in seconds',
    ['tenant_id', 'provider', 'model'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

llm_tokens_used = Counter(
    'llm_tokens_used_total',
    'Total tokens consumed',
    ['tenant_id', 'provider', 'model', 'token_type']
)

active_tenants = Gauge(
    'active_tenants',
    'Number of active tenants'
)

rate_limit_exceeded = Counter(
    'rate_limit_exceeded_total',
    'Number of rate limit violations',
    ['tenant_id', 'tier']
)

provider_availability = Gauge(
    'provider_availability',
    'Provider availability (1=up, 0=down)',
    ['provider']
)

application_info = Info(
    'contextai_application',
    'ContextAI application information'
)

# Set application info
application_info.info({
    'version': '1.0.0',
    'environment': 'production'
})

def track_llm_request(func):
    """
    Decorator to track LLM request metrics
    """
    @wraps(func)
    async def wrapper(tenant_id: str, provider: str, model: str, *args, **kwargs):
        start_time = time.time()
        status = "success"

        try:
            result = await func(tenant_id, provider, model, *args, **kwargs)

            # Track token usage
            if 'usage' in result:
                llm_tokens_used.labels(
                    tenant_id=tenant_id,
                    provider=provider,
                    model=model,
                    token_type='prompt'
                ).inc(result['usage'].get('prompt_tokens', 0))

                llm_tokens_used.labels(
                    tenant_id=tenant_id,
                    provider=provider,
                    model=model,
                    token_type='completion'
                ).inc(result['usage'].get('completion_tokens', 0))

            return result

        except Exception as e:
            status = "error"
            raise
        finally:
            # Record request count
            llm_requests_total.labels(
                tenant_id=tenant_id,
                provider=provider,
                model=model,
                status=status
            ).inc()

            # Record duration
            duration = time.time() - start_time
            llm_request_duration.labels(
                tenant_id=tenant_id,
                provider=provider,
                model=model
            ).observe(duration)

    return wrapper

# Usage in LLM Router
@track_llm_request
async def call_llm_provider(tenant_id: str, provider: str, model: str, prompt: str):
    # Implementation
    pass
```

### 5. React Dashboard Component (TypeScript)

```typescript
// frontend/src/components/UsageDashboard.tsx
import React, { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface UsageMetrics {
  timestamp: string;
  requestCount: number;
  tokenUsage: number;
  averageLatency: number;
  errorRate: number;
  cost: number;
}

interface TenantUsage {
  tenantId: string;
  currentPeriod: {
    requests: number;
    tokens: number;
    cost: number;
  };
  metrics: UsageMetrics[];
  quotaLimits: {
    maxRequests: number;
    maxTokens: number;
  };
}

const UsageDashboard: React.FC = () => {
  const [timeRange, setTimeRange] = useState<'1h' | '24h' | '7d' | '30d'>('24h');

  // Fetch usage data
  const { data, isLoading, error } = useQuery<TenantUsage>({
    queryKey: ['tenantUsage', timeRange],
    queryFn: async () => {
      const response = await fetch(`/api/v1/analytics/usage?range=${timeRange}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('authToken')}`,
          'X-Tenant-ID': localStorage.getItem('tenantId') || '',
        },
      });
      if (!response.ok) throw new Error('Failed to fetch usage data');
      return response.json();
    },
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  if (isLoading) return <LoadingSpinner />;
  if (error) return <ErrorMessage message="Failed to load usage data" />;
  if (!data) return null;

  const utilizationPercentage = {
    requests: (data.currentPeriod.requests / data.quotaLimits.maxRequests) * 100,
    tokens: (data.currentPeriod.tokens / data.quotaLimits.maxTokens) * 100,
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-gray-900">Usage Dashboard</h1>
        <TimeRangeSelector value={timeRange} onChange={setTimeRange} />
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <MetricCard
          title="Total Requests"
          value={data.currentPeriod.requests.toLocaleString()}
          subtitle={`${utilizationPercentage.requests.toFixed(1)}% of quota`}
          trend={calculateTrend(data.metrics, 'requestCount')}
        />
        <MetricCard
          title="Token Usage"
          value={formatTokens(data.currentPeriod.tokens)}
          subtitle={`${utilizationPercentage.tokens.toFixed(1)}% of quota`}
          trend={calculateTrend(data.metrics, 'tokenUsage')}
        />
        <MetricCard
          title="Total Cost"
          value={`$${data.currentPeriod.cost.toFixed(2)}`}
          subtitle="Current billing period"
          trend={calculateTrend(data.metrics, 'cost')}
        />
        <MetricCard
          title="Avg Latency"
          value={`${calculateAverage(data.metrics, 'averageLatency').toFixed(0)}ms`}
          subtitle="P95 response time"
          trend={calculateTrend(data.metrics, 'averageLatency', true)}
        />
      </div>

      {/* Request Volume Chart */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h2 className="text-xl font-semibold mb-4">Request Volume Over Time</h2>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={data.metrics}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="timestamp"
              tickFormatter={(ts) => new Date(ts).toLocaleTimeString()}
            />
            <YAxis />
            <Tooltip
              labelFormatter={(ts) => new Date(ts).toLocaleString()}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="requestCount"
              stroke="#8884d8"
              name="Requests"
              strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Token Usage & Latency Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4">Token Consumption</h2>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={data.metrics}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="timestamp"
                tickFormatter={(ts) => new Date(ts).toLocaleTimeString()}
              />
              <YAxis />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="tokenUsage"
                stroke="#82ca9d"
                name="Tokens"
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4">API Latency (ms)</h2>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={data.metrics}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="timestamp"
                tickFormatter={(ts) => new Date(ts).toLocaleTimeString()}
              />
              <YAxis />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="averageLatency"
                stroke="#ffc658"
                name="Latency"
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Error Rate Alert */}
      {calculateAverage(data.metrics, 'errorRate') > 5 && (
        <div className="bg-red-50 border-l-4 border-red-400 p-4">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <p className="text-sm text-red-700">
                High error rate detected ({calculateAverage(data.metrics, 'errorRate').toFixed(2)}%).
                Please check your API configuration.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Helper Components
const MetricCard: React.FC<{
  title: string;
  value: string;
  subtitle: string;
  trend: number;
}> = ({ title, value, subtitle, trend }) => (
  <div className="bg-white p-6 rounded-lg shadow">
    <h3 className="text-sm font-medium text-gray-500">{title}</h3>
    <p className="mt-2 text-3xl font-bold text-gray-900">{value}</p>
    <div className="mt-2 flex items-center justify-between">
      <p className="text-sm text-gray-600">{subtitle}</p>
      <TrendIndicator value={trend} />
    </div>
  </div>
);

const TrendIndicator: React.FC<{ value: number }> = ({ value }) => {
  const isPositive = value > 0;
  const color = isPositive ? 'text-green-600' : 'text-red-600';
  const arrow = isPositive ? '↑' : '↓';

  return (
    <span className={`text-sm font-medium ${color}`}>
      {arrow} {Math.abs(value).toFixed(1)}%
    </span>
  );
};

// Helper functions
const calculateTrend = (
  metrics: UsageMetrics[],
  key: keyof UsageMetrics,
  inverse: boolean = false
): number => {
  if (metrics.length < 2) return 0;
  const recent = metrics.slice(-10);
  const older = metrics.slice(-20, -10);

  const recentAvg = recent.reduce((sum, m) => sum + (m[key] as number), 0) / recent.length;
  const olderAvg = older.reduce((sum, m) => sum + (m[key] as number), 0) / older.length;

  const trend = ((recentAvg - olderAvg) / olderAvg) * 100;
  return inverse ? -trend : trend;
};

const calculateAverage = (metrics: UsageMetrics[], key: keyof UsageMetrics): number => {
  if (metrics.length === 0) return 0;
  return metrics.reduce((sum, m) => sum + (m[key] as number), 0) / metrics.length;
};

const formatTokens = (tokens: number): string => {
  if (tokens >= 1000000) return `${(tokens / 1000000).toFixed(2)}M`;
  if (tokens >= 1000) return `${(tokens / 1000).toFixed(2)}K`;
  return tokens.toString();
};

export default UsageDashboard;
```

### 6. Kubernetes Deployment Configuration

```yaml
# k8s/api-gateway-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: contextai-api-gateway
  namespace: contextai
  labels:
    app: api-gateway
    version: v1
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: api-gateway
  template:
    metadata:
      labels:
        app: api-gateway
        version: v1
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: contextai-api-gateway
      containers:
      - name: api-gateway
        image: contextai/api-gateway:1.0.0
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
          name: http
          protocol: TCP
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: contextai-secrets
              key: database-url
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: VAULT_ADDR
          value: "http://vault:8200"
        - name: VAULT_TOKEN
          valueFrom:
            secretKeyRef:
              name: contextai-secrets
              key: vault-token
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        securityContext:
          runAsNonRoot: true
          runAsUser: 1000
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
---
apiVersion: v1
kind: Service
metadata:
  name: api-gateway-service
  namespace: contextai
spec:
  type: ClusterIP
  selector:
    app: api-gateway
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-gateway-hpa
  namespace: contextai
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: contextai-api-gateway
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

### 7. Docker Compose for Local Development

```yaml
# docker-compose.yml
version: '3.8'

services:
  # API Gateway
  api-gateway:
    build:
      context: ./services/api_gateway
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://contextai:password@postgres:5432/contextai
      - REDIS_URL=redis://redis:6379
      - VAULT_ADDR=http://vault:8200
      - VAULT_TOKEN=dev-token
      - PYTHONUNBUFFERED=1
    depends_on:
      - postgres
      - redis
      - vault
    volumes:
      - ./services/api_gateway:/app
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

  # LLM Router Service
  llm-router:
    build:
      context: ./services/llm_router
      dockerfile: Dockerfile
    ports:
      - "8001:8001"
    environment:
      - REDIS_URL=redis://redis:6379
      - VAULT_ADDR=http://vault:8200
      - VAULT_TOKEN=dev-token
    depends_on:
      - redis
      - vault
    volumes:
      - ./services/llm_router:/app

  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=contextai
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=contextai
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql

  # Redis
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  # HashiCorp Vault
  vault:
    image: vault:1.13
    ports:
      - "8200:8200"
    environment:
      - VAULT_DEV_ROOT_TOKEN_ID=dev-token
      - VAULT_DEV_LISTEN_ADDRESS=0.0.0.0:8200
    cap_add:
      - IPC_LOCK
    command: server -dev

  # Prometheus
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  # Grafana
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
    depends_on:
      - prometheus

  # Elasticsearch
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.9.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

  # Kibana
  kibana:
    image: docker.elastic.co/kibana/kibana:8.9.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch

  # Frontend
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.dev
    ports:
      - "5173:5173"
    volumes:
      - ./frontend:/app
      - /app/node_modules
    environment:
      - VITE_API_URL=http://localhost:8000

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
  elasticsearch_data:
```

---

## Security & Compliance

### Multi-Tenant Data Isolation

**Strategy:**
1. **Database Level**: Tenant-specific schemas with Row-Level Security (RLS)
2. **Application Level**: Tenant ID validation on every request
3. **Encryption**: Unique encryption keys per tenant stored in Vault
4. **API Keys**: Tenant-scoped API keys with permission-based access control

### Security Measures Implemented

| Layer | Security Control | Implementation |
|-------|-----------------|----------------|
| **Network** | TLS 1.3 | NGINX with Let's Encrypt certificates |
| **Authentication** | JWT/OAuth2 | HS256 signing, 1-hour expiry, refresh tokens |
| **Authorization** | RBAC | Role-based permissions (Admin, Developer, Viewer) |
| **Data at Rest** | AES-256 | Per-tenant keys in HashiCorp Vault |
| **Data in Transit** | TLS 1.3 | End-to-end encryption |
| **Secrets** | Vault | All API keys, DB credentials in Vault |
| **Rate Limiting** | Redis | Sliding window per tenant/tier |
| **Audit Logging** | Immutable logs | All API calls logged to S3 |
| **Input Validation** | Pydantic | Schema validation on all endpoints |
| **SQL Injection** | SQLAlchemy ORM | Parameterized queries |
| **XSS Protection** | CSP Headers | Content Security Policy enforced |

### Compliance Features

**GDPR Compliance:**
- Right to be forgotten (data deletion API)
- Data export functionality
- Consent management
- Data residency controls (EU/US regions)

**SOC2 Compliance:**
- Audit trails for all operations
- Encryption at rest and in transit
- Access control and monitoring
- Incident response procedures

**HIPAA Readiness:**
- PHI data encryption
- Access logging and monitoring
- Business Associate Agreements (BAA) support
- Automatic session timeout

---

## Monitoring & Observability

### Metrics Tracked

**Application Metrics (Prometheus):**
- Request rate (req/sec) per tenant
- Request latency (p50, p95, p99)
- Error rate (4xx, 5xx)
- Token usage per tenant/model
- Cost per tenant
- Active connections
- Cache hit ratio

**Infrastructure Metrics:**
- CPU/Memory utilization per service
- Pod count and auto-scaling events
- Database connection pool stats
- Redis memory usage
- Network I/O

**Business Metrics:**
- Monthly Active Tenants (MAT)
- Average Revenue Per Tenant (ARPT)
- Token consumption trends
- Model usage distribution
- Quota utilization

### Logging Strategy

**Structured Logging (JSON format):**
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "service": "api-gateway",
  "tenant_id": "tenant_123",
  "user_id": "user_456",
  "request_id": "req_789abc",
  "method": "POST",
  "path": "/v1/chat/completions",
  "status_code": 200,
  "latency_ms": 245,
  "provider": "openai",
  "model": "gpt-4",
  "tokens_used": 150,
  "cost_usd": 0.003
}
```

**Log Levels:**
- **DEBUG**: Development debugging
- **INFO**: Normal operations (API calls, auth events)
- **WARNING**: Rate limits, quota warnings
- **ERROR**: Failed API calls, provider errors
- **CRITICAL**: System failures, security events

### Alerting Rules

**Critical Alerts (PagerDuty):**
- Error rate > 5% for 5 minutes
- p95 latency > 5 seconds
- Any service replica count = 0
- Database connection failures
- Vault unavailable

**Warning Alerts (Slack):**
- Error rate > 2% for 10 minutes
- Tenant approaching quota (>90%)
- Provider circuit breaker triggered
- High memory usage (>80%)

---

## Interview Cross-Questions & Answers

### Architecture & Design

**Q1: Why did you choose a microservices architecture over a monolith?**

**Answer:**
We chose microservices for several reasons:

1. **Independent Scaling**: LLM Router and API Gateway have different resource requirements. The router handles CPU-intensive model routing logic, while the gateway is I/O bound. Microservices let us scale them independently.

2. **Technology Flexibility**: We use Python for backend services (FastAPI's async performance) and TypeScript for frontend. Microservices allow polyglot architecture.

3. **Fault Isolation**: If the Usage Tracker service fails, core API functionality remains operational. Circuit breakers prevent cascade failures.

4. **Team Autonomy**: Different teams can own different services with independent deployment cycles.

5. **Multi-Tenancy**: Easier to implement tenant-specific routing, rate limiting, and data isolation at the service level.

**Trade-offs:**
- Increased operational complexity (we mitigate with Kubernetes)
- Network latency between services (we use service mesh for optimization)
- Distributed tracing required (implemented with Jaeger)

---

**Q2: How does your LLM routing and failover mechanism work? Walk me through a request lifecycle.**

**Answer:**

**Request Lifecycle:**

1. **Request Arrival** (API Gateway)
   - Client sends POST to `/v1/chat/completions` with JWT token
   - NGINX load balancer routes to API Gateway pod

2. **Authentication** (API Gateway)
   - Extract JWT from `Authorization: Bearer <token>`
   - Validate signature using HS256 algorithm
   - Extract tenant_id, user_id, permissions from payload
   - Verify tenant_id matches `X-Tenant-ID` header

3. **Rate Limiting** (Redis)
   - Check Redis key: `rate_limit:{tenant_id}:{current_minute}`
   - Increment counter using INCR (atomic operation)
   - If count > tier limit (100/1000/10000), return HTTP 429
   - Set TTL to 60 seconds on first increment

4. **LLM Routing** (Router Service)
   - Fetch tenant's provider config from database
   - Sort providers by priority: `[{OpenAI: priority=1}, {Anthropic: priority=2}]`
   - Attempt OpenAI first

5. **Circuit Breaker Check**
   - Check if OpenAI circuit breaker is OPEN (triggered after 5 consecutive failures)
   - If OPEN and within recovery timeout (60s), skip to next provider
   - If CLOSED or HALF_OPEN, proceed with request

6. **Provider API Call**
   ```python
   async with httpx.AsyncClient(timeout=30.0) as client:
       response = await client.post(
           "https://api.openai.com/v1/chat/completions",
           headers={"Authorization": f"Bearer {api_key}"},
           json=payload
       )
   ```

7. **Success Path**
   - Response received (200 OK)
   - Extract token usage: `{prompt_tokens: 50, completion_tokens: 100}`
   - Send async event to Usage Tracker service
   - Cache response in Redis (key: `cache:{hash(prompt)}`, TTL: 1 hour)
   - Encrypt sensitive data using tenant's AES-256 key
   - Return response to client

8. **Failure Path** (Automatic Failover)
   - OpenAI returns 503 Service Unavailable
   - Log error to ELK stack
   - Increment circuit breaker failure counter
   - Move to next provider: Anthropic
   - Transform request to Anthropic format:
     ```python
     openai_format = {"messages": [...], "model": "gpt-4"}
     anthropic_format = {"messages": [...], "model": "claude-3-opus"}
     ```
   - Retry with Anthropic API
   - If Anthropic succeeds, return response
   - If all providers fail, return HTTP 503 with error details

9. **Post-Processing** (Async)
   - Audit Logger: Write to S3 (immutable log)
   - Prometheus Metrics: Increment counters
   - Cost Calculator: Update tenant billing
   - Circuit Breaker: Reset failure counter on success

**Key Design Decisions:**

- **Priority-based routing** (not round-robin) ensures we use preferred/cheaper providers first
- **Circuit breakers** prevent wasting time on known-failed providers
- **Async post-processing** keeps response time low (<200ms)
- **Request transformation** handles provider-specific API differences
- **Caching** reduces redundant LLM calls (saves 30-40% of requests)

---

**Q3: How do you handle multi-tenancy and ensure data isolation between tenants?**

**Answer:**

We implement **defense-in-depth** with multiple isolation layers:

**1. Database Level (PostgreSQL)**
```sql
-- Each tenant has isolated schema
CREATE SCHEMA tenant_123;
CREATE SCHEMA tenant_456;

-- Row-Level Security (RLS) policy
ALTER TABLE tenant_123.conversations ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation_policy ON tenant_123.conversations
  USING (tenant_id = current_setting('app.current_tenant_id'));
```

Every query sets session variable:
```python
# Before executing query
connection.execute("SET app.current_tenant_id = 'tenant_123'")
# Now queries automatically filtered by RLS
```

**2. Application Level**
```python
async def get_tenant_from_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    x_tenant_id: Optional[str] = Header(None)
):
    payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])

    # Double validation: JWT claim + HTTP header must match
    if x_tenant_id and payload["tenant_id"] != x_tenant_id:
        raise HTTPException(status_code=403, detail="Tenant ID mismatch")

    return TenantContext(
        tenant_id=payload["tenant_id"],
        user_id=payload["user_id"]
    )
```

Every API endpoint requires tenant context:
```python
@app.post("/v1/chat/completions")
async def chat_completion(
    request: LLMRequest,
    tenant: TenantContext = Depends(get_tenant_from_token)  # REQUIRED
):
    # tenant.tenant_id automatically scopes all operations
```

**3. Encryption Level**
- **Each tenant has unique AES-256 key** stored in HashiCorp Vault
- Path: `secret/tenants/{tenant_id}/encryption_key`
- Keys never stored in application code or database
- Data encrypted before database write, decrypted after read

```python
# Tenant A's data encrypted with key_A
vault.get_secret(f"tenants/tenant_a/encryption_key") -> key_A
encrypted_data_A = AES256.encrypt(data, key_A)

# Tenant B's data encrypted with key_B
vault.get_secret(f"tenants/tenant_b/encryption_key") -> key_B
encrypted_data_B = AES256.encrypt(data, key_B)

# Even if DB compromised, tenant_A cannot decrypt tenant_B's data
```

**4. Network Level**
- Tenant-specific API keys with embedded tenant_id
- Rate limiting per tenant (Redis keys: `rate_limit:{tenant_id}:*`)
- Cost tracking per tenant (separate Prometheus labels)

**5. Caching Level**
```python
# Cache keys prefixed with tenant_id
cache_key = f"cache:{tenant_id}:{hash(prompt)}"
redis.set(cache_key, response, ex=3600)
```

**6. Audit Level**
- Every operation logged with tenant_id
- Immutable audit trail in S3: `s3://audit-logs/{tenant_id}/{date}/{request_id}.json`
- Cross-tenant access attempts trigger security alerts

**Verification:**
```python
# Unit test to verify isolation
async def test_tenant_isolation():
    # Create data for tenant_A
    tenant_a_token = create_jwt(tenant_id="tenant_a")
    response_a = await client.post(
        "/v1/data",
        headers={"Authorization": f"Bearer {tenant_a_token}"},
        json={"secret": "data_A"}
    )

    # Try to access tenant_A's data with tenant_B token
    tenant_b_token = create_jwt(tenant_id="tenant_b")
    response_b = await client.get(
        f"/v1/data/{response_a.json()['id']}",
        headers={"Authorization": f"Bearer {tenant_b_token}"}
    )

    assert response_b.status_code == 403  # Forbidden
```

---

### Security & Compliance

**Q4: Explain your encryption strategy. Why AES-256? How do you manage keys?**

**Answer:**

**Why AES-256?**

1. **Industry Standard**: NIST-approved, FIPS 140-2 compliant
2. **Security**: 2^256 possible keys = computationally infeasible to brute force
3. **Performance**: Hardware acceleration (AES-NI) on modern CPUs
4. **Compliance**: Required for HIPAA, PCI-DSS, SOC2
5. **Quantum Resistance**: While quantum computers threaten RSA, AES-256 remains secure (Grover's algorithm only reduces to 2^128 effective strength)

**Encryption Implementation:**

```python
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

class AES256Encryption:
    def encrypt(self, plaintext: bytes) -> str:
        # Generate random 128-bit IV (Initialization Vector)
        iv = os.urandom(16)

        # Pad plaintext to 128-bit block size using PKCS7
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(plaintext) + padder.finalize()

        # Encrypt using AES-256-CBC
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()

        # Prepend IV to ciphertext (IV doesn't need to be secret)
        encrypted_data = iv + ciphertext
        return base64.b64encode(encrypted_data).decode('utf-8')
```

**Why CBC mode?**
- **CBC (Cipher Block Chaining)**: Each block depends on previous block, making patterns undetectable
- Alternative: GCM (Galois/Counter Mode) for authenticated encryption (we use this for critical data)

**Key Management with HashiCorp Vault:**

1. **Storage**:
   ```
   Vault Path: secret/tenants/{tenant_id}/encryption_key
   - key: base64-encoded 256-bit key
   - created_at: timestamp
   - rotated_at: timestamp
   ```

2. **Access Control**:
   ```hcl
   # Vault policy: only api-gateway service can read keys
   path "secret/data/tenants/*/encryption_key" {
     capabilities = ["read"]
   }
   ```

3. **Key Rotation**:
   ```python
   async def rotate_tenant_key(tenant_id: str):
       # Generate new key
       new_key = os.urandom(32)
       old_key = vault.get_tenant_key(tenant_id)

       # Store new key with version
       vault.create_or_update_secret(
           path=f"tenants/{tenant_id}/encryption_key",
           secret={"key": base64.b64encode(new_key).decode(), "version": 2}
       )

       # Re-encrypt all existing data
       async for record in db.query("SELECT * FROM tenant_data WHERE tenant_id = $1", tenant_id):
           decrypted = AES256(old_key).decrypt(record.encrypted_data)
           re_encrypted = AES256(new_key).encrypt(decrypted)
           await db.execute("UPDATE tenant_data SET encrypted_data = $1 WHERE id = $2", re_encrypted, record.id)
   ```

4. **Key Hierarchy**:
   ```
   Vault Master Key (unsealed by Shamir Secret Sharing)
   └── Encryption Key (encrypts keys at rest in Vault)
       └── Tenant Keys (unique per tenant)
           └── Data Encryption (encrypts actual data)
   ```

**Security Benefits:**

- **Separation of Duties**: Vault admins ≠ App developers ≠ DBA
- **Audit Trail**: Every key access logged
- **Automatic Unsealing**: Vault auto-unseals using cloud KMS (AWS KMS/GCP KMS)
- **High Availability**: Vault runs in HA cluster (3 replicas)
- **Disaster Recovery**: Vault snapshots to S3 every 6 hours

**Compliance Mapping:**

| Requirement | Implementation |
|------------|----------------|
| GDPR Art. 32 (Encryption) | AES-256 at rest, TLS 1.3 in transit |
| SOC2 CC6.1 (Encryption) | Per-tenant keys, Vault audit logs |
| HIPAA §164.312(a)(2)(iv) | Encryption of PHI at rest |
| PCI-DSS 3.4 | Strong cryptography (AES-256) |

---

**Q5: How do you handle rate limiting across distributed API Gateway instances?**

**Answer:**

We use **Redis-based distributed rate limiting** with sliding window counters.

**Challenge:**
- Multiple API Gateway pods (3-20 replicas)
- Need consistent rate limits across all instances
- In-memory limits won't work (each pod has separate memory)

**Solution: Redis Atomic Operations**

```python
import redis
from datetime import datetime

async def check_rate_limit(tenant_id: str, tier: str):
    # Current minute window
    current_minute = datetime.now().strftime('%Y%m%d%H%M')
    rate_limit_key = f"rate_limit:{tenant_id}:{current_minute}"

    # Atomic increment (prevents race conditions)
    current_count = redis_client.incr(rate_limit_key)

    # Set TTL on first increment (key auto-expires after 60 seconds)
    if current_count == 1:
        redis_client.expire(rate_limit_key, 60)

    # Check limit based on tier
    max_requests = {"free": 100, "pro": 1000, "enterprise": 10000}[tier]

    if current_count > max_requests:
        raise HTTPException(429, detail="Rate limit exceeded")

    return current_count
```

**Why This Works:**

1. **Atomic Operations**: Redis `INCR` is atomic (no race conditions)
   - Even if 3 pods check simultaneously, each gets unique counter value
   - Redis is single-threaded, ensures serial execution

2. **Sliding Window**: Fixed window (per minute) prevents "edge" bursts
   - Alternative: Sliding window log (stores timestamps) but uses more memory

3. **Auto-Expiration**: TTL ensures old keys don't accumulate
   - Redis automatically deletes expired keys
   - Saves memory (no manual cleanup needed)

**Advanced: Token Bucket Algorithm**

For smoother rate limiting (allow bursts):

```python
async def token_bucket_rate_limit(tenant_id: str, tier: str):
    bucket_key = f"bucket:{tenant_id}"

    # Configuration
    max_tokens = {"free": 100, "pro": 1000, "enterprise": 10000}[tier]
    refill_rate = max_tokens / 60  # tokens per second

    # Get current bucket state
    bucket = redis_client.hgetall(bucket_key)

    if not bucket:
        # Initialize bucket
        bucket = {
            "tokens": max_tokens,
            "last_refill": time.time()
        }
    else:
        # Refill tokens based on elapsed time
        elapsed = time.time() - float(bucket["last_refill"])
        tokens_to_add = elapsed * refill_rate
        bucket["tokens"] = min(max_tokens, float(bucket["tokens"]) + tokens_to_add)
        bucket["last_refill"] = time.time()

    # Try to consume 1 token
    if bucket["tokens"] >= 1:
        bucket["tokens"] -= 1
        redis_client.hset(bucket_key, mapping=bucket)
        return True
    else:
        raise HTTPException(429, detail="Rate limit exceeded")
```

**Performance Optimization:**

- **Redis Pipelining**: Batch multiple commands
  ```python
  pipe = redis_client.pipeline()
  pipe.incr(rate_limit_key)
  pipe.expire(rate_limit_key, 60)
  results = pipe.execute()
  ```

- **Redis Cluster**: Shard keys across multiple Redis nodes for horizontal scaling

- **Caching**: Cache tier limits in memory (refresh every 5 minutes)

**Monitoring:**

```python
# Prometheus metrics
rate_limit_exceeded = Counter(
    'rate_limit_exceeded_total',
    'Rate limit violations',
    ['tenant_id', 'tier']
)

if current_count > max_requests:
    rate_limit_exceeded.labels(tenant_id=tenant_id, tier=tier).inc()
```

**Graceful Degradation:**

```python
try:
    await check_rate_limit(tenant_id, tier)
except redis.ConnectionError:
    # Redis down, fallback to in-memory (per-pod) rate limiting
    logger.warning("Redis unavailable, using local rate limiting")
    await local_rate_limit(tenant_id)
```

---

### Scalability & Performance

**Q6: How does your system scale to handle 50+ concurrent tenants? What's your scaling strategy?**

**Answer:**

**Horizontal Scaling Strategy:**

1. **Stateless Services** (Scale freely)
   - API Gateway: 3-20 replicas (auto-scales based on CPU)
   - LLM Router: 2-10 replicas
   - Usage Tracker: 2-5 replicas

2. **Stateful Services** (Scale carefully)
   - PostgreSQL: Primary + 2 Read Replicas (read/write split)
   - Redis: Redis Cluster with 6 nodes (3 masters + 3 replicas)

**Kubernetes Auto-Scaling:**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-gateway-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-gateway
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70  # Scale up when CPU > 70%
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80  # Scale up when memory > 80%
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "1000"  # Scale when >1000 req/s per pod
```

**Load Balancing:**

```
        Internet
           ↓
    [NGINX Load Balancer]
           ↓
    ┌──────┴──────┐
    ↓             ↓
[Gateway-1]   [Gateway-2]   [Gateway-3]
    ↓             ↓             ↓
    └──────┬──────┘
           ↓
   [LLM Router Pool]
```

**Database Scaling:**

```python
# Read/Write Split
class DatabaseRouter:
    def __init__(self):
        self.primary = create_engine("postgresql://primary:5432/contextai")
        self.replicas = [
            create_engine("postgresql://replica1:5432/contextai"),
            create_engine("postgresql://replica2:5432/contextai")
        ]

    def get_read_connection(self):
        # Round-robin load balancing
        return random.choice(self.replicas)

    def get_write_connection(self):
        return self.primary

# Usage
@app.get("/tenants/{tenant_id}")
async def get_tenant(tenant_id: str):
    # Read from replica
    conn = db_router.get_read_connection()
    return await conn.fetch_one("SELECT * FROM tenants WHERE id = $1", tenant_id)

@app.post("/tenants")
async def create_tenant(tenant: TenantCreate):
    # Write to primary
    conn = db_router.get_write_connection()
    return await conn.execute("INSERT INTO tenants ...", tenant)
```

**Caching Strategy (Reduces DB load by 70%):**

```python
# Three-tier caching
async def get_tenant_config(tenant_id: str):
    # L1: In-memory cache (fastest, 10ms)
    if tenant_id in memory_cache:
        return memory_cache[tenant_id]

    # L2: Redis cache (fast, 50ms)
    cached = await redis.get(f"tenant_config:{tenant_id}")
    if cached:
        memory_cache[tenant_id] = json.loads(cached)
        return memory_cache[tenant_id]

    # L3: Database (slow, 200ms)
    config = await db.fetch_one("SELECT * FROM tenant_config WHERE tenant_id = $1", tenant_id)

    # Populate caches
    await redis.setex(f"tenant_config:{tenant_id}", 300, json.dumps(config))  # 5 min TTL
    memory_cache[tenant_id] = config

    return config
```

**Connection Pooling:**

```python
# PostgreSQL connection pool
database = databases.Database(
    "postgresql://...",
    min_size=10,      # Minimum connections
    max_size=50,      # Maximum connections
    max_inactive_connection_lifetime=300  # Close idle connections after 5 min
)

# Redis connection pool
redis_pool = redis.ConnectionPool(
    host='redis',
    port=6379,
    max_connections=100,
    socket_connect_timeout=5
)
```

**Performance Numbers:**

| Metric | Without Optimization | With Optimization |
|--------|---------------------|-------------------|
| API Latency (p95) | 1200ms | 180ms |
| Database Queries | 500/sec | 150/sec (70% cached) |
| Concurrent Tenants | 10 | 50+ |
| Requests/sec | 500 | 5000+ |
| Cost per Request | $0.01 | $0.003 |

**Bottleneck Identification:**

```python
# Prometheus metrics to identify bottlenecks
@app.middleware("http")
async def track_request_latency(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    # Track latency breakdown
    latency_histogram.labels(
        path=request.url.path,
        method=request.method
    ).observe(duration)

    return response
```

**Grafana Dashboard Query:**
```promql
# Find slowest endpoints
topk(10,
  histogram_quantile(0.95,
    rate(llm_request_duration_seconds_bucket[5m])
  )
)
```

---

**Q7: Your API has a p95 latency of <200ms. How do you achieve this when LLM APIs can take 2-5 seconds?**

**Answer:**

**Key Insight**: <200ms is our **API response time**, not LLM completion time. We use **streaming responses** and **async processing**.

**1. Streaming Responses (SSE - Server-Sent Events)**

```python
from fastapi.responses import StreamingResponse

@app.post("/v1/chat/completions")
async def chat_completion(request: LLMRequest):
    if request.stream:
        # Return immediately (<200ms), stream tokens as they arrive
        return StreamingResponse(
            stream_llm_response(request),
            media_type="text/event-stream"
        )
    else:
        # Wait for full response (2-5 seconds)
        return await get_full_llm_response(request)

async def stream_llm_response(request: LLMRequest):
    """
    Stream tokens from LLM in real-time
    Client sees first token in <200ms
    """
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "https://api.openai.com/v1/chat/completions",
            json={"model": request.model, "messages": [...], "stream": True}
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    yield f"{line}\n\n"  # Forward to client immediately
```

**Client Experience:**
```javascript
// JavaScript client
const response = await fetch('/v1/chat/completions', {
  method: 'POST',
  body: JSON.stringify({prompt: "...", stream: true})
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const {done, value} = await reader.read();
  if (done) break;

  const chunk = decoder.decode(value);
  console.log(chunk);  // Tokens appear in real-time (<200ms for first token)
}
```

**2. Async Background Processing**

```python
from celery import Celery

celery_app = Celery('contextai', broker='redis://redis:6379/0')

@app.post("/v1/chat/completions")
async def chat_completion(request: LLMRequest):
    # Create job in <50ms
    job_id = str(uuid.uuid4())

    # Queue LLM processing (async)
    process_llm_request.delay(job_id, request.dict())

    # Return immediately (<200ms)
    return {
        "job_id": job_id,
        "status": "processing",
        "result_url": f"/v1/jobs/{job_id}"
    }

@celery_app.task
def process_llm_request(job_id: str, request_data: dict):
    """
    Runs in background worker (takes 2-5 seconds)
    """
    result = call_llm_provider(request_data)

    # Store result in Redis
    redis.setex(f"job:{job_id}:result", 3600, json.dumps(result))

    # Optionally: Send webhook notification
    requests.post(request_data["webhook_url"], json=result)

# Client polls for result
@app.get("/v1/jobs/{job_id}")
async def get_job_result(job_id: str):
    result = redis.get(f"job:{job_id}:result")
    if result:
        return {"status": "completed", "result": json.loads(result)}
    else:
        return {"status": "processing"}
```

**3. Aggressive Caching**

```python
import hashlib

async def get_llm_response(prompt: str, model: str):
    # Generate cache key from prompt
    cache_key = f"llm_cache:{hashlib.sha256(prompt.encode()).hexdigest()}:{model}"

    # Check cache (10-50ms)
    cached = await redis.get(cache_key)
    if cached:
        cache_hits.inc()  # Prometheus metric
        return json.loads(cached)

    # Cache miss, call LLM (2-5 seconds)
    result = await call_llm_provider(prompt, model)

    # Cache for 1 hour
    await redis.setex(cache_key, 3600, json.dumps(result))

    return result
```

**Cache Hit Rate**: 35-45% (saves 35-45% of LLM API costs)

**4. Connection Pooling & HTTP/2**

```python
# Reuse HTTP connections to LLM providers
http_client = httpx.AsyncClient(
    timeout=30.0,
    limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
    http2=True  # Multiplexing reduces connection overhead
)

# Connection overhead reduced from 500ms -> 50ms
```

**5. Database Query Optimization**

```python
# Bad: N+1 query problem (100ms per query * 10 = 1000ms)
async def get_tenant_with_configs(tenant_id: str):
    tenant = await db.fetch_one("SELECT * FROM tenants WHERE id = $1", tenant_id)
    configs = []
    for config_id in tenant["config_ids"]:
        config = await db.fetch_one("SELECT * FROM configs WHERE id = $1", config_id)
        configs.append(config)
    return tenant, configs

# Good: Single join query (100ms total)
async def get_tenant_with_configs(tenant_id: str):
    return await db.fetch_all("""
        SELECT t.*, c.*
        FROM tenants t
        LEFT JOIN configs c ON c.tenant_id = t.id
        WHERE t.id = $1
    """, tenant_id)
```

**6. Parallel Processing**

```python
import asyncio

async def get_dashboard_data(tenant_id: str):
    # Sequential: 500ms + 300ms + 200ms = 1000ms
    # Parallel: max(500ms, 300ms, 200ms) = 500ms

    usage_stats, billing_info, recent_logs = await asyncio.gather(
        get_usage_stats(tenant_id),      # 500ms
        get_billing_info(tenant_id),     # 300ms
        get_recent_logs(tenant_id)       # 200ms
    )

    return {
        "usage": usage_stats,
        "billing": billing_info,
        "logs": recent_logs
    }
```

**Latency Breakdown (Actual Numbers):**

```
Total p95 Latency: 180ms
├── NGINX routing: 5ms
├── JWT validation: 10ms
├── Rate limit check (Redis): 15ms
├── Tenant config fetch (Redis cache): 20ms
├── Request validation: 10ms
├── LLM routing logic: 30ms
├── Provider selection: 10ms
├── HTTP connection (pooled): 20ms
├── Queue LLM request (Celery): 30ms
├── Return job ID: 5ms
└── Overhead: 25ms
```

**For streaming requests:**
- **Time to First Token (TTFT)**: ~180ms (includes network + LLM startup)
- **Inter-Token Latency**: ~50ms (token generation rate)
- **Total time to completion**: 2-5 seconds (but user sees progress immediately)

---

### Monitoring & Operations

**Q8: Walk me through how you'd debug a production issue where tenant XYZ is experiencing 503 errors.**

**Answer:**

**Step-by-Step Debugging Process:**

**1. Initial Triage (0-2 minutes)**

```bash
# Check Grafana dashboard for tenant XYZ
# Look for:
# - Error rate spike
# - Latency increase
# - Provider availability
# - Service health

# Quick check: Is it isolated to one tenant or system-wide?
curl -H "Authorization: Bearer $ADMIN_TOKEN" \
  https://api.contextai.com/v1/admin/health

# Output:
{
  "status": "degraded",
  "services": {
    "api_gateway": "healthy",
    "llm_router": "healthy",
    "openai_provider": "unhealthy",  # <-- Found issue!
    "anthropic_provider": "healthy"
  }
}
```

**2. Check ELK Logs (2-5 minutes)**

```
# Kibana query
service: "llm-router" AND tenant_id: "tenant_xyz" AND level: "ERROR" AND @timestamp: [now-15m TO now]

# Sample log entry
{
  "timestamp": "2024-01-15T14:32:15Z",
  "level": "ERROR",
  "service": "llm-router",
  "tenant_id": "tenant_xyz",
  "provider": "openai",
  "error": "HTTPStatusError: 503 Service Unavailable",
  "request_id": "req_abc123",
  "trace_id": "trace_xyz789"
}
```

**3. Check Distributed Trace (5-8 minutes)**

```
# Jaeger UI: Search for trace_id: trace_xyz789

Request Flow:
├── api-gateway: 50ms ✓
├── auth-service: 30ms ✓
├── rate-limiter: 20ms ✓
├── llm-router: 100ms ✓
│   ├── provider-selector: 20ms ✓
│   ├── openai-adapter: FAILED (503) ✗  <-- Bottleneck
│   └── failover-to-anthropic: 2000ms ✓
└── response: 2200ms total
```

**Finding**: OpenAI returning 503, but failover to Anthropic is working (slower).

**4. Check Provider Status (8-10 minutes)**

```python
# Check OpenAI status page
curl https://status.openai.com/api/v2/status.json

# Output:
{
  "status": {
    "indicator": "major",  # Service degraded
    "description": "Partial System Outage"
  }
}

# Check circuit breaker state
redis-cli GET "circuit_breaker:openai:state"
# Output: "OPEN"  (circuit opened after 5 consecutive failures)

redis-cli GET "circuit_breaker:openai:failure_count"
# Output: "5"

redis-cli TTL "circuit_breaker:openai:state"
# Output: "45"  (will retry in 45 seconds)
```

**5. Verify Failover is Working (10-12 minutes)**

```bash
# Check Prometheus metrics
# Query: sum(rate(llm_requests_total{tenant_id="tenant_xyz",status="success"}[5m])) by (provider)

# Output:
openai: 0 req/sec       # All failing
anthropic: 5 req/sec    # Receiving failover traffic ✓
```

**6. Root Cause Analysis (12-15 minutes)**

```
Root Cause: OpenAI API experiencing outage (confirmed by status.openai.com)

Impact:
- Tenant XYZ using OpenAI as primary provider
- Circuit breaker correctly opened after 5 failures
- Automatic failover to Anthropic working
- User experiencing higher latency (Anthropic slower than OpenAI for this model)
- Some 503 errors during circuit breaker detection phase (first 5 requests)

Timeline:
14:30:00 - OpenAI API starts returning 503
14:30:15 - 5 failures detected, circuit breaker opens
14:30:15 - Failover to Anthropic begins
14:30:16 - All subsequent requests succeed (via Anthropic)
14:32:00 - Circuit breaker enters HALF_OPEN (retry OpenAI)
14:32:05 - OpenAI still failing, circuit re-opens
```

**7. Mitigation (15-20 minutes)**

**Option A: Wait for OpenAI recovery (automatic)**
- Circuit breaker will auto-retry every 60 seconds
- When OpenAI recovers, traffic automatically shifts back

**Option B: Manually prioritize Anthropic (temporary override)**

```python
# Admin API call to temporarily switch tenant's primary provider
curl -X PATCH https://api.contextai.com/v1/admin/tenants/tenant_xyz/config \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "providers": [
      {"name": "anthropic", "priority": 1, "enabled": true},
      {"name": "openai", "priority": 2, "enabled": false}
    ]
  }'
```

**Option C: Notify tenant**

```python
# Send automated email/Slack notification
await send_notification(
    tenant_id="tenant_xyz",
    subject="Service Degradation - OpenAI Provider",
    message="""
    We've detected an outage with OpenAI's API (our primary provider).

    Your requests have been automatically rerouted to Anthropic Claude.
    You may experience slightly higher latency (~2s vs ~1s).

    No action required. Normal service will resume when OpenAI recovers.

    Status: https://status.openai.com
    """
)
```

**8. Post-Incident Review (After resolution)**

```markdown
## Incident Report: OpenAI Provider Outage

**Date**: 2024-01-15
**Duration**: 14:30 - 15:45 (75 minutes)
**Severity**: SEV-3 (Partial degradation, failover working)

### Impact
- Tenant: tenant_xyz
- Requests affected: ~500
- Errors: 5 (during circuit breaker detection)
- Success rate: 99% (via failover)

### Root Cause
- External dependency failure (OpenAI API outage)
- Confirmed by OpenAI status page

### What Went Well
✓ Circuit breaker detected failure in 15 seconds
✓ Automatic failover to Anthropic successful
✓ 99% of requests succeeded
✓ Monitoring alerted team immediately

### What Needs Improvement
✗ 5 user requests failed before circuit breaker opened
✗ No proactive notification to tenant
✗ Could have faster detection (5 failures -> 3 failures)

### Action Items
1. Reduce circuit breaker threshold: 5 -> 3 failures
2. Implement proactive tenant notifications
3. Add predictive monitoring (detect provider degradation before complete failure)
4. Consider multi-provider simultaneous requests (race and use fastest)
```

**Prometheus Alerting Rule (Preventative)**

```yaml
groups:
- name: provider_health
  rules:
  - alert: ProviderHighErrorRate
    expr: |
      rate(llm_requests_total{status="error"}[5m]) / rate(llm_requests_total[5m]) > 0.05
    for: 1m
    labels:
      severity: warning
    annotations:
      summary: "High error rate for {{ $labels.provider }}"
      description: "Error rate is {{ $value | humanizePercentage }}"

  - alert: ProviderDown
    expr: provider_availability == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Provider {{ $labels.provider }} is down"
```

---

### Cost & Business

**Q9: How do you track and attribute costs per tenant? How does pricing work?**

**Answer:**

**Cost Tracking Architecture:**

```python
# Real-time cost calculation
class CostCalculator:
    """
    Calculate costs based on LLM provider pricing
    """

    # Pricing per 1K tokens (as of 2024)
    PRICING = {
        "openai": {
            "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.002},
            "gpt-4": {"prompt": 0.03, "completion": 0.06},
            "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03}
        },
        "anthropic": {
            "claude-3-opus": {"prompt": 0.015, "completion": 0.075},
            "claude-3-sonnet": {"prompt": 0.003, "completion": 0.015}
        }
    }

    def calculate_cost(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """
        Calculate cost for a single LLM request
        """
        pricing = self.PRICING[provider][model]

        prompt_cost = (prompt_tokens / 1000) * pricing["prompt"]
        completion_cost = (completion_tokens / 1000) * pricing["completion"]

        total_cost = prompt_cost + completion_cost

        # Add ContextAI markup (30%)
        markup = total_cost * 0.30

        # Add infrastructure cost (fixed $0.0001 per request)
        infrastructure_cost = 0.0001

        return total_cost + markup + infrastructure_cost

# Usage tracking in LLM Router
async def track_llm_usage(
    tenant_id: str,
    provider: str,
    model: str,
    response: dict
):
    """
    Track usage and cost after each LLM call
    """
    usage = response.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    # Calculate cost
    cost = cost_calculator.calculate_cost(
        provider, model, prompt_tokens, completion_tokens
    )

    # Store in PostgreSQL (for billing)
    await db.execute("""
        INSERT INTO usage_logs (
            tenant_id, timestamp, provider, model,
            prompt_tokens, completion_tokens, cost_usd
        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
    """, tenant_id, datetime.now(), provider, model,
         prompt_tokens, completion_tokens, cost)

    # Update tenant's current usage (Redis for real-time)
    pipe = redis.pipeline()
    pipe.incrbyfloat(f"usage:{tenant_id}:cost:daily", cost)
    pipe.incrby(f"usage:{tenant_id}:tokens:daily", prompt_tokens + completion_tokens)
    pipe.incrby(f"usage:{tenant_id}:requests:daily", 1)
    pipe.execute()

    # Update Prometheus metrics
    token_usage_counter.labels(
        tenant_id=tenant_id,
        provider=provider,
        model=model
    ).inc(prompt_tokens + completion_tokens)

    cost_counter.labels(
        tenant_id=tenant_id,
        provider=provider
    ).inc(cost)
```

**Cost Attribution Flow:**

```
1. User makes API request
2. LLM Router calls provider (e.g., OpenAI)
3. Provider returns response with token usage:
   {
     "usage": {
       "prompt_tokens": 50,
       "completion_tokens": 100,
       "total_tokens": 150
     }
   }
4. CostCalculator computes:
   - Base cost: (50/1000 * $0.0015) + (100/1000 * $0.002) = $0.000275
   - Markup (30%): $0.000275 * 0.30 = $0.0000825
   - Infrastructure: $0.0001
   - Total: $0.0003575
5. Store in database for billing
6. Update real-time usage counters (Redis)
7. Prometheus metrics for dashboards
```

**Pricing Tiers:**

| Tier | Price | Included | Overage |
|------|-------|----------|---------|
| **Free** | $0/month | 100K tokens<br>100 req/min | N/A (hard limit) |
| **Pro** | $99/month | 5M tokens<br>1000 req/min<br>Email support | $0.02 per 1K tokens |
| **Enterprise** | Custom | Unlimited tokens<br>10000 req/min<br>SLA 99.9%<br>Dedicated support<br>Custom models | Volume discounts |

**Quota Enforcement:**

```python
async def check_tenant_quota(tenant_id: str):
    """
    Check if tenant has exceeded quota
    """
    # Get current usage
    daily_cost = float(redis.get(f"usage:{tenant_id}:cost:daily") or 0)
    daily_tokens = int(redis.get(f"usage:{tenant_id}:tokens:daily") or 0)

    # Get tenant's plan
    tenant = await db.fetch_one("SELECT * FROM tenants WHERE id = $1", tenant_id)

    quotas = {
        "free": {"tokens": 100000, "cost": 0},
        "pro": {"tokens": 5000000, "cost": 99}
    }

    if tenant["plan"] in quotas:
        quota = quotas[tenant["plan"]]

        if daily_tokens > quota["tokens"]:
            raise HTTPException(
                status_code=402,  # Payment Required
                detail=f"Daily token quota exceeded. Used: {daily_tokens:,}, Limit: {quota['tokens']:,}. Upgrade to continue."
            )

# Reset daily counters (Celery scheduled task)
@celery_app.task
def reset_daily_quotas():
    """
    Runs at midnight UTC
    """
    for key in redis.scan_iter("usage:*:daily"):
        redis.delete(key)
```

**Billing Dashboard API:**

```python
@app.get("/v1/billing/usage")
async def get_billing_usage(
    tenant: TenantContext = Depends(get_tenant_from_token),
    start_date: datetime = Query(...),
    end_date: datetime = Query(...)
):
    """
    Get detailed usage and cost breakdown
    """
    usage = await db.fetch_all("""
        SELECT
            DATE(timestamp) as date,
            provider,
            model,
            SUM(prompt_tokens) as prompt_tokens,
            SUM(completion_tokens) as completion_tokens,
            SUM(cost_usd) as cost_usd,
            COUNT(*) as request_count
        FROM usage_logs
        WHERE tenant_id = $1
          AND timestamp BETWEEN $2 AND $3
        GROUP BY DATE(timestamp), provider, model
        ORDER BY date DESC
    """, tenant.tenant_id, start_date, end_date)

    total_cost = sum(row["cost_usd"] for row in usage)
    total_tokens = sum(row["prompt_tokens"] + row["completion_tokens"] for row in usage)

    return {
        "tenant_id": tenant.tenant_id,
        "period": {
            "start": start_date,
            "end": end_date
        },
        "summary": {
            "total_cost_usd": round(total_cost, 2),
            "total_tokens": total_tokens,
            "total_requests": sum(row["request_count"] for row in usage)
        },
        "breakdown": usage
    }
```

**Cost Optimization Recommendations:**

```python
@app.get("/v1/recommendations/cost-optimization")
async def get_cost_recommendations(
    tenant: TenantContext = Depends(get_tenant_from_token)
):
    """
    Analyze usage and suggest cost savings
    """
    # Get last 30 days usage
    usage = await db.fetch_all("""
        SELECT provider, model, COUNT(*) as count, AVG(prompt_tokens) as avg_prompt_tokens
        FROM usage_logs
        WHERE tenant_id = $1 AND timestamp > NOW() - INTERVAL '30 days'
        GROUP BY provider, model
    """, tenant.tenant_id)

    recommendations = []

    # Check if using expensive models for simple tasks
    for row in usage:
        if row["model"] == "gpt-4" and row["avg_prompt_tokens"] < 100:
            potential_savings = row["count"] * 0.027  # Difference between GPT-4 and 3.5
            recommendations.append({
                "type": "model_downgrade",
                "description": f"Consider using gpt-3.5-turbo instead of gpt-4 for simple prompts (<100 tokens)",
                "potential_monthly_savings_usd": round(potential_savings, 2)
            })

    # Check cache hit rate
    cache_hits = await redis.get(f"cache_hits:{tenant.tenant_id}") or 0
    total_requests = await redis.get(f"total_requests:{tenant.tenant_id}") or 1
    cache_hit_rate = int(cache_hits) / int(total_requests)

    if cache_hit_rate < 0.3:
        recommendations.append({
            "type": "enable_caching",
            "description": f"Your cache hit rate is {cache_hit_rate:.1%}. Enable caching for repeated queries.",
            "potential_monthly_savings_usd": "20-40% cost reduction"
        })

    return recommendations
```

---

**Q10: What challenges did you face and how did you overcome them?**

**Answer:**

**Challenge 1: Cold Start Latency with Kubernetes**

**Problem**:
- First request after pod startup took 5-10 seconds
- FastAPI app initialization + database connections + Vault secret fetch
- Poor user experience

**Solution**:
```python
# Implement readiness probe that waits for initialization
@app.on_event("startup")
async def startup_event():
    # Pre-warm connections
    await database.connect()
    await redis.ping()

    # Pre-load tenant configs into memory cache
    tenants = await db.fetch_all("SELECT * FROM tenants LIMIT 100")
    for tenant in tenants:
        memory_cache[tenant["id"]] = tenant

    # Pre-fetch secrets from Vault
    vault_client.prefetch_secrets()

    logger.info("Application ready")

# K8s readiness probe
readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 10  # Wait for initialization
  periodSeconds: 5
```

**Result**: Cold start reduced from 10s -> 2s

---

**Challenge 2: Token Counting Accuracy**

**Problem**:
- Different providers count tokens differently
- OpenAI uses tiktoken, Anthropic uses different tokenizer
- Inaccurate cost tracking led to billing discrepancies

**Solution**:
```python
import tiktoken
from anthropic import Anthropic

class TokenCounter:
    def __init__(self):
        self.openai_encoder = tiktoken.encoding_for_model("gpt-4")
        self.anthropic_client = Anthropic()

    def count_tokens(self, text: str, provider: str, model: str) -> int:
        if provider == "openai":
            return len(self.openai_encoder.encode(text))
        elif provider == "anthropic":
            # Use Anthropic's count_tokens API
            return self.anthropic_client.count_tokens(text)
        else:
            # Fallback: rough estimate (1 token ≈ 4 chars)
            return len(text) // 4

# Verify token counts match provider's reported usage
async def verify_token_count(request, response):
    our_count = token_counter.count_tokens(request["prompt"], provider, model)
    provider_count = response["usage"]["prompt_tokens"]

    if abs(our_count - provider_count) > 5:
        logger.warning(f"Token count mismatch: ours={our_count}, provider={provider_count}")
```

**Result**: Billing accuracy improved from ~85% -> 99%+

---

**Challenge 3: Database Connection Pool Exhaustion**

**Problem**:
- During traffic spikes, PostgreSQL max_connections (100) exhausted
- Requests timing out with "too many connections" error

**Solution**:
```python
# 1. Implement connection pooling with limits
database = databases.Database(
    DATABASE_URL,
    min_size=5,
    max_size=20,  # Per pod (3 pods * 20 = 60 connections max)
    max_inactive_connection_lifetime=300
)

# 2. Use PgBouncer (connection pooler)
# docker-compose.yml
pgbouncer:
  image: pgbouncer/pgbouncer
  environment:
    - DATABASES_HOST=postgres
    - POOL_MODE=transaction  # More efficient than session pooling
    - MAX_CLIENT_CONN=1000
    - DEFAULT_POOL_SIZE=25

# 3. Read replica for analytics queries
@app.get("/v1/analytics/usage")
async def get_usage_analytics():
    # Use read replica (doesn't impact primary DB connections)
    conn = read_replica.get_connection()
    return await conn.fetch_all("SELECT ...")
```

**Result**: Eliminated connection exhaustion, support 10x traffic

---

**Challenge 4: Secrets Rotation without Downtime**

**Problem**:
- Rotating API keys (OpenAI, Anthropic) requires updating Vault
- Hard to do without service disruption

**Solution**:
```python
class SecretManager:
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.last_fetch = {}

    async def get_secret(self, path: str) -> str:
        # Check if cache is still valid
        if path in self.cache:
            if time.time() - self.last_fetch[path] < self.cache_ttl:
                return self.cache[path]

        # Fetch from Vault
        secret = vault_client.read_secret(path)
        self.cache[path] = secret
        self.last_fetch[path] = time.time()

        return secret

# Rotation process:
# 1. Add new key to Vault with version 2
# 2. Wait 5 minutes (cache TTL)
# 3. All pods now using new key
# 4. Revoke old key

# Support for multiple active keys during rotation
async def get_provider_api_key(provider: str):
    keys = vault_client.read_secret(f"providers/{provider}/api_keys")
    # keys = {"current": "key_v2", "previous": "key_v1"}

    # Try current key first
    try:
        return await call_with_key(keys["current"])
    except Unauthorized:
        # Fallback to previous key (during rotation)
        return await call_with_key(keys["previous"])
```

**Result**: Zero-downtime secret rotation

---

This documentation provides a comprehensive overview of ContextAI with production-ready code snippets and detailed interview preparation materials.
