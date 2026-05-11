# TXT2CREATE - System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Web App    │  │  Mobile App  │  │   REST API   │  │  WebSocket   │   │
│  │   (React)    │  │ (React Native)│ │   Clients    │  │   Clients    │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            API GATEWAY LAYER                                 │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  NGINX / Kong API Gateway                                          │    │
│  │  - JWT Verification  - Rate Limiting  - Load Balancing            │    │
│  │  - SSL/TLS Termination  - Request Routing                          │    │
│  └────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          APPLICATION LAYER                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐         │
│  │  Auth Service    │  │  FastAPI Service │  │  Task Orchestrator│         │
│  │  (FastAPI)       │  │  (Python)        │  │  (Celery/Temporal)│         │
│  │  - User Auth     │  │  - Request Val.  │  │  - Job Scheduling │         │
│  │  - JWT Tokens    │  │  - Token Check   │  │  - Pipeline Mgmt  │         │
│  │  - Token Mgmt    │  │  - Cost Calc     │  │  - Token Deduct   │         │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘         │
│  ┌──────────────────┐                                                       │
│  │  WebSocket Server│                                                       │
│  │  (FastAPI/Socket.io)│                                                    │
│  │  - Real-time Updates                                                     │
│  │  - Progress Tracking                                                     │
│  └──────────────────┘                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PROCESSING LAYER                                    │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ Text-to-    │ │ Text-to-    │ │ Text-to-    │ │ Video       │          │
│  │ Image       │ │ Video       │ │ Audio       │ │ Captioning  │          │
│  │ Pipeline    │ │ Pipeline    │ │ Pipeline    │ │ Pipeline    │          │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘          │
│                   ┌─────────────┐                                           │
│                   │ Virtual     │                                           │
│                   │ Avatar Gen  │                                           │
│                   └─────────────┘                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          AI INFERENCE LAYER                                  │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │                    MODEL SERVING CLUSTER                          │      │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │      │
│  │  │  TorchServe  │  │    vLLM      │  │  TorchServe  │           │      │
│  │  │   Stable     │  │   LLM CoT    │  │   VAE Video  │           │      │
│  │  │  Diffusion   │  │   Reasoning  │  │  Compression │           │      │
│  │  │  (GPU Pool)  │  │  (GPU Pool)  │  │  (GPU Pool)  │           │      │
│  │  └──────────────┘  └──────────────┘  └──────────────┘           │      │
│  │  ┌──────────────┐  ┌──────────────┐                             │      │
│  │  │  Audio Model │  │ Avatar Model │                             │      │
│  │  │  (Whisper/   │  │  (Custom/    │                             │      │
│  │  │   MusicGen)  │  │   GAN-based) │                             │      │
│  │  └──────────────┘  └──────────────┘                             │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────┐        │
│  │  Model Router / Load Balancer (Nginx or Custom)                │        │
│  │  - GPU utilization-based routing                                │        │
│  │  - Model versioning                                              │        │
│  └────────────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA & CACHE LAYER                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  PostgreSQL  │  │    Redis     │  │   MongoDB    │  │  S3/MinIO    │  │
│  │  - User Data │  │  - Cache     │  │  - Metadata  │  │  - Generated │  │
│  │  - Jobs      │  │  - Sessions  │  │  - Logs      │  │    Media     │  │
│  │  - Analytics │  │  - Queue     │  │  - Configs   │  │  - Models    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MONITORING & OBSERVABILITY                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Prometheus   │  │   Grafana    │  │  ELK Stack   │  │   Jaeger     │  │
│  │ - Metrics    │  │  - Dashboards│  │  - Logs      │  │  - Tracing   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Architecture Principles

1. **Microservices Architecture**: Each pipeline is independently deployable and scalable
2. **Async Processing**: Long-running AI tasks are handled asynchronously via task queues
3. **Token-Based Usage**: Fair resource allocation and cost tracking via token system
4. **JWT Authentication**: Stateless, secure user authentication and authorization
5. **GPU Resource Pooling**: Efficient GPU utilization across multiple models
6. **Horizontal Scalability**: All services can scale independently based on load
7. **Fault Tolerance**: Retry mechanisms, circuit breakers, and graceful degradation
8. **Observability**: Comprehensive monitoring, logging, and tracing

## Authentication & Token Flow

### Complete Request Flow with Authentication

```
┌────────────────────────────────────────────────────────────────────┐
│                     AUTHENTICATED REQUEST FLOW                      │
└────────────────────────────────────────────────────────────────────┘

1. User Authenticates (One-time)
   User Login → Auth Service
   ↓
   Verify Credentials (bcrypt password check)
   ↓
   Generate JWT Tokens:
   - Access Token (15 min, for API calls)
   - Refresh Token (7 days, for token renewal)
   ↓
   Store session in Redis
   ↓
   Return tokens to client

2. User Initiates Generation (With JWT)
   Browser sends: POST /api/v1/generate/video
   Headers: Authorization: Bearer <access_token>
   Body: {prompt, duration, resolution}
   ↓
   API Gateway: Verify JWT signature
   ↓
   FastAPI Service:
     - Decode JWT → Extract user_id, tier, permissions
     - Calculate tokens needed (e.g., 350 tokens for 10s video)
     - Check permission ("generate_video")
   ↓
   Token Service (PostgreSQL):
     BEGIN TRANSACTION;
     - Lock user's token row (SELECT ... FOR UPDATE)
     - Check balance >= 350 tokens
     - Deduct 350 tokens
     - Log usage
     COMMIT;
   ↓
   If sufficient tokens:
     - Create job in database
     - Queue Celery task
     - Return job_id

   If insufficient tokens:
     - Rollback transaction
     - Return 402 Payment Required
     - Show current balance & purchase options

3. Processing & Token Management
   Celery Worker:
   ↓
   Process video generation
   ↓
   If SUCCESS:
     - Upload result to S3
     - Mark job complete
     - Tokens already deducted (no refund)

   If FAILURE:
     - Refund 350 tokens (automatic)
     - Mark job failed
     - Log refund reason

4. Result Delivery
   WebSocket → Real-time notification
   User receives: {job_id, status, video_url}
```

### Token Calculation Example

```
Video Generation Request:
- Duration: 10 seconds
- Resolution: 720p
- Model: Stable Diffusion XL
- Priority: Normal

Token Calculation:
base_cost = 100 (text-to-video base)
resolution_mult = 2.0 (720p)
duration_cost = 1 + (10 × 2) = 21
model_mult = 1.5 (SD XL)
priority_mult = 1.0 (normal)

total_tokens = 100 × 2.0 × 21 × 1.5 × 1.0 = 6,300 tokens
(This seems high - likely need to adjust formula)

Revised formula:
total_tokens = 100 + (duration × 20) × resolution_mult × model_mult
             = 100 + (10 × 20) × 2.0 × 1.5
             = 100 + 200 × 2.0 × 1.5
             = 100 + 600 = 700 tokens

This is more reasonable for a 10-second video.
User with PRO tier (1,000 tokens/month) can generate 1-2 videos.
```

## Infrastructure Layout

### Kubernetes Cluster Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      KUBERNETES CLUSTER                          │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Namespace: api-services                               │    │
│  │  - FastAPI Pods (CPU, 3 replicas)                      │    │
│  │  - WebSocket Pods (CPU, 2 replicas)                    │    │
│  │  - HPA: CPU > 70% → scale up                           │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Namespace: task-orchestration                         │    │
│  │  - Celery Workers (CPU, auto-scale 2-10)               │    │
│  │  - Celery Beat Scheduler (1 replica)                   │    │
│  │  - Redis as message broker                             │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Namespace: ml-inference (GPU Node Pool)               │    │
│  │  - TorchServe StatefulSet (GPU, 4 replicas)            │    │
│  │  - vLLM StatefulSet (GPU, 2 replicas)                  │    │
│  │  - VAE Service (GPU, 2 replicas)                       │    │
│  │  - GPU Resource Limits: 1 GPU per pod                  │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Namespace: data-layer                                 │    │
│  │  - PostgreSQL StatefulSet (1 primary, 2 replicas)      │    │
│  │  - Redis Cluster (3 masters, 3 slaves)                 │    │
│  │  - MongoDB ReplicaSet (3 nodes)                        │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Namespace: monitoring                                 │    │
│  │  - Prometheus (2 replicas)                             │    │
│  │  - Grafana (2 replicas)                                │    │
│  │  - Elasticsearch (3 nodes)                             │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Network Architecture

```
Internet
   ↓
[CloudFlare CDN] → Caching, DDoS Protection
   ↓
[Load Balancer] → AWS ALB / GCP Load Balancer
   ↓
[API Gateway] → Kong/NGINX (SSL Termination, Rate Limiting)
   ↓
[Service Mesh] → Istio (Optional, for advanced routing)
   ↓
[Kubernetes Services] → Internal routing within cluster
```

## Security Architecture

1. **Authentication**: JWT-based authentication with refresh tokens
2. **Authorization**: RBAC (Role-Based Access Control)
3. **API Security**: Rate limiting, request validation, CORS
4. **Data Encryption**: TLS in transit, encryption at rest for S3
5. **Secrets Management**: Kubernetes Secrets / HashiCorp Vault
6. **Network Security**: VPC, private subnets for databases, security groups

## Scalability Strategy

### Horizontal Scaling Triggers

| Component | Metric | Threshold | Action |
|-----------|--------|-----------|--------|
| FastAPI | CPU > 70% | 30s | Add pod (max 10) |
| TorchServe | GPU Util > 85% | 60s | Add GPU pod (max 8) |
| Celery Workers | Queue depth > 100 | 30s | Add worker (max 20) |
| Redis | Memory > 80% | Manual | Add cluster node |
| PostgreSQL | Connections > 80% | Manual | Read replica |

### Cost Optimization

- **Spot Instances**: Use for non-critical batch processing
- **GPU Sharing**: Multiple small models on single GPU with MPS
- **Auto-scaling**: Scale down during low traffic
- **CDN**: Serve generated content via CloudFlare
- **Result Caching**: Cache identical prompts in Redis (TTL: 24h)

---

This architecture supports:
- ✅ High availability (99.9% uptime)
- ✅ Horizontal scalability
- ✅ Multi-region deployment ready
- ✅ Cost-effective GPU utilization
- ✅ Real-time monitoring and alerting
- ✅ Disaster recovery and backups
