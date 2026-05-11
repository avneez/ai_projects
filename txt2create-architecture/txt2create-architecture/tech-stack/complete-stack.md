# Complete Tech Stack - TXT2CREATE

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                       │
├─────────────────────────────────────────────────────────────┤
│                     APPLICATION LAYER                        │
├─────────────────────────────────────────────────────────────┤
│                     AI/ML INFERENCE LAYER                    │
├─────────────────────────────────────────────────────────────┤
│                     DATA LAYER                               │
├─────────────────────────────────────────────────────────────┤
│                     INFRASTRUCTURE LAYER                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. PRESENTATION LAYER

### Frontend

| Component | Technology | Purpose | Version |
|-----------|-----------|---------|---------|
| **Web App** | React 18 | User interface | 18.2.0 |
| **State Management** | Redux Toolkit | Global state | 2.0.1 |
| **UI Framework** | Material-UI / Tailwind | Component library | Latest |
| **HTTP Client** | Axios | API communication | 1.6.0 |
| **WebSocket** | Socket.IO Client | Real-time updates | 4.6.0 |
| **Media Player** | Video.js | Video playback | 8.10.0 |
| **Audio Player** | Howler.js | Audio playback | 2.2.4 |
| **3D Viewer** | Three.js | Avatar preview | r160 |
| **Form Validation** | React Hook Form + Zod | Input validation | Latest |
| **Build Tool** | Vite | Fast bundling | 5.0.0 |

### Mobile App (Optional)

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Framework** | React Native | Cross-platform | 0.73.0 |
| **Navigation** | React Navigation | Routing | 6.1.0 |
| **State** | Redux Toolkit | State management | 2.0.1 |

---

## 2. APPLICATION LAYER

### API Services

| Component | Technology | Purpose | Configuration |
|-----------|-----------|---------|---------------|
| **API Framework** | FastAPI | REST API server | Python 3.11 |
| **ASGI Server** | Uvicorn | Production server | Workers: 4-8 |
| **WebSocket** | FastAPI WebSocket | Real-time comms | Socket.IO compatible |
| **API Documentation** | OpenAPI/Swagger | Auto-generated docs | Built-in FastAPI |
| **Validation** | Pydantic v2 | Request/response validation | Type-safe |
| **Authentication** | JWT + OAuth2 | User auth | PyJWT 2.8.0 |
| **Rate Limiting** | SlowAPI | API throttling | Redis-backed |
| **CORS** | FastAPI CORS Middleware | Cross-origin | Configured domains |

### Task Orchestration

| Component | Technology | Purpose | Configuration |
|-----------|-----------|---------|---------------|
| **Task Queue** | Celery 5.3 | Async job processing | Python |
| **Message Broker** | Redis 7.2 | Task queue backend | Persistent |
| **Result Backend** | Redis | Task results storage | TTL: 24h |
| **Beat Scheduler** | Celery Beat | Periodic tasks | Single instance |
| **Monitoring** | Flower | Celery dashboard | Web UI |
| **Workflow Engine** | Temporal (Alternative) | Complex workflows | Optional |

### API Gateway

| Component | Technology | Purpose | Configuration |
|-----------|-----------|---------|---------------|
| **Gateway** | Kong / NGINX | API gateway | Load balancing |
| **Rate Limiting** | Kong plugins | Request throttling | Per-user limits |
| **SSL/TLS** | Let's Encrypt | HTTPS termination | Auto-renewal |
| **Auth Plugin** | JWT plugin | Token validation | Stateless |
| **Logging** | Kong logging | Request logs | JSON format |

---

## 3. AI/ML INFERENCE LAYER

### Model Serving Infrastructure

| Component | Technology | Purpose | Hardware |
|-----------|-----------|---------|----------|
| **Primary Serving** | TorchServe 0.9 | PyTorch model serving | GPU |
| **LLM Serving** | vLLM 0.2.7 | Fast LLM inference | GPU (A100) |
| **Model Format** | TorchScript / ONNX | Optimized models | - |
| **Batch Inference** | TorchServe batching | Throughput optimization | Batch size: 4-8 |
| **Model Versioning** | TorchServe MAR | Version management | Rollback support |

### AI Models by Pipeline

#### Text-to-Image

| Model | Version | Purpose | Hardware | Memory |
|-------|---------|---------|----------|--------|
| **Stable Diffusion XL** | 1.0 | High-quality images | 1x A100 40GB | 12-15 GB |
| **Stable Diffusion 2.1** | 2.1 | Fast generation | 1x A100 40GB | 8-10 GB |
| **VAE** | Automatic (SD built-in) | Latent decoding | Same GPU | 2-3 GB |
| **CLIP** | ViT-L/14 | Text encoding | CPU/GPU | 1-2 GB |
| **Safety Checker** | CompVis | NSFW filtering | CPU | 500 MB |

#### Text-to-Video

| Model | Version | Purpose | Hardware | Memory |
|-------|---------|---------|----------|--------|
| **Stable Diffusion** | XL 1.0 | Keyframe generation | 1x A100 | 12 GB |
| **FILM** | Large | Frame interpolation | 1x A100 | 8 GB |
| **Custom VAE** | v1.0 | Video compression | 1x A100 | 10 GB |
| **AnimateDiff** | v2 (Alternative) | Motion generation | 2x A100 | 24 GB |

#### Text-to-Audio

| Model | Version | Purpose | Hardware | Memory |
|-------|---------|---------|----------|--------|
| **MusicGen** | Large (3.3B) | Music generation | 1x A100 | 18 GB |
| **AudioLDM 2** | Base | Sound effects | 1x A100 | 10 GB |
| **Bark** | Large | Text-to-speech | 1x A100 | 12 GB |
| **Whisper** | Large v3 | Audio transcription | 1x A100 | 10 GB |

#### Video Captioning

| Model | Version | Purpose | Hardware | Memory |
|-------|---------|---------|----------|--------|
| **BLIP-2** | FlanT5-XL | Image captioning | 1x A100 | 15 GB |
| **LLaVA** | 1.6 (34B) | Visual reasoning | 2x A100 | 70 GB |
| **Whisper** | Large v3 | Audio transcription | 1x A100 | 10 GB |
| **Llama 3** | 8B/70B | Temporal reasoning | 1-2x A100 | 16-140 GB |

#### Virtual Avatar

| Model | Version | Purpose | Hardware | Memory |
|-------|---------|---------|----------|--------|
| **Stable Diffusion** | Custom (portrait) | Face generation | 1x A100 | 12 GB |
| **PIFuHD** | - | 2D to 3D mesh | 1x A100 | 8 GB |
| **MediaPipe** | Latest | Pose/rigging | CPU | 500 MB |
| **Instant-NGP** | - | NeRF (optional) | 1x A100 | 10 GB |

### LLM Chain-of-Thought

| Model | Version | Purpose | Hardware | Memory |
|-------|---------|---------|----------|--------|
| **Llama 3** | 8B Instruct | Prompt enhancement | 1x A100 | 16 GB |
| **Llama 3** | 70B Instruct | Complex reasoning | 2x A100 | 140 GB |
| **Mistral** | 7B Instruct | Fast reasoning | 1x A100 | 14 GB |
| **GPT-4** | API (Fallback) | Best quality | API | N/A |

### Model Optimization Techniques

| Technique | Tool/Method | Benefit | Trade-off |
|-----------|-------------|---------|-----------|
| **Quantization** | BitsAndBytes INT8 | 2x faster, 50% memory | Slight quality loss |
| **FP16 Precision** | PyTorch AMP | 2x faster | Minimal quality impact |
| **Flash Attention** | xFormers | 40% faster attention | None |
| **Compile** | torch.compile | 20-30% speedup | First run slower |
| **LoRA Fine-tuning** | PEFT | Custom models | Requires training |
| **Model Pruning** | PyTorch pruning | Smaller models | Quality depends |

---

## 4. DATA LAYER

### Databases

| Component | Technology | Purpose | Configuration |
|-----------|-----------|---------|---------------|
| **Primary DB** | PostgreSQL 16 | Relational data | 1 primary, 2 replicas |
| **Schema** | SQLAlchemy ORM | Database ORM | Alembic migrations |
| **Caching** | Redis 7.2 | In-memory cache | Cluster mode (3+3) |
| **NoSQL** | MongoDB 7 | Logs, metadata | ReplicaSet (3 nodes) |
| **Time-series** | TimescaleDB | Metrics, analytics | PostgreSQL extension |
| **Search** | Elasticsearch 8 | Full-text search | 3-node cluster |

### Data Models (PostgreSQL)

```sql
-- Users
users (
  id UUID PRIMARY KEY,
  email VARCHAR UNIQUE,
  password_hash VARCHAR,
  created_at TIMESTAMP,
  subscription_tier VARCHAR
)

-- Jobs
jobs (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users,
  pipeline_type VARCHAR,
  status VARCHAR,
  prompt TEXT,
  parameters JSONB,
  result_url VARCHAR,
  created_at TIMESTAMP,
  completed_at TIMESTAMP,
  error_message TEXT
)

-- Generated Assets
assets (
  id UUID PRIMARY KEY,
  job_id UUID REFERENCES jobs,
  asset_type VARCHAR,
  s3_key VARCHAR,
  file_size BIGINT,
  metadata JSONB
)

-- Usage Tracking
usage_logs (
  id BIGSERIAL PRIMARY KEY,
  user_id UUID,
  timestamp TIMESTAMP,
  pipeline VARCHAR,
  gpu_seconds DECIMAL,
  cost DECIMAL
)
```

### Object Storage

| Component | Technology | Purpose | Configuration |
|-----------|-----------|---------|---------------|
| **Primary Storage** | AWS S3 / MinIO | Generated media | Multi-region |
| **CDN** | CloudFlare R2 + CDN | Content delivery | Global edge |
| **Backup** | S3 Glacier | Long-term archival | Lifecycle policies |
| **Bucket Structure** | - | Organization | See below |

**S3 Bucket Structure:**
```
txt2create-media/
├── images/{user_id}/{job_id}/image.png
├── videos/{user_id}/{job_id}/video.mp4
├── audio/{user_id}/{job_id}/audio.mp3
├── avatars/{user_id}/{job_id}/avatar.fbx
└── thumbnails/{user_id}/{job_id}/thumb.jpg

txt2create-models/
├── stable-diffusion/xl-1.0.safetensors
├── llm/llama-3-8b/
└── custom-loras/{user_id}/
```

### Caching Strategy (Redis)

| Key Pattern | Purpose | TTL |
|-------------|---------|-----|
| `prompt:{hash}` | Generated image cache | 24h |
| `session:{user_id}` | User session data | 7d |
| `queue:celery` | Task queue | Persistent |
| `ratelimit:{user_id}:{endpoint}` | Rate limiting | 1h |
| `model:embeddings:{hash}` | Cached embeddings | 168h (7d) |

---

## 5. INFRASTRUCTURE LAYER

### Container Orchestration

| Component | Technology | Purpose | Configuration |
|-----------|-----------|---------|---------------|
| **Orchestrator** | Kubernetes 1.28 | Container orchestration | 3 control planes |
| **Container Runtime** | containerd | Container runtime | CRI compatible |
| **Service Mesh** | Istio (Optional) | Traffic management | mTLS enabled |
| **Ingress** | NGINX Ingress | Load balancing | SSL termination |
| **Storage** | Persistent Volumes | Stateful storage | EBS/GCP PD |

### Kubernetes Node Pools

| Node Pool | Instance Type | GPUs | vCPUs | RAM | Nodes |
|-----------|--------------|------|-------|-----|-------|
| **API Tier** | n1-standard-4 | 0 | 4 | 15 GB | 3-10 (autoscale) |
| **Worker Tier** | n1-standard-8 | 0 | 8 | 30 GB | 5-20 (autoscale) |
| **GPU Inference** | n1-standard-16 + A100 | 1 | 16 | 60 GB | 4-8 (autoscale) |
| **GPU LLM** | a2-highgpu-2g + 2xA100 | 2 | 24 | 170 GB | 2-4 (autoscale) |
| **Database** | n1-highmem-8 | 0 | 8 | 52 GB | 3 (static) |

### GPU Infrastructure

| Component | Specification | Quantity | Purpose |
|-----------|--------------|----------|---------|
| **Primary GPU** | NVIDIA A100 40GB | 8-16 | SD, VAE, Audio models |
| **LLM GPU** | NVIDIA A100 80GB | 2-4 | Llama 70B, vLLM |
| **Backup GPU** | NVIDIA L4 24GB | 4-8 | Cost-effective inference |
| **GPU Driver** | NVIDIA Driver 535+ | - | CUDA 12.2 support |
| **CUDA** | CUDA 12.2 | - | PyTorch 2.1+ |
| **GPU Sharing** | NVIDIA MPS / MIG | - | Multi-tenant GPUs |

### Cloud Provider Stack

**Option A: AWS**
```
┌─────────────────────────────────────┐
│ AWS Cloud                           │
├─────────────────────────────────────┤
│ Compute                             │
│ - EKS (Kubernetes)                  │
│ - EC2 P4 instances (A100)           │
│ - EC2 G5 instances (A10G backup)    │
│                                     │
│ Storage                             │
│ - S3 (media storage)                │
│ - EBS (persistent volumes)          │
│ - EFS (shared storage)              │
│                                     │
│ Database                            │
│ - RDS PostgreSQL (managed)          │
│ - ElastiCache Redis (managed)       │
│ - DocumentDB (MongoDB compatible)   │
│                                     │
│ Networking                          │
│ - VPC (private network)             │
│ - ALB (load balancer)               │
│ - CloudFront (CDN)                  │
│ - Route 53 (DNS)                    │
│                                     │
│ Monitoring                          │
│ - CloudWatch (logs, metrics)        │
│ - X-Ray (tracing)                   │
└─────────────────────────────────────┘
```

**Option B: GCP**
```
┌─────────────────────────────────────┐
│ Google Cloud Platform               │
├─────────────────────────────────────┤
│ Compute                             │
│ - GKE (Kubernetes)                  │
│ - A2 instances (A100)               │
│ - G2 instances (L4)                 │
│                                     │
│ Storage                             │
│ - Cloud Storage (media)             │
│ - Persistent Disk (SSD)             │
│ - Filestore (NFS)                   │
│                                     │
│ Database                            │
│ - Cloud SQL PostgreSQL              │
│ - Memorystore Redis                 │
│ - Firestore                         │
│                                     │
│ Networking                          │
│ - VPC                               │
│ - Cloud Load Balancing              │
│ - Cloud CDN                         │
│ - Cloud DNS                         │
│                                     │
│ Monitoring                          │
│ - Cloud Monitoring                  │
│ - Cloud Logging                     │
│ - Cloud Trace                       │
└─────────────────────────────────────┘
```

### CI/CD Pipeline

| Stage | Tool | Purpose |
|-------|------|---------|
| **Version Control** | GitHub / GitLab | Source code |
| **CI** | GitHub Actions / GitLab CI | Automated testing |
| **Container Registry** | Docker Hub / GCR | Image storage |
| **CD** | ArgoCD / Flux | GitOps deployment |
| **Testing** | Pytest, Jest | Unit/integration tests |
| **Code Quality** | SonarQube | Static analysis |
| **Security Scan** | Trivy, Snyk | Vulnerability scanning |

---

## 6. MONITORING & OBSERVABILITY

### Monitoring Stack

| Component | Technology | Purpose | Storage |
|-----------|-----------|---------|---------|
| **Metrics** | Prometheus | Time-series metrics | 30d retention |
| **Visualization** | Grafana | Dashboards | - |
| **Logging** | Elasticsearch | Centralized logs | 7d retention |
| **Log Shipper** | Fluentd / Fluent Bit | Log aggregation | - |
| **APM** | Jaeger / Tempo | Distributed tracing | 7d retention |
| **Alerting** | AlertManager | Alert routing | - |
| **Uptime** | UptimeRobot | Availability monitoring | External |

### Key Metrics Tracked

**System Metrics:**
- CPU, Memory, Disk, Network per pod
- GPU utilization, memory, temperature
- Request latency (p50, p95, p99)
- Error rates (4xx, 5xx)
- Queue depth, task completion rate

**Business Metrics:**
- Images/videos/audio generated per hour
- Average generation time per pipeline
- Cache hit rate
- Cost per generation (GPU hours)
- User sign-ups, active users

### Log Levels and Structure

```json
{
  "timestamp": "2025-12-21T10:30:45.123Z",
  "level": "INFO",
  "service": "text-to-image-worker",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "123e4567-e89b-12d3-a456-426614174000",
  "message": "Image generation started",
  "metadata": {
    "prompt": "A cat in a garden",
    "model": "sd-xl-1.0",
    "steps": 30,
    "gpu_id": 0
  }
}
```

---

## 7. SECURITY & COMPLIANCE

### Security Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **WAF** | CloudFlare / AWS WAF | Web application firewall |
| **DDoS Protection** | CloudFlare | DDoS mitigation |
| **Secrets Management** | HashiCorp Vault | API keys, credentials |
| **Certificate Management** | cert-manager | Auto SSL/TLS renewal |
| **Network Policy** | Calico / Cilium | K8s network security |
| **RBAC** | Kubernetes RBAC | Access control |
| **Vulnerability Scanning** | Trivy, Grype | Container scanning |

### Authentication & Authorization

| Component | Implementation | Details |
|-----------|---------------|---------|
| **Auth Method** | JWT + Refresh Tokens | Access: 15m, Refresh: 7d |
| **Password Hashing** | bcrypt | Cost factor: 12 |
| **OAuth 2.0** | Google, GitHub | Social login |
| **API Keys** | SHA-256 hashed | For programmatic access |
| **MFA** | TOTP (Google Authenticator) | Optional for users |
| **RBAC Roles** | admin, user, api_client | Permission-based |

---

## 8. DEVELOPMENT TOOLS

### Development Stack

| Tool | Purpose | Version |
|------|---------|---------|
| **Python** | Backend development | 3.11 |
| **Poetry** | Python package manager | 1.7.1 |
| **Node.js** | Frontend tooling | 20 LTS |
| **npm/yarn** | Package manager | Latest |
| **Docker** | Containerization | 24.0 |
| **Docker Compose** | Local orchestration | 2.23 |
| **Pre-commit** | Git hooks | Latest |
| **Black** | Python formatting | Latest |
| **Prettier** | JS/TS formatting | Latest |
| **mypy** | Python type checking | Latest |
| **ESLint** | JS/TS linting | Latest |

### Local Development Setup

```yaml
# docker-compose.yml (simplified)
services:
  api:
    image: txt2create-api:dev
    ports: ["8000:8000"]
    environment:
      - DATABASE_URL=postgresql://localhost/txt2create
      - REDIS_URL=redis://localhost:6379

  redis:
    image: redis:7.2-alpine
    ports: ["6379:6379"]

  postgres:
    image: postgres:16-alpine
    ports: ["5432:5432"]

  celery-worker:
    image: txt2create-worker:dev
    command: celery -A app.worker worker

  # Mock GPU services for local dev
  torchserve-mock:
    image: pytorch/torchserve:latest-cpu
    ports: ["8080:8080", "8081:8081"]
```

---

## Cost Estimation (Monthly)

| Component | Quantity | Unit Cost | Total |
|-----------|----------|-----------|-------|
| **Compute (API)** | 5 nodes (n1-standard-4) | $150 | $750 |
| **Compute (Workers)** | 10 nodes (n1-standard-8) | $300 | $3,000 |
| **GPU (A100 40GB)** | 8 GPUs × 730h | $2.50/h | $14,600 |
| **GPU (A100 80GB)** | 4 GPUs × 730h | $4.00/h | $11,680 |
| **Database (PostgreSQL)** | 1 primary + 2 replicas | $500 | $500 |
| **Redis Cluster** | 6 nodes | $300 | $300 |
| **S3 Storage** | 10 TB | $230 | $230 |
| **CDN (CloudFlare)** | 100 TB bandwidth | $1,000 | $1,000 |
| **Monitoring** | - | $200 | $200 |
| **Load Balancer** | - | $100 | $100 |
| **DNS, Misc** | - | $50 | $50 |
| **TOTAL** | | | **~$32,410/month** |

**Cost per Generation (estimated):**
- Text-to-Image: $0.05 - $0.10
- Text-to-Video: $0.50 - $1.00
- Text-to-Audio: $0.10 - $0.20
- Video Captioning: $0.20 - $0.40
- Virtual Avatar: $0.80 - $1.50

---

## Summary

This tech stack provides:
- ✅ **Scalability**: Handle 1000+ concurrent users
- ✅ **Reliability**: 99.9% uptime SLA
- ✅ **Performance**: <60s for most generations
- ✅ **Cost-Effective**: Optimized GPU utilization
- ✅ **Modern**: Latest AI models and infrastructure
- ✅ **Observable**: Comprehensive monitoring
- ✅ **Secure**: Enterprise-grade security

All technologies are production-tested and widely adopted in the industry.
