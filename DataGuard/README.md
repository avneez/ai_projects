# DataGuard — Secure AI Data Gateway for Enterprises

## Overview
DataGuard is a privacy-first platform that enables companies to utilize Large Language Models (LLMs) securely without exposing sensitive internal data. It acts as an intelligent middleware layer between your enterprise data and AI models, ensuring data security, privacy compliance, and vendor-agnostic LLM deployment.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Applications                       │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DataGuard Gateway API                       │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │   Request    │  │ Rate Limiter │  │  Authentication &   │  │
│  │  Validation  │  │   & Quota    │  │   Authorization     │  │
│  └──────────────┘  └──────────────┘  └─────────────────────┘  │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data Processing Pipeline                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              1. PII Detection & Classification            │  │
│  │  • SpaCy NER (Person, Org, Location, Date)              │  │
│  │  • Regex Patterns (SSN, Credit Cards, Emails, Phone)    │  │
│  │  • LLM-based Sensitivity Classifier                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              2. Selective Redaction & Tokenization        │  │
│  │  • Policy-based redaction rules                          │  │
│  │  • Token mapping & reversible substitution               │  │
│  │  • Context preservation for LLM understanding            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              3. Encryption & Key Management               │  │
│  │  • AES-256 encryption for data at rest                   │  │
│  │  • HashiCorp Vault for secret management                 │  │
│  │  • AWS KMS for key rotation & auditing                   │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
┌─────────────────────────────┐   ┌─────────────────────────────┐
│   On-Premise RAG Storage    │   │   LLM Adapter Layer         │
│  • FAISS Vector Store       │   │  • OpenAI API               │
│  • Milvus Vector DB         │   │  • Anthropic Claude         │
│  • Private Document Store   │   │  • vLLM (Self-hosted)       │
│  • Embedding Generation     │   │  • Hugging Face Models      │
└─────────────────────────────┘   └─────────────────────────────┘
```

## Key Components

### 1. PII Detection Engine
- **SpaCy NER**: Identifies named entities (persons, organizations, locations)
- **Regex Patterns**: Detects structured sensitive data (SSN, credit cards, emails)
- **LLM Classifier**: Uses transformer models to identify contextual sensitivity

### 2. Security Layer
- **Encryption**: AES-256-GCM for data at rest and in transit
- **Key Management**: HashiCorp Vault for secrets, AWS KMS for key rotation
- **Access Control**: Role-based access control (RBAC) with policy enforcement

### 3. On-Premise RAG System
- **Vector Storage**: FAISS for fast similarity search, Milvus for production scale
- **Document Store**: Private storage ensuring no external data exposure
- **Embedding Models**: Self-hosted models for complete data isolation

### 4. LLM Adapter Layer
- Unified interface for multiple LLM providers
- Automatic retry logic and failover
- Cost tracking and usage monitoring

## Technology Stack

| Component | Technology |
|-----------|-----------|
| **Backend Framework** | FastAPI (Python) |
| **PII Detection** | SpaCy, Transformers, Regex |
| **Encryption** | cryptography (AES-256-GCM) |
| **Secret Management** | HashiCorp Vault, AWS KMS |
| **Vector Database** | FAISS, Milvus |
| **Message Queue** | Redis, Celery |
| **Database** | PostgreSQL (metadata), MongoDB (logs) |
| **Monitoring** | Prometheus, Grafana |
| **Container** | Docker, Kubernetes |
| **LLM Integration** | OpenAI SDK, Anthropic SDK, vLLM, Hugging Face |

## Core Features

1. **Automatic PII Detection & Redaction**
   - Real-time identification of 20+ PII types
   - Customizable redaction policies per organization
   - Reversible tokenization for result reconstruction

2. **End-to-End Encryption**
   - AES-256-GCM encryption for all sensitive data
   - Encrypted storage for embeddings and documents
   - TLS 1.3 for data in transit

3. **On-Premise RAG**
   - Complete data isolation with local vector storage
   - No external API calls for embedding generation
   - Self-hosted retrieval without cloud dependencies

4. **Vendor-Agnostic LLM Support**
   - Single API interface for multiple LLM providers
   - Easy switching between providers without code changes
   - Support for both cloud and self-hosted models

5. **Compliance & Auditing**
   - Detailed audit logs for all data access
   - GDPR, HIPAA, SOC2 compliance support
   - Data lineage tracking

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/yourcompany/dataguard.git
cd dataguard

# Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Initialize database
alembic upgrade head

# Start HashiCorp Vault (development mode)
vault server -dev

# Start the application
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Configuration

Create a `config.yaml` file:

```yaml
security:
  encryption_algorithm: "AES-256-GCM"
  key_rotation_days: 90
  vault_address: "http://localhost:8200"

pii_detection:
  enabled: true
  confidence_threshold: 0.85
  entity_types:
    - PERSON
    - ORG
    - EMAIL
    - PHONE
    - SSN
    - CREDIT_CARD

redaction:
  strategy: "tokenize"  # Options: mask, tokenize, hash, remove
  preserve_context: true

rag:
  vector_db: "faiss"  # Options: faiss, milvus
  embedding_model: "sentence-transformers/all-mpnet-base-v2"
  chunk_size: 512
  chunk_overlap: 50

llm_providers:
  default: "openai"
  fallback: ["anthropic", "vllm"]
  timeout: 30
```

## Usage Examples

### Basic Text Processing with PII Redaction

```python
from dataguard import DataGuardClient

client = DataGuardClient(api_key="your-api-key")

# Process text with automatic PII detection
response = client.process(
    text="John Doe's SSN is 123-45-6789 and email is john@example.com",
    redact_pii=True
)

print(response.redacted_text)
# Output: "[PERSON]'s SSN is [SSN] and email is [EMAIL]"
print(response.entities)
# Output: [{'type': 'PERSON', 'value': 'John Doe', 'start': 0, 'end': 8}, ...]
```

### Secure LLM Query

```python
# Query LLM with automatic PII protection
result = client.query_llm(
    prompt="Analyze this customer feedback: John Smith called about order #12345",
    provider="openai",
    model="gpt-4",
    redact_pii=True,
    restore_pii=True  # Restore PII in response
)

print(result.response)
# PII redacted before sending to LLM, restored in response
```

### On-Premise RAG Query

```python
# Index documents in private vector store
client.rag.index_documents(
    documents=["doc1.pdf", "doc2.txt"],
    namespace="company_policies"
)

# Query without external API calls
answer = client.rag.query(
    question="What is the vacation policy?",
    namespace="company_policies",
    top_k=3
)

print(answer.response)
print(answer.sources)  # Retrieved document chunks
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/process` | POST | Process text with PII detection |
| `/api/v1/llm/query` | POST | Query LLM with security layer |
| `/api/v1/rag/index` | POST | Index documents in RAG |
| `/api/v1/rag/query` | POST | Query RAG system |
| `/api/v1/policies` | GET/POST | Manage security policies |
| `/api/v1/audit` | GET | Retrieve audit logs |
| `/health` | GET | Health check |

## Security Considerations

1. **Data Isolation**: All sensitive data remains within your infrastructure
2. **Zero Trust**: Every request is authenticated and authorized
3. **Encryption at Rest**: All stored data is encrypted using AES-256
4. **Key Rotation**: Automatic key rotation every 90 days
5. **Audit Logging**: Comprehensive logs for compliance and forensics
6. **Rate Limiting**: Prevent abuse and ensure fair usage

## Performance Metrics

- PII Detection Latency: <50ms for 1KB text
- Encryption Overhead: <10ms per request
- RAG Query Latency: <200ms for 1M documents (FAISS)
- Throughput: 1000+ requests/second (horizontal scaling)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache 2.0 License - see [LICENSE](LICENSE) for details.

## Support

- Documentation: https://docs.dataguard.io
- Issues: https://github.com/yourcompany/dataguard/issues
- Email: support@dataguard.io
