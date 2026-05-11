# DataGuard - How It Actually Works (Detailed Explanation)

## Table of Contents
1. [The Core Problem](#the-core-problem)
2. [Complete Data Flow](#complete-data-flow)
3. [Why We Need Vault (Despite Redaction)](#why-we-need-vault)
4. [How We Preserve Context While Redacting](#preserving-context)
5. [Why Vector Database is Critical](#why-vector-database)
6. [End-to-End Example](#end-to-end-example)

---

## The Core Problem

### Scenario
Your company wants to use ChatGPT/Claude to analyze customer support tickets, but tickets contain:
```
Customer: John Smith
Email: john@example.com
SSN: 123-45-6789
Issue: "My credit card ending in 4532 was charged twice"
```

### The Dilemma
- ❌ **Send as-is to OpenAI**: Violates GDPR, exposes customer PII to third-party
- ❌ **Don't use AI**: Miss out on automation, insights, efficiency
- ✅ **DataGuard Solution**: Use AI safely without exposing sensitive data

---

## Complete Data Flow

### Phase 1: Incoming Request (User → DataGuard)

```
┌─────────────────────────────────────────────────────────────┐
│  Your Application sends request to DataGuard                │
│  ----------------------------------------------------------- │
│  POST /api/v1/llm/query                                     │
│  {                                                           │
│    "prompt": "Analyze this ticket:                          │
│                John Smith (john@example.com)                │
│                SSN: 123-45-6789                             │
│                Credit card ending 4532 charged twice"       │
│  }                                                           │
└─────────────────────────────────────────────────────────────┘
                          ↓
```

### Phase 2: PII Detection & Tokenization

```
┌─────────────────────────────────────────────────────────────┐
│  DataGuard detects PII using 3 methods:                     │
│  ----------------------------------------------------------- │
│  1. SpaCy NER:       "John Smith" → PERSON                  │
│  2. Regex:           "123-45-6789" → SSN                    │
│  3. Regex:           "john@example.com" → EMAIL             │
│  4. Regex:           "4532" → CREDIT_CARD (partial)         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Tokenization (REVERSIBLE replacement)                      │
│  ----------------------------------------------------------- │
│  Original → Token → Stored in Memory (encrypted)            │
│  "John Smith" → [PERSON_A1B2C3D4] → Vault/Redis             │
│  "john@example.com" → [EMAIL_E5F6G7H8] → Vault/Redis        │
│  "123-45-6789" → [SSN_I9J0K1L2] → Vault/Redis               │
│  "4532" → [CC_M3N4O5P6] → Vault/Redis                       │
│                                                              │
│  Result (sent to LLM):                                      │
│  "Analyze this ticket:                                      │
│   [PERSON_A1B2C3D4] ([EMAIL_E5F6G7H8])                      │
│   SSN: [SSN_I9J0K1L2]                                       │
│   Credit card ending [CC_M3N4O5P6] charged twice"          │
└─────────────────────────────────────────────────────────────┘
                          ↓
```

### Phase 3: Send to LLM (OpenAI/Claude/etc)

```
┌─────────────────────────────────────────────────────────────┐
│  DataGuard → OpenAI API                                     │
│  ----------------------------------------------------------- │
│  NO REAL PII IS SENT!                                       │
│  OpenAI only sees:                                          │
│  "[PERSON_A1B2C3D4] has issue with [CC_M3N4O5P6]..."       │
│                                                              │
│  LLM Response:                                              │
│  "The customer [PERSON_A1B2C3D4] experienced a double       │
│   charge on card [CC_M3N4O5P6]. Recommend refund."         │
└─────────────────────────────────────────────────────────────┘
                          ↓
```

### Phase 4: De-tokenization (Restore PII)

```
┌─────────────────────────────────────────────────────────────┐
│  DataGuard reverses the tokens before sending to user      │
│  ----------------------------------------------------------- │
│  [PERSON_A1B2C3D4] → "John Smith" (from Vault/Redis)       │
│  [CC_M3N4O5P6] → "4532" (from Vault/Redis)                 │
│                                                              │
│  Final Response to User:                                    │
│  "The customer John Smith experienced a double              │
│   charge on card 4532. Recommend refund."                  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Your Application receives clean response with PII intact   │
└─────────────────────────────────────────────────────────────┘
```

---

## Why We Need Vault (Despite Redaction)

### You might ask: "If we're removing PII, why encrypt anything?"

#### Reason 1: Token Mapping Storage
```python
# Without Vault (INSECURE):
token_map = {
    "[PERSON_A1B2C3D4]": "John Smith",          # ❌ PII in plain text in memory
    "[SSN_I9J0K1L2]": "123-45-6789",           # ❌ Anyone with memory access sees it
    "[EMAIL_E5F6G7H8]": "john@example.com"     # ❌ Vulnerable to memory dumps
}

# With Vault (SECURE):
vault.set_secret("tokens/session_xyz/person", {
    "token": "[PERSON_A1B2C3D4]",
    "value": encrypt("John Smith")              # ✅ Encrypted even in Vault
})
# ✅ Automatic TTL (expires after use)
# ✅ Access logs (who accessed what PII)
# ✅ Can't dump memory to get secrets
```

#### Reason 2: Database Storage (Long-term Data)
```python
# Scenario: You store customer data for analytics
customer_record = {
    "id": 12345,
    "name": "John Smith",              # ❌ PII in database
    "email": "john@example.com",       # ❌ Vulnerable to SQL injection, breaches
    "analysis_result": "..."
}

# With Vault + Field Encryption:
customer_record = {
    "id": 12345,
    "name": "v1:abc123def...",         # ✅ Encrypted with Vault Transit Engine
    "email": "v1:xyz789ghi...",        # ✅ Can only decrypt with Vault key
    "analysis_result": "..."            # ✅ Non-PII remains searchable
}

# Even if attacker steals database:
# - They see encrypted gibberish
# - Need Vault access token to decrypt
# - Vault logs all access attempts
```

#### Reason 3: Dynamic Credentials
```python
# Traditional (INSECURE):
DATABASE_PASSWORD = "hardcoded_password_123"   # ❌ In .env file, git history

# With Vault (SECURE):
creds = vault.get_database_credentials(role="dataguard_app")
# Returns: {
#   "username": "v-root-app-abc123",    # ✅ Auto-generated
#   "password": "random-xyz789",         # ✅ Unique per session
#   "lease_duration": 3600               # ✅ Expires in 1 hour
# }
# ✅ No hardcoded secrets
# ✅ Automatic rotation
# ✅ Revoked when lease expires
```

#### Reason 4: Encryption Key Management
```python
# Problem: Where do you store the AES encryption key?
ENCRYPTION_KEY = "my-secret-key-12345"   # ❌ Defeats purpose of encryption

# With Vault Transit Engine:
# 1. Key NEVER leaves Vault
# 2. You send data to Vault → Vault encrypts → Returns ciphertext
ciphertext = vault.encrypt_with_transit(
    key_name="customer-data-key",
    plaintext="John Smith"
)
# Result: "vault:v1:abc123..."
# ✅ Key rotation without re-encrypting everything
# ✅ Centralized audit of all encryption operations
```

### Summary: Why Vault?
| Without Vault | With Vault |
|---------------|------------|
| PII in memory (dumpable) | Encrypted in Vault (isolated) |
| Hardcoded DB passwords | Dynamic credentials (auto-rotate) |
| Manual key rotation | Automatic key rotation |
| No audit trail | Complete access logs |
| Single point of failure | High availability, disaster recovery |

---

## Preserving Context While Redacting

### The Context Problem

```python
# Naive Redaction (LOSES CONTEXT):
Original: "John called about his account, John is very upset"
Redacted: "[REDACTED] called about his account, [REDACTED] is very upset"
#         ❌ LLM doesn't know both refer to same person!

# Smart Tokenization (PRESERVES CONTEXT):
Original: "John called about his account, John is very upset"
Redacted: "[PERSON_A1B2C3D4] called about his account, [PERSON_A1B2C3D4] is very upset"
#         ✅ Same token = Same person (LLM understands continuity)
```

### Techniques We Use

#### 1. Consistent Tokenization
```python
class PIIRedactor:
    def __init__(self):
        self.entity_cache = {}  # Remember entities within session

    def redact(self, text, entities):
        for entity in entities:
            # Check if we've seen this exact PII before
            cache_key = f"{entity.label}:{entity.text}"

            if cache_key in self.entity_cache:
                # Reuse same token for consistency
                token = self.entity_cache[cache_key]
            else:
                # Generate new token
                token = f"[{entity.label}_{uuid.uuid4().hex[:8]}]"
                self.entity_cache[cache_key] = token

            text = text.replace(entity.text, token)

        return text

# Example:
texts = [
    "John Smith called at 9am",
    "John Smith called again at 2pm"
]

# Both use [PERSON_A1B2C3D4]:
# "[PERSON_A1B2C3D4] called at 9am"
# "[PERSON_A1B2C3D4] called again at 2pm"
# ✅ LLM knows it's the same person across conversations
```

#### 2. Semantic Preservation
```python
# BAD: Remove all context
"Customer *** has issue with card ***"
# ❌ LLM has no idea what's being discussed

# GOOD: Keep entity types
"Customer [PERSON] has issue with card [CREDIT_CARD]"
# ✅ LLM knows it's about a person and credit card

# BETTER: Keep semantic hints
"Customer [PERSON_A1B2C3D4] has issue with card [CREDIT_CARD_ending_M3N4]"
# ✅ LLM knows exact entities + card hint preserved
```

#### 3. Partial Redaction for Context
```python
def smart_redact_credit_card(card_number):
    # Instead of: "[CREDIT_CARD]"
    # Do: "XXXX-XXXX-XXXX-4532"
    return f"XXXX-XXXX-XXXX-{card_number[-4:]}"

def smart_redact_email(email):
    # Instead of: "[EMAIL]"
    # Do: "j***@example.com"
    username, domain = email.split('@')
    return f"{username[0]}***@{domain}"

# LLM can still infer:
# - Card type (Visa starts with 4)
# - Email domain (work vs personal)
# While actual PII is protected
```

#### 4. Metadata Enrichment
```python
# Add context WITHOUT exposing PII
redacted_with_context = {
    "text": "[PERSON_A1B2C3D4] purchased item for [PRICE_50]",
    "metadata": {
        "[PERSON_A1B2C3D4]": {
            "type": "PERSON",
            "attributes": {
                "customer_segment": "premium",     # ✅ Safe to share
                "tenure_years": 5,                 # ✅ Aggregated data
                "sentiment": "frustrated"          # ✅ Derived, not PII
            }
        }
    }
}

# LLM gets richer context without seeing "John Smith"
```

---

## Why Vector Database is Critical

### Use Case 1: Enterprise Knowledge Base (On-Premise RAG)

#### The Problem
```
Your company has:
- 10,000 internal documents (HR policies, product specs, customer data)
- Employees want to ask: "What's our parental leave policy?"
- Documents contain PII: "John Smith took leave in 2023..."

Risk: If you send documents to OpenAI for search, you leak:
- Employee names, salaries, performance reviews
- Customer contracts, pricing
- Proprietary information
```

#### The Solution: On-Premise RAG with Vector DB

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: Index Documents (ONE TIME)                     │
└─────────────────────────────────────────────────────────┘

Document: "Parental leave policy: Employees get 16 weeks.
           John Smith used this in 2023."

         ↓ [Split into chunks]

Chunk 1: "Parental leave policy: Employees get 16 weeks."
Chunk 2: "John Smith used this in 2023."

         ↓ [Detect PII in Chunk 2]

Chunk 2: "[PERSON_A1B2C3D4] used this in 2023."

         ↓ [Encrypt with Vault]

Chunk 2: "vault:v1:encrypted_blob_xyz..."

         ↓ [Generate embeddings LOCALLY]

embedding = SentenceTransformer("all-mpnet-base-v2").encode(chunk)
# [0.23, -0.45, 0.67, ...] (768-dimensional vector)

         ↓ [Store in FAISS (Local Vector DB)]

FAISS Index (ON YOUR SERVERS):
- Chunk 1 vector: [0.12, 0.34, ...]
- Chunk 2 vector: [0.23, -0.45, ...] (encrypted text)

┌─────────────────────────────────────────────────────────┐
│ Step 2: Query (EVERY TIME)                             │
└─────────────────────────────────────────────────────────┘

User asks: "What's the parental leave policy?"

         ↓ [Convert query to vector LOCALLY]

query_vector = encode("What's the parental leave policy?")
# [0.15, 0.30, ...]

         ↓ [Search FAISS LOCALLY - NO API CALLS]

FAISS finds most similar: Chunk 1 (95% similarity)

         ↓ [Retrieve chunk]

Retrieved: "Parental leave policy: Employees get 16 weeks."

         ↓ [Send to LLM with context]

Prompt: "Based on: 'Parental leave policy: Employees get 16 weeks.'
         Answer: What's the parental leave policy?"

         ↓ [LLM responds]

Answer: "Employees receive 16 weeks of parental leave."
```

### Why Vector DB (FAISS/Milvus) Instead of Traditional Search?

```python
# Traditional Keyword Search (FAILS):
query = "maternity leave"
# Won't match: "parental leave" (different words)
# Won't match: "time off for new parents" (no keywords)

# Vector Search (SUCCEEDS):
query_vector = embed("maternity leave")
# Finds: "parental leave" (semantically similar)
# Finds: "time off for new parents" (conceptually related)
# Finds: "paternity policy" (related concept)

# Because embeddings capture MEANING, not just words
```

### Use Case 2: Semantic PII Detection

```python
# Traditional regex MISSES context-based PII:
text = "My son attends Lincoln Elementary in Boston"
# ❌ Regex sees: "Lincoln Elementary" (not flagged)
# ✅ But combined with "my son" → reveals child's school (PII!)

# Vector-based detection:
# 1. Embed sentence
embedding = embed("My son attends Lincoln Elementary")

# 2. Compare to known PII patterns (stored in vector DB)
similar_patterns = vector_db.search(embedding)
# Finds: "My daughter goes to X school" (labeled as PII)
#        "Child's school: Y" (labeled as PII)

# 3. Classify as sensitive
# ✅ Flags entire sentence as potential child privacy issue
```

### Use Case 3: Efficient Multi-Document Analysis

```
Without Vector DB:
User: "Summarize all customer complaints about billing"

DataGuard:
1. Load ALL 50,000 documents
2. Send to LLM (costs $$$, slow, context limit)

With Vector DB:
User: "Summarize all customer complaints about billing"

DataGuard:
1. Search vector DB for "billing complaints" → 50 relevant docs
2. Send only top 10 to LLM (fast, cheap, focused)
```

---

## End-to-End Example

### Scenario: Customer Support Ticket Analysis

```python
# ==========================================
# 1. APPLICATION SENDS REQUEST
# ==========================================

import requests

ticket = """
Customer: Sarah Johnson (sarah.j@email.com)
Phone: +1-555-0123
SSN: 987-65-4321
Issue: My credit card ending in 8765 was charged $299
       but I never received the product. Order #AB-12345.
       I'm very frustrated!
"""

response = requests.post(
    "http://dataguard.company.com/api/v1/llm/query",
    json={
        "prompt": f"Analyze this support ticket and suggest resolution:\n{ticket}",
        "provider": "openai",
        "redact_pii": True,
        "restore_pii": True
    },
    headers={"Authorization": "Bearer your-api-key"}
)

# ==========================================
# 2. DATAGUARD PROCESSES (INTERNAL)
# ==========================================

# Step A: PII Detection
detected_entities = [
    PIIEntity(text="Sarah Johnson", label="PERSON", start=10, end=23),
    PIIEntity(text="sarah.j@email.com", label="EMAIL", start=25, end=42),
    PIIEntity(text="+1-555-0123", label="PHONE", start=50, end=61),
    PIIEntity(text="987-65-4321", label="SSN", start=67, end=78),
    PIIEntity(text="8765", label="CREDIT_CARD", start=113, end=117),
]

# Step B: Tokenization + Vault Storage
tokens = {
    "[PERSON_F8A2E1C9]": encrypt_and_store_vault("Sarah Johnson"),
    "[EMAIL_B3D5E7F1]": encrypt_and_store_vault("sarah.j@email.com"),
    "[PHONE_A1B2C3D4]": encrypt_and_store_vault("+1-555-0123"),
    "[SSN_E5F6G7H8]": encrypt_and_store_vault("987-65-4321"),
    "[CC_I9J0K1L2]": encrypt_and_store_vault("8765"),
}

redacted_ticket = """
Customer: [PERSON_F8A2E1C9] ([EMAIL_B3D5E7F1])
Phone: [PHONE_A1B2C3D4]
SSN: [SSN_E5F6G7H8]
Issue: My credit card ending in [CC_I9J0K1L2] was charged $299
       but I never received the product. Order #AB-12345.
       I'm very frustrated!
"""

# Step C: Send to OpenAI
openai_response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{
        "role": "user",
        "content": f"Analyze this support ticket:\n{redacted_ticket}"
    }]
)

llm_response = """
Analysis:
- Customer [PERSON_F8A2E1C9] has a billing issue
- Charged $299 on card [CC_I9J0K1L2] but product not delivered
- Order number: AB-12345
- Sentiment: Frustrated

Suggested Resolution:
1. Verify order #AB-12345 in system
2. Issue immediate refund to card [CC_I9J0K1L2]
3. Expedite product shipping or offer full refund
4. Follow up with [PERSON_F8A2E1C9] via [EMAIL_B3D5E7F1]
5. Offer 20% discount on next purchase for inconvenience
"""

# Step D: De-tokenization
final_response = detokenize(llm_response, tokens)
# Retrieves encrypted values from Vault, decrypts, replaces tokens

final_response = """
Analysis:
- Customer Sarah Johnson has a billing issue
- Charged $299 on card 8765 but product not delivered
- Order number: AB-12345
- Sentiment: Frustrated

Suggested Resolution:
1. Verify order #AB-12345 in system
2. Issue immediate refund to card 8765
3. Expedite product shipping or offer full refund
4. Follow up with Sarah Johnson via sarah.j@email.com
5. Offer 20% discount on next purchase for inconvenience
"""

# Step E: Cleanup
vault.delete_tokens(session_id="xyz")  # Remove temporary PII mapping

# ==========================================
# 3. APPLICATION RECEIVES RESPONSE
# ==========================================

print(response.json())
# {
#   "response": "Analysis: Customer Sarah Johnson...",
#   "pii_protected": true,
#   "entities_redacted": [
#     {"type": "PERSON", "text": "Sarah Johnson"},
#     {"type": "EMAIL", "text": "sarah.j@email.com"},
#     ...
#   ],
#   "model_used": "gpt-4",
#   "tokens_used": 450,
#   "latency_ms": 1250
# }

# ✅ Application got intelligent AI response
# ✅ OpenAI NEVER saw real customer PII
# ✅ Response contains original PII for agent to act on
# ✅ Full audit trail of PII access in Vault logs
```

---

## Architecture Visualization

```
┌──────────────────────────────────────────────────────────────┐
│                     YOUR APPLICATION                          │
│  (Support Agent Dashboard, Analytics Tool, Chatbot, etc.)    │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         │ Request with PII
                         ↓
┌──────────────────────────────────────────────────────────────┐
│                      DATAGUARD GATEWAY                        │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ 1. PII DETECTION                                         │ │
│ │    SpaCy + Regex + LLM Classifier                        │ │
│ │    → Identifies: Names, SSN, Emails, Cards, etc.         │ │
│ └──────────────────────────────────────────────────────────┘ │
│                         ↓                                     │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ 2. POLICY EVALUATION                                     │ │
│ │    Should we: ALLOW / DENY / REDACT / ENCRYPT?           │ │
│ └──────────────────────────────────────────────────────────┘ │
│                         ↓                                     │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ 3. TOKENIZATION                                          │ │
│ │    "John" → [PERSON_A1B2]                                │ │
│ │    Store mapping in Vault (encrypted)                    │ │
│ └──────────────────────────────────────────────────────────┘ │
│                         ↓                                     │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ 4. RAG SEARCH (if needed)                                │ │
│ │    Query local vector DB for context                     │ │
│ │    → Returns relevant docs (encrypted PII)               │ │
│ └──────────────────────────────────────────────────────────┘ │
│                         ↓                                     │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ 5. LLM ADAPTER                                           │ │
│ │    Route to: OpenAI / Anthropic / vLLM                   │ │
│ │    (Sends redacted text only)                            │ │
│ └──────────────────────────────────────────────────────────┘ │
│                         ↓                                     │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ 6. DE-TOKENIZATION                                       │ │
│ │    [PERSON_A1B2] → "John" (from Vault)                   │ │
│ │    Restore original PII in response                      │ │
│ └──────────────────────────────────────────────────────────┘ │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         │ Response with PII restored
                         ↓
┌──────────────────────────────────────────────────────────────┐
│                     YOUR APPLICATION                          │
│              (Gets actionable response with PII)             │
└──────────────────────────────────────────────────────────────┘

SUPPORTING COMPONENTS (Run alongside DataGuard):

┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│  HASHICORP VAULT    │  │ FAISS/MILVUS (VDB)  │  │    LLM PROVIDERS    │
│  ─────────────────  │  │  ─────────────────  │  │  ─────────────────  │
│  • Token mappings   │  │  • Doc embeddings   │  │  • OpenAI API       │
│  • Encryption keys  │  │  • Semantic search  │  │  • Anthropic API    │
│  • DB credentials   │  │  • On-premise only  │  │  • vLLM (self-host) │
│  • Access logs      │  │  • Encrypted docs   │  │  • Hugging Face     │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘
```

---

## Key Takeaways

### 1. **Vault is NOT redundant**
- Stores token→PII mappings securely (temporary, encrypted, logged)
- Manages encryption keys (never hardcoded)
- Provides dynamic credentials (auto-rotating DB passwords)
- Encrypts long-term storage (customer records in database)

### 2. **Context is preserved through**
- Consistent tokenization (same entity = same token)
- Semantic hints (entity types in tokens)
- Partial redaction (last 4 digits of card)
- Metadata enrichment (aggregate info without PII)

### 3. **Vector DB enables**
- On-premise semantic search (no data sent to external vector DBs)
- Efficient document retrieval (find relevant docs from millions)
- Context-aware PII detection (embeddings capture semantic meaning)
- RAG without cloud dependencies (full data isolation)

### 4. **The complete flow ensures**
- ✅ LLMs never see real PII (tokenized before sending)
- ✅ Users get actionable responses (PII restored after LLM)
- ✅ Complete audit trail (Vault logs all PII access)
- ✅ Compliance with GDPR/HIPAA (data never leaves infrastructure)
- ✅ Cost optimization (vector search reduces LLM context size)

---

## Common Misconceptions Clarified

| Misconception | Reality |
|---------------|---------|
| "If we redact PII, why encrypt?" | Token mappings contain PII and must be protected |
| "Can't we just use Redis for tokens?" | Redis is vulnerable to memory dumps; Vault adds encryption + audit + TTL |
| "Vector DB is just for search" | Also used for: semantic PII detection, embedding storage, context retrieval |
| "We can use Pinecone/Weaviate" | Those are cloud services (data leaves your servers); FAISS/Milvus are on-prem |
| "Why not just remove PII entirely?" | Users need original PII to take action (refund card, email customer, etc.) |
| "This seems overcomplicated" | Enterprise security requires defense-in-depth; each layer protects against different threats |

---

## When to Use Each Component

| Component | Use When | Skip When |
|-----------|----------|-----------|
| **PII Detection** | Always (for any external LLM usage) | Internal analysis on anonymized data only |
| **Vault** | Storing any secrets, PII mappings, encryption keys | Demo/prototype (use encrypted Redis temporarily) |
| **Vector DB** | Document search, large knowledge bases, semantic analysis | Simple Q&A with small static FAQs |
| **RAG** | Need to search internal docs without cloud APIs | All context fits in LLM prompt (<100k tokens) |
| **Encryption** | Data at rest, regulatory compliance | Fully air-gapped system with physical security |

