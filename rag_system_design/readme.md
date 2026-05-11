# RAG System Design: Private Corpus Q&A
### Built with LangChain + LangGraph

A production-grade Retrieval-Augmented Generation system that answers natural language questions over a private, frequently-updated document corpus (e.g., internal logs, PDFs, reports).

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Dependencies](#3-dependencies)
4. [Document Ingestion Pipeline](#4-document-ingestion-pipeline)
5. [Chunking Strategy](#5-chunking-strategy)
6. [Embedding + Vector Store](#6-embedding--vector-store)
7. [What Gets Stored in the Database](#7-what-gets-stored-in-the-database)
8. [Database Selection](#8-database-selection)
9. [LangGraph Query Pipeline](#9-langgraph-query-pipeline)
10. [Each LangGraph Node Explained](#10-each-langgraph-node-explained)
11. [LLM Generation with Grounding](#11-llm-generation-with-grounding)
12. [Handling Frequent Document Updates](#12-handling-frequent-document-updates)
13. [Reducing Hallucination](#13-reducing-hallucination)
14. [Low Latency Design](#14-low-latency-design)
15. [Scalability](#15-scalability)
16. [Full End-to-End Example](#16-full-end-to-end-example)
17. [Hybrid RAG: Improving Retrieval Beyond Dense Vectors](#17-hybrid-rag-improving-retrieval-beyond-dense-vectors)
18. [Evaluating the RAG Pipeline](#18-evaluating-the-rag-pipeline)

---

## 1. System Overview

```
User Question → LangGraph Pipeline → Retrieval → Grounding → LLM → Answer + Sources
```

| Flow | Trigger | Tools Used |
|------|---------|-----------|
| **Ingestion** | Document upload / scheduled crawl | LangChain Loaders → Splitters → Embeddings → Qdrant |
| **Query** | User asks a question | LangGraph StateGraph → Retriever → Re-ranker → LLM Chain |

---

## 2. Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                       INGESTION PIPELINE                         │
│                                                                  │
│  [PDF / TXT / LOG]                                               │
│       │                                                          │
│       ▼                                                          │
│  LangChain Document Loaders                                      │
│  (PyPDFLoader, TextLoader, custom LogLoader)                     │
│       │                                                          │
│       ▼                                                          │
│  RecursiveCharacterTextSplitter  (chunk_size=512, overlap=64)    │
│       │                                                          │
│       ▼                                                          │
│  OpenAIEmbeddings / HuggingFaceEmbeddings                        │
│       │                                                          │
│       ▼                                                          │
│  QdrantVectorStore  ◄──────────────────── PostgreSQL (metadata)  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                    LANGGRAPH QUERY PIPELINE                      │
│                                                                  │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│   │  embed_  │───►│ retrieve │───►│ rerank   │───►│ generate │  │
│   │  query   │    │ (Qdrant) │    │(optional)│    │  (LLM)   │  │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                                        │                         │
│                              score < threshold?                  │
│                                        │                         │
│                                        ▼                         │
│                              ┌──────────────────┐               │
│                              │ no_answer_node   │               │
│                              └──────────────────┘               │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Dependencies

```bash
pip install \
  langchain \
  langchain-openai \
  langchain-community \
  langchain-qdrant \
  langgraph \
  qdrant-client \
  pdfplumber \
  sentence-transformers \
  psycopg2-binary \
  redis \
  python-dotenv
```

```python
# .env
OPENAI_API_KEY=sk-...
QDRANT_HOST=localhost
QDRANT_PORT=6333
POSTGRES_DSN=postgresql://user:pass@localhost:5432/ragdb
```

---

## 4. Document Ingestion Pipeline

LangChain provides built-in **Document Loaders** that parse files and return `Document` objects — each containing `page_content` (text) and `metadata` (source, page, etc.).

### 4.1 PDF Loader

**Input:** `network_access_report_2024_Q1.pdf`

```
Page 1:
  "Q1 2024 Internet Access Report
   Prepared by: IT Security Team

   Summary: A total of 4.2 million HTTP requests were logged between
   January and March 2024. Of these, 12,400 were flagged as suspicious..."

Page 2:
  "Top Blocked Domains:
   1. malicious-ads.xyz    — 8,200 blocks
   2. tracking.badactor.io — 3,100 blocks"
```

```python
from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path: str) -> list:
    loader = PyPDFLoader(file_path)
    documents = loader.load()  # returns one Document per page
    return documents

# Each Document looks like:
# Document(
#   page_content="Q1 2024 Internet Access Report\nPrepared by: IT Security Team\n...",
#   metadata={
#     "source": "network_access_report_2024_Q1.pdf",
#     "page": 0   ← 0-indexed
#   }
# )
```

### 4.2 Plain Text Loader

```python
from langchain_community.document_loaders import TextLoader

def load_text(file_path: str) -> list:
    loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()
```

### 4.3 Custom Internet Log Loader

LangChain allows custom loaders by subclassing `BaseLoader`.

**Input:** `access.log`

```
2024-01-15 08:32:11 user=john.doe src=192.168.1.45 dst=github.com action=ALLOW bytes=4096
2024-01-15 08:32:45 user=jane.smith src=192.168.1.67 dst=malicious-ads.xyz action=BLOCK bytes=0
2024-01-15 08:33:01 user=john.doe src=192.168.1.45 dst=slack.com action=ALLOW bytes=8192
```

```python
import re
from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document

LOG_PATTERN = re.compile(
    r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) "
    r"user=(?P<user>\S+) src=(?P<src>\S+) dst=(?P<dst>\S+) "
    r"action=(?P<action>\S+) bytes=(?P<bytes>\d+)"
)

class InternetLogLoader(BaseLoader):
    def __init__(self, file_path: str, batch_size: int = 100):
        self.file_path = file_path
        self.batch_size = batch_size

    def load(self) -> list[Document]:
        docs = []
        batch = []

        with open(self.file_path) as f:
            for line in f:
                m = LOG_PATTERN.match(line.strip())
                if m:
                    entry = m.groupdict()
                    batch.append(entry)

                    if len(batch) == self.batch_size:
                        docs.append(self._batch_to_document(batch))
                        batch = []

        if batch:
            docs.append(self._batch_to_document(batch))

        return docs

    def _batch_to_document(self, batch: list[dict]) -> Document:
        text = "\n".join(
            f"{e['timestamp']} - {e['user']} accessed {e['dst']} [{e['action']}]"
            for e in batch
        )
        return Document(
            page_content=text,
            metadata={
                "source": self.file_path,
                "start_time": batch[0]["timestamp"],
                "end_time": batch[-1]["timestamp"],
                "file_type": "log",
            }
        )

# Usage
loader = InternetLogLoader("access.log", batch_size=100)
documents = loader.load()

# Each Document:
# Document(
#   page_content="2024-01-15 08:32:11 - john.doe accessed github.com [ALLOW]\n
#                 2024-01-15 08:32:45 - jane.smith accessed malicious-ads.xyz [BLOCK]\n...",
#   metadata={"source": "access.log", "start_time": "2024-01-15 08:32:11", ...}
# )
```

### 4.4 Unified Loader Router

```python
from pathlib import Path

def load_document(file_path: str) -> list[Document]:
    ext = Path(file_path).suffix.lower()

    loaders = {
        ".pdf":  PyPDFLoader(file_path),
        ".txt":  TextLoader(file_path),
        ".md":   TextLoader(file_path),
        ".log":  InternetLogLoader(file_path),
    }

    loader = loaders.get(ext)
    if not loader:
        raise ValueError(f"Unsupported file type: {ext}")

    return loader.load()
```

---

## 5. Chunking Strategy

LangChain's `RecursiveCharacterTextSplitter` splits text hierarchically — tries `\n\n` first, then `\n`, then `.`, then ` ` — until chunks fit the size limit.

### 5.1 Why Chunk?

- Embedding models have a token limit (~512 tokens)
- A full PDF page is too noisy for precise retrieval
- Each chunk = one focused topic = one meaningful vector

### 5.2 The Splitter

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,          # 64-char overlap preserves cross-boundary context
    separators=["\n\n", "\n", ". ", " ", ""],
    add_start_index=True,      # adds char offset to metadata
)

# Input: list[Document] from loader
documents = load_document("network_access_report_2024_Q1.pdf")
chunks = splitter.split_documents(documents)

# What split_documents does:
# 1. Takes each Document's page_content
# 2. Splits it into smaller pieces
# 3. Inherits original metadata + adds "start_index"

# chunk[0]:
# Document(
#   page_content="Q1 2024 Internet Access Report\nPrepared by: IT Security Team\n\n
#                 Summary: A total of 4.2 million HTTP requests were logged between
#                 January and March 2024. Of these, 12,400 were flagged as suspicious
#                 based on threat intelligence feeds.",
#   metadata={"source": "network_access_report_2024_Q1.pdf", "page": 0, "start_index": 0}
# )

# chunk[1]:
# Document(
#   page_content="The top blocked category was malvertising, accounting for 68%
#                 of all blocked traffic.\n\nTop Blocked Domains:\n
#                 1. malicious-ads.xyz — 8,200 blocks\n2. tracking.badactor.io — 3,100 blocks",
#   metadata={"source": "network_access_report_2024_Q1.pdf", "page": 0, "start_index": 448}
#                                                                              ^^^^^^^^^^^
#                                                              char offset in original page
# )

# chunk[2]:
# Document(
#   page_content="2. tracking.badactor.io — 3,100 blocks\n3. crypto-miner.net — 980 blocks",
#                  ^^^^ overlap from chunk[1] — context preserved
#   metadata={"source": "...", "page": 0, "start_index": 576}
# )
```

### 5.3 Add Custom Metadata to Every Chunk

```python
import hashlib
from datetime import datetime

def enrich_chunks(chunks: list[Document], doc_id: str) -> list[Document]:
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "doc_id": doc_id,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "chunk_id": f"{doc_id}_chunk_{i}",
            "ingested_at": datetime.utcnow().isoformat(),
        })
    return chunks
```

---

## 6. Embedding + Vector Store

LangChain's `QdrantVectorStore` wraps the Qdrant client and handles embedding + insertion together.

### 6.1 Setup

```python
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Embedding model — same model MUST be used for both ingestion and queries
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # 1536 dimensions

# Qdrant client
qdrant_client = QdrantClient(host="localhost", port=6333)

# Create collection (one-time)
qdrant_client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

# LangChain vector store wrapper
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="documents",
    embedding=embeddings,
)
```

### 6.2 Ingest Chunks

```python
def ingest(file_path: str, doc_id: str):
    # 1. Load
    raw_docs = load_document(file_path)

    # 2. Split into chunks
    chunks = splitter.split_documents(raw_docs)

    # 3. Enrich with metadata
    chunks = enrich_chunks(chunks, doc_id)

    # 4. Embed + store in Qdrant (batched internally)
    vector_store.add_documents(
        documents=chunks,
        ids=[c.metadata["chunk_id"] for c in chunks],
    )
    # add_documents does:
    #   embeddings.embed_documents([c.page_content for c in chunks])
    #   → calls OpenAI API in batches
    #   → upserts vectors + payloads into Qdrant

    print(f"Ingested {len(chunks)} chunks from {file_path}")
```

### 6.3 For Private / On-Prem Deployments

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

# Runs locally — no API calls
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
```

---

## 7. What Gets Stored in the Database

### 7.1 Qdrant Point (one per chunk)

```json
{
  "id": "doc_001_chunk_1",
  "vector": [0.0023, -0.0145, 0.0782, "...1536 floats..."],
  "payload": {
    "page_content": "The top blocked category was malvertising, accounting for 68% of all blocked traffic. Top Blocked Domains: 1. malicious-ads.xyz — 8,200 blocks...",
    "metadata": {
      "doc_id": "doc_001",
      "source": "network_access_report_2024_Q1.pdf",
      "page": 0,
      "chunk_index": 1,
      "total_chunks": 12,
      "chunk_id": "doc_001_chunk_1",
      "start_index": 448,
      "ingested_at": "2024-03-01T10:00:00Z"
    }
  }
}
```

> **Note:** LangChain's `QdrantVectorStore` stores `page_content` and `metadata` as separate payload keys. The `vector` is the embedding of `page_content` only.

### 7.2 PostgreSQL Schema

```sql
-- Source-of-truth for documents
CREATE TABLE documents (
    doc_id       TEXT PRIMARY KEY,
    filename     TEXT NOT NULL,
    source_path  TEXT NOT NULL,
    file_type    TEXT NOT NULL,        -- 'pdf', 'txt', 'log'
    file_hash    TEXT NOT NULL,        -- SHA256, used for change detection
    version      INTEGER DEFAULT 1,
    status       TEXT DEFAULT 'active',
    total_chunks INTEGER,
    ingested_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at   TIMESTAMPTZ DEFAULT NOW()
);

-- Individual chunks (mirrors what's in Qdrant)
CREATE TABLE chunks (
    chunk_id        TEXT PRIMARY KEY,
    doc_id          TEXT REFERENCES documents(doc_id),
    chunk_index     INTEGER,
    text            TEXT NOT NULL,
    page_number     INTEGER,
    start_index     INTEGER,
    token_count     INTEGER,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Audit log for every user query
CREATE TABLE query_log (
    query_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question      TEXT NOT NULL,
    retrieved_ids TEXT[],
    answer        TEXT,
    had_answer    BOOLEAN,
    latency_ms    INTEGER,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);
```

**Example data in `documents`:**

```
doc_id   | filename                           | file_type | version | status
---------|------------------------------------|-----------|---------|--------
doc_001  | network_access_report_2024_Q1.pdf | pdf       | 1       | active
doc_002  | access.log                         | log       | 3       | active
doc_003  | security_policy_v1.pdf             | pdf       | 1       | superseded
```

**Example data in `chunks`:**

```
chunk_id         | doc_id  | idx | text (first 80 chars)                       | page
-----------------|---------|-----|---------------------------------------------|-----
doc_001_chunk_0  | doc_001 | 0   | "Q1 2024 Internet Access Report..."         | 0
doc_001_chunk_1  | doc_001 | 1   | "The top blocked category was malvert..."   | 0
doc_001_chunk_2  | doc_001 | 2   | "Top Blocked Domains: 1. malicious-a..."    | 1
doc_002_chunk_0  | doc_002 | 0   | "2024-01-15 08:32:11 - john.doe acce..."    | null
```

---

## 8. Database Selection

### 8.1 Vector Store Options

| DB | Type | LangChain Integration | When to Use |
|----|------|----------------------|-------------|
| **Qdrant** | Dedicated vector DB | `langchain-qdrant` | Recommended — filtering, sharding, self-hosted |
| **pgvector** | PostgreSQL extension | `langchain-postgres` | Small-medium scale, unified store |
| **Pinecone** | Managed cloud | `langchain-pinecone` | Zero ops, large scale |
| **Chroma** | Embedded | `langchain-chroma` | Local dev only |
| **Weaviate** | Hybrid | `langchain-weaviate` | Keyword + semantic hybrid |

### 8.2 Why Qdrant

1. **Payload filtering** — filter by `file_type`, `doc_id`, date alongside vector search
2. **HNSW index** — sub-10ms search at millions of vectors
3. **Self-hosted** — private corpus stays private
4. **LangChain native** — `langchain-qdrant` wraps it cleanly

---

## 9. LangGraph Query Pipeline

LangGraph models the query pipeline as a **state machine** — each node is a function that reads and writes a shared `State` dict. Edges define the flow, including conditional branches (e.g., skip to "no answer" if nothing was retrieved).

### 9.1 State Definition

```python
from typing import TypedDict, Optional
from langchain.schema import Document

class RAGState(TypedDict):
    question: str
    query_embedding: Optional[list[float]]
    retrieved_chunks: list[Document]
    reranked_chunks: list[Document]
    answer: str
    sources: list[dict]
    has_answer: bool
```

### 9.2 Node Functions

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

MINIMUM_SCORE = 0.60

# ── Node 1: retrieve ─────────────────────────────────────────────
def retrieve_node(state: RAGState) -> RAGState:
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 8, "score_threshold": MINIMUM_SCORE},
    )
    chunks = retriever.invoke(state["question"])
    return {**state, "retrieved_chunks": chunks}

# ── Node 2: rerank ───────────────────────────────────────────────
def rerank_node(state: RAGState) -> RAGState:
    chunks = state["retrieved_chunks"]
    if not chunks:
        return {**state, "reranked_chunks": []}

    pairs = [(state["question"], c.page_content) for c in chunks]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    top3 = [doc for doc, _ in ranked[:3]]

    return {**state, "reranked_chunks": top3}

# ── Node 3: generate ─────────────────────────────────────────────
PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that answers questions based ONLY
on the provided context. If the answer is not in the context, say exactly:
'I don't have enough information to answer this.'
Always cite [Source N] after each fact."""),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

def generate_node(state: RAGState) -> RAGState:
    chunks = state["reranked_chunks"]

    context_parts = []
    sources = []
    for i, chunk in enumerate(chunks):
        label = f"[Source {i+1}: {chunk.metadata['source']}"
        if chunk.metadata.get("page") is not None:
            label += f", page {chunk.metadata['page'] + 1}"
        label += "]"
        context_parts.append(f"{label}\n{chunk.page_content}")
        sources.append({
            "source": chunk.metadata["source"],
            "page": chunk.metadata.get("page"),
            "excerpt": chunk.page_content[:200],
            "chunk_id": chunk.metadata.get("chunk_id"),
        })

    context = "\n\n---\n\n".join(context_parts)
    chain = PROMPT | llm
    response = chain.invoke({"context": context, "question": state["question"]})

    return {
        **state,
        "answer": response.content,
        "sources": sources,
        "has_answer": True,
    }

# ── Node 4: no_answer ────────────────────────────────────────────
def no_answer_node(state: RAGState) -> RAGState:
    return {
        **state,
        "answer": "I don't have enough information in the corpus to answer this question.",
        "sources": [],
        "has_answer": False,
    }
```

### 9.3 Build the Graph

```python
from langgraph.graph import StateGraph, END

def has_results(state: RAGState) -> str:
    """Routing function: go to generate if we have chunks, else no_answer."""
    return "generate" if state["reranked_chunks"] else "no_answer"

# Build graph
builder = StateGraph(RAGState)

# Add nodes
builder.add_node("retrieve", retrieve_node)
builder.add_node("rerank",   rerank_node)
builder.add_node("generate", generate_node)
builder.add_node("no_answer", no_answer_node)

# Define flow
builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "rerank")

# Conditional edge: after rerank, branch based on whether chunks exist
builder.add_conditional_edges(
    "rerank",
    has_results,
    {
        "generate":  "generate",
        "no_answer": "no_answer",
    }
)

builder.add_edge("generate",  END)
builder.add_edge("no_answer", END)

# Compile into a runnable
rag_graph = builder.compile()
```

### 9.4 Visualize the Graph

```python
# Print ASCII graph structure
print(rag_graph.get_graph().draw_ascii())

# Output:
# +-----------+
# | __start__ |
# +-----------+
#       *
#       *
#       *
# +----------+
# | retrieve |
# +----------+
#       *
#       *
#       *
# +--------+
# | rerank |
# +--------+
#      **         **
#    **               **
#   *                   *
# +----------+    +-----------+
# | generate |    | no_answer |
# +----------+    +-----------+
#      *                *
#      *                *
#      *                *
# +---------+
# | __end__ |
# +---------+
```

### 9.5 Run the Pipeline

```python
def ask(question: str) -> dict:
    initial_state: RAGState = {
        "question": question,
        "query_embedding": None,
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "answer": "",
        "sources": [],
        "has_answer": False,
    }

    result = rag_graph.invoke(initial_state)

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "has_answer": result["has_answer"],
    }
```

---

## 10. Each LangGraph Node Explained

```
User Question
     │
     ▼
┌─────────────────────────────────────────────────────┐
│ NODE: retrieve                                      │
│                                                     │
│  vector_store.as_retriever(                         │
│      search_type="similarity_score_threshold",      │
│      search_kwargs={"k": 8, "score_threshold": 0.6} │
│  )                                                  │
│                                                     │
│  → Embeds the question                              │
│  → Cosine search in Qdrant                          │
│  → Returns up to 8 chunks scoring ≥ 0.60            │
│  → Writes to state["retrieved_chunks"]              │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│ NODE: rerank                                        │
│                                                     │
│  CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6")  │
│                                                     │
│  → Scores each (question, chunk) pair jointly       │
│    (bi-encoder finds candidates, cross-encoder      │
│     picks the best from those candidates)           │
│  → Sorts descending, keeps top 3                    │
│  → Writes to state["reranked_chunks"]               │
└─────────────────────────────────────────────────────┘
     │
     ▼ (conditional)
  has chunks?
  ┌────YES────┐     ┌────NO────┐
  ▼           │     ▼          │
┌──────────┐  │  ┌───────────┐ │
│ generate │  │  │ no_answer │ │
│          │  │  │           │ │
│ PROMPT   │  │  │ Returns   │ │
│  | LLM   │  │  │ fallback  │ │
│  chain   │  │  │ message   │ │
└──────────┘  │  └───────────┘ │
     │        └────────────────┘
     ▼
  [Answer + Sources]
```

---

## 11. LLM Generation with Grounding

### LangChain LCEL Chain (used inside `generate_node`)

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer ONLY from the context below.
Rules:
- Cite [Source N] after each fact you state
- If the answer is not in the context, say: 'I don't have enough information'
- Do not invent statistics, names, or dates"""),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

# LCEL pipe: prompt → llm → parse to string
chain = PROMPT | llm | StrOutputParser()

# Usage
answer = chain.invoke({
    "context": "...",
    "question": "What were the top blocked domains?"
})
```

**Example output:**

```json
{
  "answer": "The top blocked domains in Q1 2024 were:\n1. malicious-ads.xyz — 8,200 blocks [Source 1]\n2. tracking.badactor.io — 3,100 blocks [Source 1]\n3. crypto-miner.net — 980 blocks [Source 1]\n\nMalvertising was the dominant blocked category at 68% of all blocked traffic [Source 2].",
  "sources": [
    {
      "source": "network_access_report_2024_Q1.pdf",
      "page": 1,
      "excerpt": "Top Blocked Domains: 1. malicious-ads.xyz — 8,200 blocks...",
      "chunk_id": "doc_001_chunk_2"
    },
    {
      "source": "network_access_report_2024_Q1.pdf",
      "page": 0,
      "excerpt": "The top blocked category was malvertising, accounting for 68%...",
      "chunk_id": "doc_001_chunk_1"
    }
  ],
  "has_answer": true
}
```

---

## 12. Handling Frequent Document Updates

### 12.1 Change Detection via SHA256

```python
import hashlib

def compute_hash(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def is_changed(file_path: str, db_session) -> bool:
    row = db_session.execute(
        "SELECT file_hash FROM documents WHERE source_path = %s", (file_path,)
    ).fetchone()
    if not row:
        return True  # new document
    return row["file_hash"] != compute_hash(file_path)
```

### 12.2 Update Flow

```python
from qdrant_client.models import PointIdsList

def update_document(file_path: str, doc_id: str, db_session):
    # 1. Delete old Qdrant vectors for this doc
    old_chunk_ids = db_session.execute(
        "SELECT chunk_id FROM chunks WHERE doc_id = %s", (doc_id,)
    ).fetchall()

    qdrant_client.delete(
        collection_name="documents",
        points_selector=PointIdsList(points=[r["chunk_id"] for r in old_chunk_ids]),
    )

    # 2. Delete old chunks from Postgres
    db_session.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))

    # 3. Bump version in documents table
    db_session.execute("""
        UPDATE documents
        SET version = version + 1,
            file_hash = %s,
            updated_at = NOW()
        WHERE doc_id = %s
    """, (compute_hash(file_path), doc_id))

    # 4. Re-ingest (LangChain pipeline)
    ingest(file_path, doc_id)
```

### 12.3 Real-time Log Tail Ingestion

```python
import time

def tail_and_ingest(log_path: str, doc_id: str, poll_interval: int = 30):
    """Continuously watches a log file and ingests new lines."""
    last_pos = 0

    while True:
        with open(log_path) as f:
            f.seek(last_pos)
            new_lines = f.readlines()
            last_pos = f.tell()

        if new_lines:
            # Wrap new lines as a Document and push through LangChain pipeline
            from langchain.schema import Document
            doc = Document(
                page_content="".join(new_lines),
                metadata={"source": log_path, "file_type": "log"},
            )
            chunks = splitter.split_documents([doc])
            chunks = enrich_chunks(chunks, doc_id=f"{doc_id}_tail_{int(time.time())}")
            vector_store.add_documents(chunks)

        time.sleep(poll_interval)
```

---

## 13. Reducing Hallucination

| Technique | Where Applied |
|-----------|--------------|
| Strict system prompt | `generate_node` — "answer ONLY from context" |
| `temperature=0.1` | `ChatOpenAI(temperature=0.1)` |
| Score threshold | `retriever(score_threshold=0.60)` |
| Re-ranking | `rerank_node` — cross-encoder picks most relevant chunks |
| No-answer fallback | `no_answer_node` — triggered when `reranked_chunks` is empty |
| Citation requirement | Prompt enforces `[Source N]` after every fact |
| Context size cap | Top 3 chunks only — avoids diluting context with weak results |

---

## 14. Low Latency Design

### Target: < 500ms end-to-end

| Step | Latency | Optimization |
|------|---------|-------------|
| Query embedding | ~50ms | Redis cache on embedding |
| Qdrant HNSW search | ~10-30ms | In-memory index |
| Cross-encoder re-rank | ~80-120ms | MiniLM (small model) |
| LLM generation | ~300-600ms | `gpt-4o-mini` + streaming |
| **Total** | **~440-800ms** | Stream tokens to reduce perceived latency |

### Redis Cache for Embeddings + Retrieval

```python
import redis
import json
import hashlib

cache = redis.Redis()

def cached_ask(question: str) -> dict:
    key = "rag:" + hashlib.md5(question.encode()).hexdigest()
    hit = cache.get(key)

    if hit:
        return json.loads(hit)

    result = ask(question)
    cache.setex(key, 3600, json.dumps(result))  # cache 1 hour
    return result
```

### Streaming Tokens via LangChain

```python
from langchain_openai import ChatOpenAI

streaming_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, streaming=True)

async def ask_streaming(question: str):
    chunks = retrieve_node({"question": question, "retrieved_chunks": []})["retrieved_chunks"]
    context = "\n\n".join(c.page_content for c in chunks[:3])

    chain = PROMPT | streaming_llm | StrOutputParser()

    async for token in chain.astream({"context": context, "question": question}):
        yield token  # push each token to WebSocket / SSE as it arrives
```

---

## 15. Scalability

### 15.1 Async Ingestion with LangChain + Kafka

```
[File Upload API]
      │
      ▼
[Kafka Topic: doc-ingestion]
      │
   ┌──┴───────────────────┐
   ▼                      ▼
[Worker 1]           [Worker 2]
load → chunk         load → chunk
→ embed → store      → embed → store
(LangChain pipeline) (LangChain pipeline)
```

```python
from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    "doc-ingestion",
    bootstrap_servers=["localhost:9092"],
    value_deserializer=lambda m: json.loads(m.decode()),
)

for message in consumer:
    payload = message.value
    # Each message: {"file_path": "...", "doc_id": "..."}
    ingest(payload["file_path"], payload["doc_id"])
```

### 15.2 Qdrant Sharded Collection

```python
qdrant_client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    shard_number=4,         # distribute load across 4 shards
    replication_factor=2,   # 2 replicas per shard for HA
)
```

---

## 16. Full End-to-End Example

### User Question

```
"How many requests were blocked in Q1 2024 and which domain had the most blocks?"
```

### LangGraph Execution Trace

```
State entering retrieve_node:
  question = "How many requests were blocked in Q1 2024..."

── retrieve_node ──────────────────────────────────────────────────
  Embeds question → [0.012, -0.034, ...]
  Qdrant search → 8 candidates

  doc_001_chunk_2  score=0.91  "Top Blocked Domains: malicious-ads.xyz — 8,200..."
  doc_001_chunk_1  score=0.88  "...12,400 flagged as suspicious..."
  doc_001_chunk_0  score=0.82  "4.2 million HTTP requests were logged..."
  doc_002_chunk_7  score=0.63  "jane.smith accessed malicious-ads.xyz [BLOCK]"
  doc_002_chunk_2  score=0.55  "john.doe accessed github.com [ALLOW]"  ← below threshold, dropped

  → state["retrieved_chunks"] = 4 chunks

── rerank_node ────────────────────────────────────────────────────
  CrossEncoder scores 4 pairs:
  chunk_2: 8.91  chunk_1: 7.34  chunk_0: 6.12  chunk_7: 4.88

  → state["reranked_chunks"] = [chunk_2, chunk_1, chunk_0]  (top 3)

── has_results → "generate" ───────────────────────────────────────

── generate_node ──────────────────────────────────────────────────
  Builds prompt with 3 source blocks
  Calls gpt-4o-mini (temperature=0.1)

  → state["answer"] = "In Q1 2024, a total of 12,400..."
  → state["sources"] = [{source: ..., page: ...}, ...]
```

### Final Output

```json
{
  "answer": "In Q1 2024, a total of 12,400 HTTP requests were flagged and blocked out of 4.2 million total requests [Source 2]. The domain with the most blocks was malicious-ads.xyz with 8,200 blocks, followed by tracking.badactor.io with 3,100 blocks and crypto-miner.net with 980 blocks [Source 1].",
  "sources": [
    {
      "source": "network_access_report_2024_Q1.pdf",
      "page": 1,
      "excerpt": "Top Blocked Domains: 1. malicious-ads.xyz — 8,200 blocks...",
      "chunk_id": "doc_001_chunk_2"
    },
    {
      "source": "network_access_report_2024_Q1.pdf",
      "page": 0,
      "excerpt": "...12,400 were flagged as suspicious based on threat intelligence feeds...",
      "chunk_id": "doc_001_chunk_1"
    }
  ],
  "has_answer": true
}
```

---

## Summary: Technology Stack

| Component | Technology | LangChain Integration |
|-----------|-----------|----------------------|
| PDF parsing | `PyPDFLoader` | `langchain-community` |
| Log parsing | Custom `BaseLoader` | `langchain` base class |
| Chunking | `RecursiveCharacterTextSplitter` | `langchain` |
| Embeddings | `OpenAIEmbeddings` | `langchain-openai` |
| Vector store | **Qdrant** | `langchain-qdrant` |
| Metadata store | **PostgreSQL** | `psycopg2` (direct) |
| Query pipeline | **LangGraph StateGraph** | `langgraph` |
| LLM | `ChatOpenAI` (gpt-4o-mini) | `langchain-openai` |
| Re-ranker | `CrossEncoder` (MiniLM) | `sentence-transformers` |
| Caching | Redis | `redis-py` |
| Queue (scale) | Kafka | `kafka-python` |

---

## 17. Hybrid RAG: Improving Retrieval Beyond Dense Vectors

Pure dense vector search has a known weakness: it struggles with **exact keyword matches**, rare terms, and entity names (e.g. `CVE-2024-1234`, `ProjectAlpha_v2`). Hybrid RAG combines **dense (semantic) + sparse (keyword) retrieval**, then fuses results for best-of-both worlds.

### How It Works

```
User Query
    │
    ├──► Dense Retriever   (Qdrant cosine similarity on OpenAI embeddings)
    │         └── Top-K semantic matches
    │
    ├──► Sparse Retriever  (Qdrant sparse vectors — BM25-style keyword index)
    │         └── Top-K exact/keyword matches
    │
    └──► Reciprocal Rank Fusion (RRF)
              └── Unified ranked list → Re-ranker → LLM
```

### Reciprocal Rank Fusion (RRF)

RRF merges two ranked lists without needing score normalization. Each document's fused score is:

```
RRF_score(d) = Σ  1 / (k + rank_i(d))
```

Where `k=60` (default constant), `rank_i(d)` is the document's rank in retriever `i`. Documents appearing in both lists get a natural boost.

```python
def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    scores = {}
    for rank, doc in enumerate(dense_results):
        scores[doc.id] = scores.get(doc.id, 0) + 1 / (k + rank + 1)
    for rank, doc in enumerate(sparse_results):
        scores[doc.id] = scores.get(doc.id, 0) + 1 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
```

### Sparse Retriever Options

| Option | Pros | Cons |
|--------|------|------|
| **BM25** (rank_bm25 lib) | Zero infra, fast | In-memory, no persistence |
| **Elasticsearch** | Production-grade, scalable | Extra infra to manage |
| **Qdrant sparse vectors** | Single store, BM25-compatible | Requires sparse vector ingestion |

Using Qdrant's native sparse vector support keeps everything in one store:

```python
from qdrant_client.models import SparseVector

# At ingest time, compute sparse (BM25-style) vectors alongside dense embeddings
client.upsert(
    collection_name="docs",
    points=[
        PointStruct(
            id=doc_id,
            vector={
                "dense": dense_embedding,       # float list
                "sparse": SparseVector(          # BM25 token weights
                    indices=token_ids,
                    values=token_weights
                )
            },
            payload=metadata
        )
    ]
)
```

### LangGraph Node: Hybrid Retrieval

```python
from qdrant_client.models import SparseVector, NamedSparseVector
from langchain_core.documents import Document

def bm25_retrieve(query: str, k: int = 20) -> list[Document]:
    """Query Qdrant's sparse vector index (BM25-style) for keyword matches."""
    query_tokens = tokenize(query)  # same tokenizer used at ingest time
    query_sparse = SparseVector(
        indices=query_tokens.ids,
        values=query_tokens.weights
    )
    results = client.search(
        collection_name="docs",
        query_vector=NamedSparseVector(name="sparse", vector=query_sparse),
        limit=k,
    )
    return [
        Document(page_content=r.payload["text"], metadata={**r.payload, "id": r.id})
        for r in results
    ]


def hybrid_retrieve_node(state: RAGState) -> RAGState:
    query = state["query"]

    # Dense retrieval
    dense_docs = qdrant_retriever.get_relevant_documents(query, k=20)

    # Sparse retrieval — queries Qdrant sparse vector index (BM25-style)
    sparse_docs = bm25_retrieve(query, k=20)

    # Fuse with RRF
    fused_ids = reciprocal_rank_fusion(dense_docs, sparse_docs)
    all_docs = {d.metadata["id"]: d for d in dense_docs + sparse_docs}
    fused_docs = [all_docs[id] for id in fused_ids[:10] if id in all_docs]

    return {**state, "retrieved_docs": fused_docs}
```

### When to Use Hybrid RAG

| Scenario | Dense Only | Hybrid |
|----------|------------|--------|
| Semantic / conceptual questions | ✅ Good | ✅ Good |
| Exact product names, IDs, codes | ❌ Weak | ✅ Strong |
| Mixed queries ("errors in ProjectAlpha") | ❌ Weak | ✅ Strong |
| Short keyword queries | ❌ Weak | ✅ Strong |

---

## 18. Evaluating the RAG Pipeline

Evaluation must cover both stages independently: **retrieval quality** and **generation quality**. Mixing them makes it impossible to diagnose failures.

### Retrieval Evaluation

The goal: did we fetch the right chunks?

#### Metrics

| Metric | Formula | What It Tells You |
|--------|---------|-------------------|
| **Recall@K** | relevant retrieved / total relevant | Are the right docs in the top-K? |
| **Precision@K** | relevant retrieved / K | Are retrieved docs mostly relevant? |
| **MRR** (Mean Reciprocal Rank) | avg(1 / rank of first relevant) | How high does the first relevant doc rank? |
| **NDCG@K** | discounted cumulative gain at K | Graded relevance; penalizes relevant docs ranked low |
| **Context Relevance** | LLM-as-judge (0–1 per chunk) | Is each retrieved chunk relevant to the query? |

#### Test Set Construction

Build a **golden dataset** of (query, expected_chunk_ids) pairs. Two approaches:

```python
# Option 1: Manual annotation
golden = [
    {"query": "What is the auth token expiry?", "relevant_ids": ["doc_42_chunk_3"]},
    ...
]

# Option 2: Synthetic via LLM (generate questions from known chunks)
# For each chunk, prompt the LLM: "Write 3 questions answered by this chunk"
# This gives 100s of eval pairs cheaply
```

#### Running Retrieval Eval

```python
from ragas import evaluate
from ragas.metrics import context_recall, context_precision

result = evaluate(
    dataset=golden_dataset,       # HuggingFace Dataset with question + ground_truth_context
    metrics=[context_recall, context_precision],
    llm=eval_llm,
    embeddings=eval_embeddings
)
print(result)  # {'context_recall': 0.87, 'context_precision': 0.79}
```

### Generation Evaluation

The goal: given the retrieved context, did the LLM answer correctly?

#### Metrics

| Metric | Tool | What It Measures |
|--------|------|-----------------|
| **Faithfulness** | RAGAS | Is the answer grounded in retrieved context? (hallucination detector) |
| **Answer Relevance** | RAGAS | Does the answer address the question? |
| **Answer Correctness** | RAGAS | Factual match against a reference answer |
| **ROUGE-L / BERTScore** | HF evaluate | Surface-level / semantic similarity to reference |
| **LLM-as-judge** | GPT-4o / Claude | Holistic quality score; useful when no reference exists |

#### Faithfulness Check (Anti-Hallucination)

RAGAS faithfulness decomposes the answer into atomic claims and verifies each against retrieved chunks:

```python
from ragas.metrics import faithfulness, answer_relevancy

result = evaluate(
    dataset=eval_dataset,   # question, answer, contexts
    metrics=[faithfulness, answer_relevancy]
)
# faithfulness: 0.0 = full hallucination, 1.0 = fully grounded
```

#### LLM-as-Judge (No Reference Needed)

When you don't have ground-truth answers, use the LLM to score responses:

```python
JUDGE_PROMPT = """
You are an evaluation assistant.
Question: {question}
Retrieved Context: {context}
Answer: {answer}

Score on three dimensions (1-5 each):
- Faithfulness: Is every claim in the answer supported by the context?
- Relevance: Does the answer directly address the question?
- Completeness: Does the answer cover all relevant information in the context?

Return JSON: {{"faithfulness": N, "relevance": N, "completeness": N, "reasoning": "..."}}
"""
```

### End-to-End Evaluation Pipeline

```
Golden Dataset (query + expected answer + expected chunk IDs)
          │
          ├──► Run RAG pipeline → collected (query, context, answer)
          │
          ├──► Retrieval metrics:  Recall@K, Precision@K, Context Recall (RAGAS)
          │
          └──► Generation metrics: Faithfulness, Answer Relevance (RAGAS)
                                   + LLM-as-Judge for edge cases
```

```python
# Recommended tooling
# RAGAS:       pip install ragas          — retrieval + generation metrics out of the box
# DeepEval:    pip install deepeval       — CI integration, regression tracking
# LangSmith:   LangChain tracing + eval  — trace every node, tag eval runs
```

### Evaluation Stack Summary

| Stage | Primary Tool | Key Metric to Watch |
|-------|-------------|---------------------|
| Retrieval | RAGAS + golden dataset | Context Recall ≥ 0.85 |
| Generation | RAGAS faithfulness | Faithfulness ≥ 0.90 |
| Regression CI | DeepEval | No metric drops > 5% vs baseline |
| Production monitoring | LangSmith traces | Flag low-confidence answers in real time |
