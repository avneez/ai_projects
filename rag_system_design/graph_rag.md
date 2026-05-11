# Graph RAG — Complete In-Depth Guide

## Table of Contents
1. [What is Graph RAG?](#1-what-is-graph-rag)
2. [Why Graph RAG over Vanilla RAG?](#2-why-graph-rag-over-vanilla-rag)
3. [Core Concepts](#3-core-concepts)
4. [Architecture](#4-architecture)
5. [The Two Phases: Indexing & Querying](#5-the-two-phases-indexing--querying)
6. [Community Detection & Summarization](#6-community-detection--summarization)
7. [Query Strategies: Local vs Global](#7-query-strategies-local-vs-global)
8. [Full Working Example (Python)](#8-full-working-example-python)
9. [Graph RAG with Neo4j + LangChain](#9-graph-rag-with-neo4j--langchain)
10. [Comparison: Vanilla RAG vs Graph RAG](#10-comparison-vanilla-rag-vs-graph-rag)
11. [When to Use Graph RAG](#11-when-to-use-graph-rag)
12. [Best Practices](#12-best-practices)

---

## 1. What is Graph RAG?

**Graph RAG** (Graph Retrieval-Augmented Generation) is an evolution of standard RAG that represents a document corpus as a **knowledge graph** — a network of entities and relationships — instead of isolated vector chunks.

Developed and open-sourced by Microsoft Research (2024), the key insight is:

> "Real-world knowledge is relational. A graph preserves those relationships. A flat vector index throws them away."

```
Vanilla RAG:                    Graph RAG:
                                
  Doc1 → chunk → vector          Doc1 ─┐
  Doc2 → chunk → vector          Doc2 ─┤──► Knowledge Graph ──► Communities
  Doc3 → chunk → vector          Doc3 ─┘    (entities +          (summaries
                                             relationships)        at each level)
  Query → nearest vectors         
        → answer from snippets   Query → graph traversal / community lookup
                                        → answer from structured context
```

**What a knowledge graph looks like:**
```
(Alice) ──[WORKS_AT]──► (Acme Corp)
  │                          │
[MANAGES]             [LOCATED_IN]
  │                          │
(Bob)          (New York) ◄──┘
  │
[AUTHORED]
  │
(Report Q3 2024)
```

---

## 2. Why Graph RAG over Vanilla RAG?

### The Core Limitation of Vector RAG

Vanilla RAG retrieves **semantically similar chunks** — but similarity ≠ relevance for complex questions.

```
Question: "What are the main themes across all of our Q3 reports?"

Vanilla RAG:
  → finds top-k chunks most similar to the question text
  → misses cross-document patterns (different chunks never "talk" to each other)
  → can't answer dataset-wide, holistic questions

Graph RAG:
  → has pre-built community summaries that capture cross-document themes
  → answers: "The 3 main themes are X, Y, Z based on analysis of all documents"
```

### Problem Breakdown

| Question Type | Vanilla RAG | Graph RAG |
|---|---|---|
| "What did Alice say about pricing?" | ✅ Good (local, specific) | ✅ Good |
| "How does Alice's pricing view relate to Bob's?" | ❌ Misses connection | ✅ Traverses relationship |
| "What are the main topics in all company memos?" | ❌ Only sees top-k chunks | ✅ Community summaries |
| "Who are the key people connected to Project X?" | ❌ Poor | ✅ Graph traversal |
| "Summarize the entire document corpus" | ❌ Poor | ✅ Root community summary |

---

## 3. Core Concepts

### Entity
A named, real-world thing extracted from text.
```
Types: Person, Organization, Location, Concept, Event, Product, ...
Example: "Apple Inc.", "Tim Cook", "iPhone 15", "WWDC 2024"
```

### Relationship
A directed, labeled connection between two entities.
```
(Tim Cook) ──[CEO_OF]──► (Apple Inc.)
(iPhone 15) ──[RELEASED_AT]──► (WWDC 2024)
(Apple Inc.) ──[COMPETES_WITH]──► (Samsung)
```

### Community
A cluster of closely-connected entities and relationships — like a "topic cluster."
```
Community 1: Apple Inc., Tim Cook, iPhone, iOS, App Store
Community 2: Samsung, Galaxy, Android, Google
Community 3: WWDC, Apple Events, Product Launches
```

### Community Summary
An LLM-generated paragraph describing what each community is about — pre-computed at index time.
```
Community 1 Summary:
"Apple Inc., led by CEO Tim Cook, is the company behind the iPhone product 
line and iOS ecosystem. The App Store serves as the primary distribution 
platform for iOS applications..."
```

### Leiden Algorithm
The graph clustering algorithm used to detect communities at multiple hierarchical levels (C0, C1, C2...).

---

## 4. Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║                         INDEXING PHASE                               ║
║                                                                       ║
║  Raw Documents                                                        ║
║       │                                                               ║
║       ▼                                                               ║
║  ┌─────────────┐    LLM Extraction    ┌───────────────────────┐      ║
║  │   Chunking  │ ──────────────────►  │  Entity + Relation    │      ║
║  │  (text split│                      │  Extraction           │      ║
║  │   ~600 tok) │                      └──────────┬────────────┘      ║
║  └─────────────┘                                 │                   ║
║                                                  ▼                   ║
║                                     ┌────────────────────────┐       ║
║                                     │   Knowledge Graph      │       ║
║                                     │   (nodes + edges)      │       ║
║                                     └──────────┬─────────────┘       ║
║                                                │                     ║
║                                                ▼                     ║
║                                     ┌────────────────────────┐       ║
║                                     │  Community Detection   │       ║
║                                     │  (Leiden Algorithm)    │       ║
║                                     └──────────┬─────────────┘       ║
║                                                │                     ║
║                                                ▼                     ║
║                                     ┌────────────────────────┐       ║
║                                     │  Community Summaries   │       ║
║                                     │  (LLM-generated,       │       ║
║                                     │   hierarchical)        │       ║
║                                     └────────────────────────┘       ║
╚══════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════╗
║                         QUERY PHASE                                  ║
║                                                                       ║
║  User Query                                                           ║
║       │                                                               ║
║       ▼                                                               ║
║  ┌──────────────────────────────────────────────────────────┐        ║
║  │  Query Router                                             │        ║
║  │    LOCAL query?  ──► Entity search → Graph traversal     │        ║
║  │    GLOBAL query? ──► Community summaries → Map-reduce    │        ║
║  └──────────────────────────────────────────────────────────┘        ║
║                                                                       ║
║  Final Answer ◄── LLM synthesis from retrieved context               ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 5. The Two Phases: Indexing & Querying

### Phase 1: Indexing (One-time, Expensive)

```
Step 1: Chunk documents
        "Alice manages the Berlin office. Bob reports to Alice."
                           │
Step 2: Extract entities + relationships  (via LLM)
        Entities:   Alice (Person), Bob (Person), Berlin (Location)
        Relations:  Alice -[MANAGES]-> Berlin office
                    Bob -[REPORTS_TO]-> Alice
                           │
Step 3: Build graph
        Nodes: Alice, Bob, Berlin office
        Edges: MANAGES, REPORTS_TO, LOCATED_IN
                           │
Step 4: Detect communities (Leiden)
        Community A: { Alice, Bob, Berlin office }  ← all connected
                           │
Step 5: Summarize communities (via LLM)
        "Alice leads the Berlin team. Bob is a direct report. 
         The Berlin office handles EMEA operations."
```

### Phase 2: Querying

```
Query: "What does Alice do?"

LOCAL search:
  1. Find entity "Alice" in graph
  2. Retrieve her neighbors (Bob, Berlin office) and edge labels
  3. Fetch source text chunks mentioning Alice
  4. LLM generates answer from entity context + chunks

Query: "What are the key themes in all company documents?"

GLOBAL search:
  1. Retrieve all community summaries (or top-level ones)
  2. Map step: LLM scores each community summary for relevance to query
  3. Reduce step: LLM synthesizes a final answer from top-scored summaries
```

---

## 6. Community Detection & Summarization

### Hierarchical Levels

The Leiden algorithm builds communities at multiple resolutions:

```
Level 0 (finest):
  Community 0-A: { Alice, Bob }
  Community 0-B: { Berlin, Paris, London }
  Community 0-C: { Q3 Report, Q4 Report }

Level 1 (coarser):
  Community 1-A: { Alice, Bob, Berlin, Paris }  ← merged from 0-A + 0-B
  Community 1-B: { Q3 Report, Q4 Report, ... }

Level 2 (coarsest / root):
  Community 2-A: everything  ← global summary of entire corpus
```

Global queries use higher-level summaries (fast, broad).
Local queries drill into lower-level communities (specific, precise).

### Summary Generation Prompt (used internally)

```python
COMMUNITY_SUMMARY_PROMPT = """
You are given a set of entities and relationships extracted from documents.

Entities:
{entities}

Relationships:
{relationships}

Source text excerpts:
{text_units}

Write a comprehensive summary of this community of information. 
Include: key entities, their roles, important relationships, and notable facts.
Keep it under 500 words.
"""
```

---

## 7. Query Strategies: Local vs Global

### Local Search — best for specific, entity-focused questions

```
"What projects is Alice working on?"
"How are Apple and Samsung related?"
"What did the Q3 report say about revenue?"

Process:
  1. Embed query → find similar entities/relationships in vector index
  2. Pull entity descriptions + relationship context from graph
  3. Pull raw text chunks linked to those entities
  4. Feed all into LLM context window → answer
```

### Global Search — best for holistic, thematic questions

```
"What are the main risks across all reports?"
"Summarize the overall state of the company"
"What recurring themes appear in all meeting notes?"

Process:
  1. Retrieve all community summaries at an appropriate level
  2. MAP: for each summary, ask LLM "Is this relevant? Rate 0-100."
  3. Keep top-scored summaries
  4. REDUCE: ask LLM to synthesize a final answer from top summaries
```

```
MAP prompt (per community):
  "Given this community summary: {summary}
   How relevant is this to the question: {query}?
   Score 0-100 and extract the key points relevant to the question."

REDUCE prompt:
  "Using these relevant points from across the document corpus:
   {map_results}
   Answer the question: {query}
   Provide a comprehensive, well-supported answer."
```

---

## 8. Full Working Example (Python)

This example builds a mini Graph RAG system from scratch — no external Graph RAG library required — to show the internals clearly.

### Install Dependencies

```bash
pip install openai networkx python-louvain numpy scikit-learn
```

### Full Code

```python
# graph_rag_example.py

import json
import re
import networkx as nx
from openai import OpenAI
import community as community_louvain  # python-louvain

client = OpenAI()  # set OPENAI_API_KEY env var, or swap for Anthropic client

# ─────────────────────────────────────────────────────────────
# SAMPLE CORPUS
# ─────────────────────────────────────────────────────────────

DOCUMENTS = [
    "Alice is the VP of Engineering at TechCorp. She manages three teams: Platform, Data, and Security.",
    "Bob leads the Data team under Alice. His team built the real-time analytics pipeline used by 50+ clients.",
    "The Platform team, led by Carol, is responsible for TechCorp's cloud infrastructure on AWS.",
    "TechCorp competes directly with DataSoft in the analytics market. DataSoft recently raised $200M Series C.",
    "Alice and the CEO, David, presented TechCorp's Q3 results. Revenue grew 40% YoY driven by the Data team.",
    "The Security team, managed by Eve, reported three critical vulnerabilities patched in Q3.",
    "Bob's analytics pipeline integrates with Snowflake, dbt, and Apache Kafka for real-time processing.",
    "DataSoft's CEO, Frank, announced a new AI-powered analytics product targeting TechCorp's core market.",
]

# ─────────────────────────────────────────────────────────────
# STEP 1: ENTITY + RELATIONSHIP EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_entities_and_relations(text: str) -> dict:
    """Use LLM to extract entities and relationships from a text chunk."""
    prompt = f"""Extract entities and relationships from the following text.

Text: "{text}"

Return JSON with this exact format:
{{
  "entities": [
    {{"name": "Alice", "type": "Person", "description": "VP of Engineering at TechCorp"}}
  ],
  "relationships": [
    {{"source": "Alice", "target": "TechCorp", "relation": "WORKS_AT", "description": "Alice is VP of Engineering"}}
  ]
}}

Only return valid JSON. No explanation."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown code blocks if present
    raw = re.sub(r"^```json\s*|```$", "", raw, flags=re.MULTILINE).strip()
    return json.loads(raw)


def build_knowledge_graph(documents: list[str]) -> tuple[nx.Graph, dict]:
    """Extract entities/relations from all docs and build a NetworkX graph."""
    G = nx.Graph()
    entity_info = {}  # entity_name -> {type, description, source_texts}

    for i, doc in enumerate(documents):
        print(f"  Extracting from doc {i+1}/{len(documents)}...")
        data = extract_entities_and_relations(doc)

        for entity in data.get("entities", []):
            name = entity["name"]
            if name not in entity_info:
                entity_info[name] = {
                    "type": entity["type"],
                    "description": entity["description"],
                    "source_texts": [],
                }
            entity_info[name]["source_texts"].append(doc)
            G.add_node(name, **entity_info[name])

        for rel in data.get("relationships", []):
            src, tgt = rel["source"], rel["target"]
            if src in G.nodes and tgt in G.nodes:
                G.add_edge(src, tgt, relation=rel["relation"], description=rel["description"])

    return G, entity_info


# ─────────────────────────────────────────────────────────────
# STEP 2: COMMUNITY DETECTION
# ─────────────────────────────────────────────────────────────

def detect_communities(G: nx.Graph) -> dict:
    """Use Louvain algorithm to detect communities. Returns node -> community_id."""
    partition = community_louvain.best_partition(G)
    return partition  # {node_name: community_id}


def group_by_community(partition: dict) -> dict:
    """Invert partition: community_id -> [nodes]."""
    communities = {}
    for node, cid in partition.items():
        communities.setdefault(cid, []).append(node)
    return communities


# ─────────────────────────────────────────────────────────────
# STEP 3: COMMUNITY SUMMARIZATION
# ─────────────────────────────────────────────────────────────

def summarize_community(community_nodes: list[str], G: nx.Graph, entity_info: dict) -> str:
    """Ask LLM to summarize a community of entities."""
    entities_text = "\n".join(
        f"- {n} ({entity_info.get(n, {}).get('type', 'Unknown')}): {entity_info.get(n, {}).get('description', '')}"
        for n in community_nodes
    )

    edges_in_community = [
        (u, v, d) for u, v, d in G.edges(data=True)
        if u in community_nodes and v in community_nodes
    ]
    relations_text = "\n".join(
        f"- {u} --[{d['relation']}]--> {v}: {d.get('description', '')}"
        for u, v, d in edges_in_community
    )

    # Collect source texts
    source_texts = set()
    for n in community_nodes:
        for t in entity_info.get(n, {}).get("source_texts", []):
            source_texts.add(t)
    sources_text = "\n".join(f"- {t}" for t in list(source_texts)[:5])

    prompt = f"""You are analyzing a cluster of related entities and their relationships.

Entities in this cluster:
{entities_text}

Relationships:
{relations_text if relations_text else "None within this cluster"}

Source text excerpts mentioning these entities:
{sources_text}

Write a 2-4 sentence summary of what this cluster represents, who the key players are,
and what their main roles and relationships are."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────
# STEP 4: LOCAL SEARCH
# ─────────────────────────────────────────────────────────────

def local_search(query: str, G: nx.Graph, entity_info: dict) -> str:
    """Answer a specific entity-focused question using graph traversal."""
    # Simple keyword match to find relevant entities (production: use vector similarity)
    query_lower = query.lower()
    matched_entities = [
        n for n in G.nodes
        if n.lower() in query_lower
    ]

    if not matched_entities:
        # Fall back to any entity whose description matches
        matched_entities = [
            n for n in G.nodes
            if any(word in entity_info.get(n, {}).get("description", "").lower()
                   for word in query_lower.split())
        ][:3]

    if not matched_entities:
        return "No relevant entities found for this query."

    # Build context: entity info + neighbors + source texts
    context_parts = []
    for entity in matched_entities[:3]:
        info = entity_info.get(entity, {})
        neighbors = list(G.neighbors(entity))
        edges = [(entity, nb, G[entity][nb]) for nb in neighbors]

        context_parts.append(f"""
Entity: {entity} ({info.get('type', '')})
Description: {info.get('description', '')}
Connected to: {', '.join(neighbors)}
Relationships: {'; '.join(f"{u} -[{d['relation']}]-> {v}" for u,v,d in edges[:5])}
Source texts: {' | '.join(info.get('source_texts', [])[:2])}
""")

    context = "\n---\n".join(context_parts)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer questions using the provided knowledge graph context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────
# STEP 5: GLOBAL SEARCH (Map-Reduce over community summaries)
# ─────────────────────────────────────────────────────────────

def global_search(query: str, community_summaries: dict) -> str:
    """Answer a holistic question using map-reduce over community summaries."""

    # MAP: score each community summary for relevance
    map_results = []
    for cid, summary in community_summaries.items():
        map_prompt = f"""Community summary: "{summary}"

Question: "{query}"

On a scale of 0-10, how relevant is this community summary to answering the question?
Also extract the 1-2 most relevant points from the summary.

Return JSON: {{"score": 7, "points": ["point 1", "point 2"]}}
Only return valid JSON."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": map_prompt}],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```json\s*|```$", "", raw, flags=re.MULTILINE).strip()
        try:
            result = json.loads(raw)
            if result["score"] >= 5:
                map_results.append((result["score"], result["points"]))
        except Exception:
            pass

    # Sort by relevance score
    map_results.sort(key=lambda x: x[0], reverse=True)
    top_points = [p for _, points in map_results[:5] for p in points]

    if not top_points:
        return "No relevant information found across the document corpus."

    # REDUCE: synthesize final answer
    points_text = "\n".join(f"- {p}" for p in top_points)
    reduce_prompt = f"""Using these key points gathered from across the document corpus:

{points_text}

Answer the following question comprehensively:
{query}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": reduce_prompt}],
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────
# MAIN: BUILD INDEX + QUERY
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Building Knowledge Graph ===")
    G, entity_info = build_knowledge_graph(DOCUMENTS)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("\n=== Detecting Communities ===")
    partition = detect_communities(G)
    communities = group_by_community(partition)
    print(f"Found {len(communities)} communities: {communities}")

    print("\n=== Generating Community Summaries ===")
    community_summaries = {}
    for cid, nodes in communities.items():
        print(f"  Summarizing community {cid}: {nodes}")
        community_summaries[cid] = summarize_community(nodes, G, entity_info)
        print(f"  Summary: {community_summaries[cid][:100]}...")

    print("\n=== LOCAL SEARCH ===")
    q1 = "What does Alice do at TechCorp?"
    print(f"Q: {q1}")
    print(f"A: {local_search(q1, G, entity_info)}")

    print("\n=== GLOBAL SEARCH ===")
    q2 = "What are the main themes and competitive dynamics in these documents?"
    print(f"Q: {q2}")
    print(f"A: {global_search(q2, community_summaries)}")
```

### Expected Output

```
=== Building Knowledge Graph ===
  Extracting from doc 1/8...
  Extracting from doc 2/8...
  ...
Graph: 14 nodes, 18 edges

=== Detecting Communities ===
Found 3 communities: {
  0: ['Alice', 'Bob', 'Carol', 'Eve', 'David', 'TechCorp', 'Platform', 'Data', 'Security'],
  1: ['DataSoft', 'Frank'],
  2: ['Snowflake', 'dbt', 'Apache Kafka']
}

=== Generating Community Summaries ===
  Community 0 Summary: "TechCorp is led by CEO David and VP of Engineering Alice,
  who oversees three teams: Data (Bob), Platform (Carol), and Security (Eve).
  The company saw 40% YoY revenue growth in Q3..."

  Community 1 Summary: "DataSoft is a direct competitor to TechCorp in the analytics
  market. Led by CEO Frank, DataSoft raised $200M and is launching an AI-powered
  analytics product..."

=== LOCAL SEARCH ===
Q: What does Alice do at TechCorp?
A: Alice is the VP of Engineering at TechCorp. She manages three teams:
   the Platform team (led by Carol), the Data team (led by Bob), and the
   Security team (led by Eve). She also co-presented Q3 results with CEO David.

=== GLOBAL SEARCH ===
Q: What are the main themes and competitive dynamics in these documents?
A: The documents reveal two main themes:
   1. TechCorp's internal structure and growth — strong Q3 performance (40% YoY)
      driven by Bob's data team, with security vulnerabilities being addressed.
   2. Competitive pressure — DataSoft's $200M raise and new AI product directly
      threatens TechCorp's core analytics market.
```

---

## 9. Graph RAG with Neo4j + LangChain

For production systems, use a real graph database:

```python
# neo4j_graph_rag.py
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

# Connect to Neo4j
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="password"
)

# Use LangChain's LLM Graph Transformer to auto-extract entities + relations
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
transformer = LLMGraphTransformer(llm=llm)

# Your documents
docs = [
    Document(page_content="Alice manages the Data team at TechCorp. Bob reports to Alice."),
    Document(page_content="TechCorp's Data team built a pipeline using Snowflake and Kafka."),
]

# Extract graph nodes and relationships from documents
graph_docs = transformer.convert_to_graph_documents(docs)

# Store in Neo4j
graph.add_graph_documents(graph_docs, baseEntityLabel=True, include_source=True)

# Now query with Cypher
result = graph.query("""
MATCH (p:Person)-[r]->(o)
RETURN p.id as person, type(r) as relation, o.id as target
""")
print(result)
# [{'person': 'Alice', 'relation': 'MANAGES', 'target': 'Data team'}, ...]


# ── Graph RAG QA Chain ──────────────────────────────────────────────────────
from langchain.chains import GraphCypherQAChain

chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True,
)

response = chain.invoke("Who does Alice manage?")
print(response["result"])
# "Alice manages Bob and the Data team at TechCorp."
```

### Cypher query generated automatically by the chain:
```cypher
MATCH (p:Person {id: "Alice"})-[:MANAGES]->(x)
RETURN x.id
```

---

## 10. Comparison: Vanilla RAG vs Graph RAG

```
┌─────────────────────┬──────────────────────────┬──────────────────────────┐
│ Aspect              │ Vanilla RAG               │ Graph RAG                │
├─────────────────────┼──────────────────────────┼──────────────────────────┤
│ Storage             │ Vector index (flat)       │ Knowledge graph + vectors│
│ Retrieval unit      │ Text chunk                │ Entity + relationship    │
│ Context awareness   │ Within chunk only         │ Across entire graph      │
│ Indexing cost       │ Low (just embed)          │ High (LLM extraction)    │
│ Query latency       │ Fast (ANN search)         │ Moderate (graph lookup)  │
│ Global questions    │ Poor                      │ Excellent                │
│ Specific questions  │ Good                      │ Excellent                │
│ Multi-hop reasoning │ Poor                      │ Good (graph traversal)   │
│ Explainability      │ Low (opaque chunks)       │ High (visible graph)     │
│ Maintenance         │ Re-embed on update        │ Incremental graph update │
│ Best corpus size    │ Any                       │ Medium–large             │
└─────────────────────┴──────────────────────────┴──────────────────────────┘
```

---

## 11. When to Use Graph RAG

**Use Graph RAG when:**
- Questions span multiple documents ("What are the themes across all reports?")
- Relationships between entities matter ("How is Alice connected to the revenue growth?")
- You need multi-hop reasoning ("Who works for someone who reports to David?")
- Corpus has rich entity relationships (org charts, research papers, legal docs)
- You need explainability (you can show the graph path that led to the answer)

**Stick with Vanilla RAG when:**
- Questions are specific and local ("What did the invoice say about payment terms?")
- Documents are loosely related (diverse FAQs, unstructured support tickets)
- Indexing cost/latency is a hard constraint
- Corpus is small (< 50 docs) — graph overhead isn't justified

**Hybrid approach (best of both):**
```python
def smart_router(query: str) -> str:
    """Route to local, global, or vector search based on query type."""
    
    GLOBAL_KEYWORDS = ["main themes", "overall", "across all", "summarize", "trends", "compare"]
    ENTITY_KEYWORDS = ["who", "what does X do", "how is X related"]
    
    query_lower = query.lower()
    
    if any(kw in query_lower for kw in GLOBAL_KEYWORDS):
        return "global"   # → community summary map-reduce
    elif any(kw in query_lower for kw in ENTITY_KEYWORDS):
        return "local"    # → entity graph traversal
    else:
        return "vector"   # → standard vector similarity search
```

---

## 12. Best Practices

### Entity Extraction Prompt Quality
```python
# Be specific about what you want extracted
EXTRACTION_PROMPT = """Extract entities and relationships.

Focus on:
- Named people, organizations, products, locations, events
- Explicit relationships (not inferred ones)
- Use consistent naming (always "Alice" not "Alice Smith" and "A. Smith")

Do NOT extract:
- Generic concepts like "team" or "project" without a specific name
- Pronouns or vague references

Text: {text}
"""
```

### Graph Maintenance on Updates
```python
# When a new document arrives, don't rebuild from scratch
def incremental_update(new_doc: str, G: nx.Graph, entity_info: dict):
    data = extract_entities_and_relations(new_doc)
    
    for entity in data["entities"]:
        if entity["name"] not in G:
            G.add_node(entity["name"], **entity)
            entity_info[entity["name"]] = {...}
        else:
            # Merge: add new source text, update description if richer
            entity_info[entity["name"]]["source_texts"].append(new_doc)
    
    for rel in data["relationships"]:
        G.add_edge(rel["source"], rel["target"], relation=rel["relation"])
    
    # Re-run community detection only on affected subgraph
    # Re-summarize only affected communities
```

### Caching Community Summaries
```python
import hashlib, json

def get_community_summary(nodes: list, G, entity_info, cache: dict) -> str:
    # Cache key = sorted nodes hash
    key = hashlib.md5(json.dumps(sorted(nodes)).encode()).hexdigest()
    if key in cache:
        return cache[key]
    summary = summarize_community(nodes, G, entity_info)
    cache[key] = summary
    return summary
```

---

## Quick Reference

```bash
# Microsoft's official GraphRAG library
pip install graphrag

# Initialize a project
python -m graphrag.index --init --root ./my_project

# Run indexing pipeline on your documents
python -m graphrag.index --root ./my_project

# Run a global query
python -m graphrag.query --root ./my_project --method global "What are the main themes?"

# Run a local query  
python -m graphrag.query --root ./my_project --method local "Who is Alice?"
```

### Project structure with Microsoft GraphRAG:
```
my_project/
├── .env                  # GRAPHRAG_API_KEY, model settings
├── settings.yml          # Chunking, extraction, community config
├── input/
│   └── *.txt             # Your documents go here
└── output/
    └── artifacts/        # Auto-generated: graph, communities, summaries
```

---

## Further Reading

- [Microsoft GraphRAG Paper (2024)](https://arxiv.org/abs/2404.16130)
- [Microsoft GraphRAG GitHub](https://github.com/microsoft/graphrag)
- [LangChain Graph RAG docs](https://python.langchain.com/docs/use_cases/graph/)
- [Neo4j + LangChain integration](https://neo4j.com/labs/langchain/)
- [NetworkX Graph Algorithms](https://networkx.org/documentation/stable/)
