# Multi-Agent System Design
### Real-World Examples with LangChain + LangGraph

A multi-agent system is a network of specialized AI agents, each responsible for a narrow task, that coordinate to solve problems too complex for a single agent. This document walks through **three real-world scenarios** with full architecture, state design, and code.

---

## Table of Contents

1. [What is a Multi-Agent System?](#1-what-is-a-multi-agent-system)
2. [Core Concepts](#2-core-concepts)
3. [Real-World Example 1 — Customer Support Triage](#3-real-world-example-1--customer-support-triage)
4. [Real-World Example 2 — Autonomous Research Pipeline](#4-real-world-example-2--autonomous-research-pipeline)
5. [Real-World Example 3 — E-Commerce Order Fraud Detection](#5-real-world-example-3--e-commerce-order-fraud-detection)
6. [Agent Communication Patterns](#6-agent-communication-patterns)
7. [Memory Across Agents](#7-memory-across-agents)
8. [When to Use Multi-Agent vs Single Agent](#8-when-to-use-multi-agent-vs-single-agent)

---

## 1. What is a Multi-Agent System?

A single LLM agent does: `think → act → observe → repeat`.

A multi-agent system breaks work across **specialized agents** that:
- Each own a narrow domain (routing, searching, writing, validating)
- Pass results through a shared state
- Run sequentially, in parallel, or with feedback loops
- Can call tools, databases, or other agents

```
Single Agent:
  User → [One Agent does everything] → Response
         (bottleneck, context overload, error-prone)

Multi-Agent:
  User → [Router] → [Specialist A] → [Specialist B] → [Validator] → Response
                        ↑                                   │
                        └─────── retry loop if invalid ─────┘
```

---

## 2. Core Concepts

### 2.1 Agent Types

| Type | Role | Example |
|------|------|---------|
| **Supervisor** | Routes tasks to the right specialist | Reads user intent, delegates |
| **Specialist** | Executes a narrow task | Search agent, SQL agent, writer agent |
| **Critic / Validator** | Reviews output for quality | Checks hallucinations, schema validity |
| **Aggregator** | Merges results from parallel agents | Combines 3 search results into one answer |
| **Memory** | Manages context across turns | Stores facts from previous messages |

### 2.2 LangGraph Primitives

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# Shared state flows between all nodes
class State(TypedDict):
    input: str
    agent_outputs: Annotated[list, operator.add]  # list that accumulates
    final_answer: str
    next: str  # which agent runs next (used by supervisor)

# Each node is a Python function
def some_agent(state: State) -> State:
    # reads from state, does work, writes back
    return {**state, "some_field": "result"}

# Build graph
builder = StateGraph(State)
builder.add_node("agent_a", some_agent)
builder.add_edge("agent_a", END)
graph = builder.compile()
```

---

## 3. Real-World Example 1 — Customer Support Triage

### Scenario

An e-commerce company receives thousands of support tickets daily:
- "My order hasn't arrived"
- "I want a refund"
- "The product is broken"
- "How do I cancel my subscription?"

A single agent handling all of these gets confused and gives generic answers. Instead, we build a multi-agent system that **classifies → routes → resolves → drafts a reply**.

### 3.1 Architecture

```
                        [User Ticket]
                              │
                              ▼
                    ┌──────────────────┐
                    │  Classifier Agent │  → reads ticket, outputs category
                    └──────────────────┘
                              │
              ┌───────────────┼────────────────┐
              ▼               ▼                ▼
     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
     │  Order Agent  │ │ Refund Agent │ │  FAQ Agent   │
     │ (tracks order)│ │(checks policy│ │(searches KB) │
     └──────────────┘ └──────────────┘ └──────────────┘
              │               │                │
              └───────────────┴────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Reply Drafter   │  → writes final customer email
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  QA Validator    │  → checks tone, completeness
                    └──────────────────┘
                              │
                        [Send to Customer]
```

### 3.2 State

```python
from typing import TypedDict, Optional

class SupportState(TypedDict):
    ticket: str                    # raw customer message
    customer_id: str
    category: str                  # "order_issue" | "refund" | "faq" | "other"
    order_info: Optional[dict]     # pulled from DB by order agent
    refund_eligible: Optional[bool]
    faq_answer: Optional[str]
    draft_reply: str
    qa_approved: bool
    final_reply: str
```

### 3.3 Agent Implementations

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── Agent 1: Classifier ───────────────────────────────────────────
def classifier_agent(state: SupportState) -> SupportState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Classify the support ticket into exactly one category:
- order_issue   (tracking, delivery, missing items)
- refund        (return, money back, cancel order)
- faq           (how-to, policy questions, general info)
- other

Reply with ONLY the category name."""),
        ("human", "{ticket}"),
    ])
    chain = prompt | llm
    category = chain.invoke({"ticket": state["ticket"]}).content.strip().lower()
    return {**state, "category": category}


# ── Agent 2: Order Agent ─────────────────────────────────────────
def order_agent(state: SupportState) -> SupportState:
    # In reality: query your orders DB
    # Simulated here with a mock
    order_db = {
        "C001": {
            "order_id": "ORD-8821",
            "status": "In Transit",
            "estimated_delivery": "2024-03-10",
            "carrier": "FedEx",
            "tracking": "FX123456789",
        }
    }
    order_info = order_db.get(state["customer_id"], {"status": "Not Found"})
    return {**state, "order_info": order_info}


# ── Agent 3: Refund Agent ────────────────────────────────────────
def refund_agent(state: SupportState) -> SupportState:
    # Policy: eligible if order is within 30 days and not delivered yet
    order = state.get("order_info", {})
    eligible = order.get("status") in ["In Transit", "Processing"]
    return {**state, "refund_eligible": eligible}


# ── Agent 4: FAQ Agent ───────────────────────────────────────────
def faq_agent(state: SupportState) -> SupportState:
    # In reality: RAG lookup against a knowledge base
    # Simulated here
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful FAQ assistant. Answer based on company policy only."),
        ("human", "{ticket}"),
    ])
    chain = prompt | llm
    answer = chain.invoke({"ticket": state["ticket"]}).content
    return {**state, "faq_answer": answer}


# ── Agent 5: Reply Drafter ───────────────────────────────────────
def reply_drafter(state: SupportState) -> SupportState:
    # Build context based on what specialist agents found
    context = f"Category: {state['category']}\n"

    if state["order_info"]:
        o = state["order_info"]
        context += f"Order status: {o['status']}, ETA: {o.get('estimated_delivery')}, Tracking: {o.get('tracking')}\n"

    if state["refund_eligible"] is not None:
        context += f"Refund eligible: {state['refund_eligible']}\n"

    if state["faq_answer"]:
        context += f"FAQ answer: {state['faq_answer']}\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Draft a friendly, professional customer support reply.
Use the context provided. Be concise (3-5 sentences). Start with 'Hi,'"""),
        ("human", "Customer ticket: {ticket}\n\nContext: {context}"),
    ])
    chain = prompt | llm
    draft = chain.invoke({"ticket": state["ticket"], "context": context}).content
    return {**state, "draft_reply": draft}


# ── Agent 6: QA Validator ────────────────────────────────────────
def qa_validator(state: SupportState) -> SupportState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Review this customer support reply. Check:
1. Is it friendly and professional?
2. Does it address the customer's issue?
3. Is it free of any rude or inappropriate language?

Reply with exactly: APPROVED or REJECTED: <reason>"""),
        ("human", "{draft}"),
    ])
    chain = prompt | llm
    verdict = chain.invoke({"draft": state["draft_reply"]}).content.strip()

    approved = verdict.upper().startswith("APPROVED")
    final = state["draft_reply"] if approved else (
        "Hi, thank you for contacting support. A team member will follow up with you shortly."
    )
    return {**state, "qa_approved": approved, "final_reply": final}
```

### 3.4 Build the LangGraph

```python
from langgraph.graph import StateGraph, END

def route_by_category(state: SupportState) -> str:
    """After classifier, decide which specialist handles the ticket."""
    routes = {
        "order_issue": "order_agent",
        "refund":      "refund_agent",
        "faq":         "faq_agent",
    }
    return routes.get(state["category"], "reply_drafter")  # fallback to drafter

builder = StateGraph(SupportState)

builder.add_node("classifier",    classifier_agent)
builder.add_node("order_agent",   order_agent)
builder.add_node("refund_agent",  refund_agent)
builder.add_node("faq_agent",     faq_agent)
builder.add_node("reply_drafter", reply_drafter)
builder.add_node("qa_validator",  qa_validator)

builder.set_entry_point("classifier")

# After classification, route to specialist
builder.add_conditional_edges(
    "classifier",
    route_by_category,
    {
        "order_agent":   "order_agent",
        "refund_agent":  "refund_agent",
        "faq_agent":     "faq_agent",
        "reply_drafter": "reply_drafter",
    }
)

# All specialists flow into reply drafter
builder.add_edge("order_agent",  "reply_drafter")
builder.add_edge("refund_agent", "reply_drafter")
builder.add_edge("faq_agent",    "reply_drafter")
builder.add_edge("reply_drafter", "qa_validator")
builder.add_edge("qa_validator",  END)

support_graph = builder.compile()
```

### 3.5 Run It

```python
result = support_graph.invoke({
    "ticket": "Hi, I ordered 5 days ago and my package still hasn't arrived. Order #ORD-8821",
    "customer_id": "C001",
    "category": "",
    "order_info": None,
    "refund_eligible": None,
    "faq_answer": None,
    "draft_reply": "",
    "qa_approved": False,
    "final_reply": "",
})

print(result["final_reply"])
```

**Output:**

```
Hi, thank you for reaching out! I can see your order ORD-8821 is currently
in transit with FedEx (tracking: FX123456789) and is estimated to arrive
by March 10th. If you'd like to track it in real time, you can use the
tracking number on the FedEx website. Please let us know if you need
further assistance!
```

---

## 4. Real-World Example 2 — Autonomous Research Pipeline

### Scenario

A financial analyst asks: _"Give me a competitive analysis of Tesla vs Rivian — revenue trends, product launches, and market sentiment."_

This requires:
1. Searching for recent data from multiple sources
2. Summarizing long articles
3. Extracting specific financial figures
4. Writing a structured report

Too much for one agent — we split it.

### 4.1 Architecture

```
                     [Analyst Query]
                           │
                           ▼
               ┌────────────────────┐
               │  Planner Agent     │  → breaks query into 3 sub-tasks
               └────────────────────┘
                           │
          ┌────────────────┼──────────────────┐
          ▼                ▼                  ▼
  ┌──────────────┐ ┌──────────────┐  ┌──────────────┐
  │ Search Agent │ │Finance Agent │  │Sentiment Agent│
  │(web + news)  │ │(revenue data)│  │(social/news) │
  └──────────────┘ └──────────────┘  └──────────────┘
          │                │                  │
          └────────────────┴──────────────────┘
                           │
                           ▼
               ┌────────────────────┐
               │  Writer Agent      │  → structured markdown report
               └────────────────────┘
                           │
                           ▼
               ┌────────────────────┐
               │  Critic Agent      │  → checks for missing sections
               └────────────────────┘
                    │           │
               APPROVED      REJECTED
                    │           │
                    ▼           ▼
               [Report]    [Writer retries]
```

### 4.2 State

```python
from typing import TypedDict, Annotated
import operator

class ResearchState(TypedDict):
    query: str
    companies: list[str]                          # ["Tesla", "Rivian"]
    search_results: Annotated[list, operator.add]  # accumulates from parallel agents
    financial_data: dict
    sentiment_summary: str
    report_draft: str
    critique: str
    approved: bool
    final_report: str
    iteration: int                                 # retry counter
```

### 4.3 Parallel Agent Execution with Send API

LangGraph's `Send` API fans out work to multiple agents in parallel, then collects results.

```python
from langgraph.types import Send

# ── Planner ──────────────────────────────────────────────────────
def planner_agent(state: ResearchState) -> ResearchState:
    # Extract companies from the query
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract company names from this research query. Return as comma-separated list."),
        ("human", "{query}"),
    ])
    result = (prompt | llm).invoke({"query": state["query"]}).content
    companies = [c.strip() for c in result.split(",")]
    return {**state, "companies": companies, "iteration": 0}


# ── Search Agent (runs once per company) ─────────────────────────
def search_agent(state: dict) -> dict:
    """
    state here is a sub-state: {"company": "Tesla", "query": "..."}
    In real implementation, call a search API (Tavily, SerpAPI, etc.)
    """
    company = state["company"]
    query = state["query"]

    # Simulated search results
    mock_results = {
        "Tesla": [
            "Tesla Q4 2023 revenue: $25.2B, up 3% YoY. Cybertruck deliveries began Dec 2023.",
            "Tesla cut prices across Model 3 and Model Y lines in Jan 2024.",
        ],
        "Rivian": [
            "Rivian 2023 revenue: $4.4B. Delivered 50,122 vehicles in 2023.",
            "Rivian announced R2 SUV at $45,000 starting price in March 2024.",
        ],
    }

    return {"search_results": mock_results.get(company, [])}


# ── Fan-out: dispatch one search_agent per company ────────────────
def dispatch_search(state: ResearchState) -> list[Send]:
    return [
        Send("search_agent", {"company": c, "query": state["query"]})
        for c in state["companies"]
    ]


# ── Finance Agent ─────────────────────────────────────────────────
def finance_agent(state: ResearchState) -> ResearchState:
    # In reality: query a financial API (Yahoo Finance, Alpha Vantage)
    financial_data = {
        "Tesla":  {"revenue_2023": "97.7B", "gross_margin": "17.6%", "pe_ratio": 51},
        "Rivian": {"revenue_2023": "4.4B",  "gross_margin": "-44%",  "pe_ratio": None},
    }
    return {**state, "financial_data": financial_data}


# ── Sentiment Agent ───────────────────────────────────────────────
def sentiment_agent(state: ResearchState) -> ResearchState:
    combined_text = "\n".join(
        result for results in state["search_results"] for result in (results if isinstance(results, list) else [results])
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize market sentiment for each company in 2-3 sentences each."),
        ("human", "News and data:\n{text}"),
    ])
    summary = (prompt | llm).invoke({"text": combined_text[:3000]}).content
    return {**state, "sentiment_summary": summary}


# ── Writer Agent ──────────────────────────────────────────────────
def writer_agent(state: ResearchState) -> ResearchState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Write a structured competitive analysis report in markdown.
Include these sections: ## Executive Summary, ## Revenue Comparison,
## Product Launches, ## Market Sentiment, ## Conclusion.
Be specific with numbers."""),
        ("human", """Query: {query}

Financial Data: {financial}

News/Search Results: {search}

Sentiment: {sentiment}"""),
    ])

    report = (prompt | llm).invoke({
        "query":     state["query"],
        "financial": str(state["financial_data"]),
        "search":    "\n".join(str(r) for r in state["search_results"]),
        "sentiment": state["sentiment_summary"],
    }).content

    return {**state, "report_draft": report, "iteration": state["iteration"] + 1}


# ── Critic Agent ──────────────────────────────────────────────────
def critic_agent(state: ResearchState) -> ResearchState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Review this competitive analysis. Check:
1. Does it have all 5 required sections?
2. Are specific numbers cited?
3. Is the comparison balanced (both companies covered)?

Reply: APPROVED or NEEDS_REVISION: <specific feedback>"""),
        ("human", "{report}"),
    ])

    verdict = (prompt | llm).invoke({"report": state["report_draft"]}).content.strip()
    approved = verdict.upper().startswith("APPROVED")

    return {
        **state,
        "approved": approved,
        "critique": verdict,
        "final_report": state["report_draft"] if approved else "",
    }


# ── Retry logic ───────────────────────────────────────────────────
def should_retry(state: ResearchState) -> str:
    if state["approved"]:
        return "done"
    if state["iteration"] >= 3:          # max 3 retries
        return "done"
    return "retry"
```

### 4.4 Build Graph with Parallel Fan-Out + Retry Loop

```python
builder = StateGraph(ResearchState)

builder.add_node("planner",        planner_agent)
builder.add_node("search_agent",   search_agent)    # runs N times in parallel
builder.add_node("finance_agent",  finance_agent)
builder.add_node("sentiment_agent", sentiment_agent)
builder.add_node("writer",         writer_agent)
builder.add_node("critic",         critic_agent)

builder.set_entry_point("planner")

# Fan-out: planner → N parallel search agents
builder.add_conditional_edges("planner", dispatch_search, ["search_agent"])

# After all search agents finish, go to finance + sentiment (parallel)
builder.add_edge("search_agent", "finance_agent")
builder.add_edge("search_agent", "sentiment_agent")

# Both feed into writer
builder.add_edge("finance_agent",   "writer")
builder.add_edge("sentiment_agent", "writer")
builder.add_edge("writer",          "critic")

# Critic either approves or sends back to writer (retry loop)
builder.add_conditional_edges(
    "critic",
    should_retry,
    {
        "done":  END,
        "retry": "writer",   # writer reads state["critique"] and revises
    }
)

research_graph = builder.compile()
```

### 4.5 Run It

```python
result = research_graph.invoke({
    "query": "Competitive analysis of Tesla vs Rivian: revenue trends, product launches, market sentiment",
    "companies": [],
    "search_results": [],
    "financial_data": {},
    "sentiment_summary": "",
    "report_draft": "",
    "critique": "",
    "approved": False,
    "final_report": "",
    "iteration": 0,
})

print(result["final_report"])
```

**Output (excerpt):**

```markdown
## Executive Summary
Tesla and Rivian represent two distinct stages of EV maturity. Tesla,
with $97.7B in 2023 revenue, is a profitable incumbent facing margin
pressure. Rivian, at $4.4B, is a high-growth startup still burning cash
at -44% gross margin.

## Revenue Comparison
| Metric        | Tesla   | Rivian  |
|---------------|---------|---------|
| 2023 Revenue  | $97.7B  | $4.4B   |
| Gross Margin  | 17.6%   | -44%    |
| Vehicles Sold | 1.81M   | 50,122  |

## Product Launches
- Tesla: Cybertruck deliveries began December 2023. Price cuts on Model
  3/Y in January 2024 signal competitive pressure.
- Rivian: R2 SUV announced at $45,000 in March 2024, targeting the mass
  market below its current R1T/R1S lineup.
...
```

---

## 5. Real-World Example 3 — E-Commerce Order Fraud Detection

### Scenario

When a new order is placed, run it through a fraud detection pipeline:
1. **Profile agent** — looks up buyer history
2. **Rules agent** — checks hard rules (country blocklist, velocity checks)
3. **ML agent** — calls a fraud scoring model
4. **Decision agent** — weighs all signals, makes final decision
5. **Action agent** — blocks or approves order, triggers alerts

### 5.1 Architecture

```
              [New Order Event]
                     │
                     ▼
          ┌─────────────────────┐
          │   Profile Agent     │ → fetch buyer history from DB
          └─────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
  ┌──────────────┐     ┌──────────────┐
  │ Rules Agent  │     │   ML Agent   │
  │(blocklist,   │     │(fraud score  │
  │ velocity)    │     │ 0.0 – 1.0)   │
  └──────────────┘     └──────────────┘
          │                     │
          └──────────┬──────────┘
                     ▼
          ┌─────────────────────┐
          │   Decision Agent    │ → APPROVE / REVIEW / BLOCK
          └─────────────────────┘
                     │
          ┌──────────┼──────────┐
     APPROVE       REVIEW      BLOCK
          │          │           │
          ▼          ▼           ▼
    [Process]  [Flag for   [Cancel +
               human]      Alert team]
```

### 5.2 State

```python
class FraudState(TypedDict):
    order: dict                   # raw order payload
    buyer_profile: dict           # history, age, location
    rules_flags: list[str]        # list of triggered rule names
    rules_passed: bool
    fraud_score: float            # 0.0 = clean, 1.0 = fraud
    decision: str                 # "APPROVE" | "REVIEW" | "BLOCK"
    reason: str
    action_taken: str
```

### 5.3 Agent Implementations

```python
# ── Profile Agent ─────────────────────────────────────────────────
def profile_agent(state: FraudState) -> FraudState:
    order = state["order"]

    # Simulate DB lookup
    profiles = {
        "user_101": {
            "account_age_days": 730,
            "total_orders": 45,
            "chargebacks": 0,
            "country": "US",
            "avg_order_value": 120.0,
        },
        "user_202": {
            "account_age_days": 2,   # brand new account
            "total_orders": 0,
            "chargebacks": 0,
            "country": "US",
            "avg_order_value": 0.0,
        },
    }

    profile = profiles.get(order["user_id"], {"account_age_days": 0, "total_orders": 0})
    return {**state, "buyer_profile": profile}


# ── Rules Agent ───────────────────────────────────────────────────
BLOCKED_COUNTRIES = {"NG", "KP", "IR"}
HIGH_RISK_THRESHOLD = 500.0
VELOCITY_LIMIT = 3  # max orders per hour

def rules_agent(state: FraudState) -> FraudState:
    order   = state["order"]
    profile = state["buyer_profile"]
    flags   = []

    # Rule 1: blocked country
    if order.get("shipping_country") in BLOCKED_COUNTRIES:
        flags.append("BLOCKED_COUNTRY")

    # Rule 2: order value far above user's average
    avg = profile.get("avg_order_value", 0)
    if avg > 0 and order["amount"] > avg * 5:
        flags.append("UNUSUAL_ORDER_VALUE")

    # Rule 3: new account + high value order
    if profile.get("account_age_days", 0) < 7 and order["amount"] > HIGH_RISK_THRESHOLD:
        flags.append("NEW_ACCOUNT_HIGH_VALUE")

    # Rule 4: high chargeback history
    if profile.get("chargebacks", 0) >= 2:
        flags.append("HIGH_CHARGEBACK_HISTORY")

    return {**state, "rules_flags": flags, "rules_passed": len(flags) == 0}


# ── ML Agent ──────────────────────────────────────────────────────
def ml_agent(state: FraudState) -> FraudState:
    """
    In reality: call your fraud ML model (sklearn, XGBoost, SageMaker endpoint).
    Features: account_age, order_value, country_risk, velocity, etc.
    """
    profile = state["buyer_profile"]
    order   = state["order"]
    flags   = state["rules_flags"]

    # Simulated heuristic scoring (replace with real model)
    score = 0.05  # baseline

    if profile.get("account_age_days", 999) < 7:
        score += 0.40
    if "UNUSUAL_ORDER_VALUE" in flags:
        score += 0.25
    if "NEW_ACCOUNT_HIGH_VALUE" in flags:
        score += 0.20
    if profile.get("chargebacks", 0) >= 2:
        score += 0.30

    score = min(score, 1.0)
    return {**state, "fraud_score": round(score, 2)}


# ── Decision Agent ────────────────────────────────────────────────
def decision_agent(state: FraudState) -> FraudState:
    score = state["fraud_score"]
    flags = state["rules_flags"]

    # Hard block: critical rules triggered
    if "BLOCKED_COUNTRY" in flags or "HIGH_CHARGEBACK_HISTORY" in flags:
        decision = "BLOCK"
        reason = f"Hard rule triggered: {', '.join(flags)}"

    # High ML score
    elif score >= 0.70:
        decision = "BLOCK"
        reason = f"Fraud score too high: {score}"

    # Medium risk: send to human review
    elif score >= 0.40 or flags:
        decision = "REVIEW"
        reason = f"Score {score}, flags: {flags}"

    # Low risk: approve
    else:
        decision = "APPROVE"
        reason = f"Score {score}, no flags"

    return {**state, "decision": decision, "reason": reason}


# ── Action Agent ──────────────────────────────────────────────────
def action_agent(state: FraudState) -> FraudState:
    decision = state["decision"]
    order_id = state["order"]["order_id"]

    if decision == "APPROVE":
        action = f"Order {order_id} approved and sent to fulfillment."

    elif decision == "REVIEW":
        action = f"Order {order_id} flagged. Added to human review queue. Customer notified of delay."

    elif decision == "BLOCK":
        action = f"Order {order_id} BLOCKED. Payment voided. Security team alerted. Customer emailed."

    print(f"[ACTION] {action}")
    return {**state, "action_taken": action}
```

### 5.4 Build Graph

```python
def route_decision(state: FraudState) -> str:
    return "action"  # all paths lead to action agent (it handles branching internally)

builder = StateGraph(FraudState)

builder.add_node("profile_agent",  profile_agent)
builder.add_node("rules_agent",    rules_agent)
builder.add_node("ml_agent",       ml_agent)
builder.add_node("decision_agent", decision_agent)
builder.add_node("action_agent",   action_agent)

builder.set_entry_point("profile_agent")

# Profile → run rules and ML in parallel
builder.add_edge("profile_agent", "rules_agent")
builder.add_edge("profile_agent", "ml_agent")

# Both feed decision
builder.add_edge("rules_agent", "decision_agent")
builder.add_edge("ml_agent",    "decision_agent")

builder.add_edge("decision_agent", "action_agent")
builder.add_edge("action_agent",   END)

fraud_graph = builder.compile()
```

### 5.5 Run It

```python
# Suspicious order: brand new account, high value
result = fraud_graph.invoke({
    "order": {
        "order_id":        "ORD-9934",
        "user_id":         "user_202",
        "amount":          850.00,
        "shipping_country": "US",
        "items":           ["iPhone 15 Pro"],
    },
    "buyer_profile": {},
    "rules_flags":   [],
    "rules_passed":  True,
    "fraud_score":   0.0,
    "decision":      "",
    "reason":        "",
    "action_taken":  "",
})

print(f"Decision: {result['decision']}")
print(f"Reason:   {result['reason']}")
print(f"Action:   {result['action_taken']}")
```

**Output:**

```
[ACTION] Order ORD-9934 BLOCKED. Payment voided. Security team alerted. Customer emailed.

Decision: BLOCK
Reason:   Fraud score too high: 0.65, flags: ['NEW_ACCOUNT_HIGH_VALUE']
Action:   Order ORD-9934 BLOCKED. Payment voided. Security team alerted. Customer emailed.
```

---

## 6. Agent Communication Patterns

### 6.1 Sequential (Pipeline)

Each agent hands off to the next. Simple, predictable.

```
A → B → C → D → END
```

Used in: Support triage (classifier → specialist → drafter → QA)

### 6.2 Parallel Fan-Out + Join

Multiple agents run simultaneously, results are merged.

```
      ┌── Agent A ──┐
A ────┤              ├──── Join ──── B
      └── Agent C ──┘
```

Used in: Research pipeline (search + finance + sentiment run in parallel)

```python
# LangGraph parallel edges — just add multiple edges from same source
builder.add_edge("planner", "search_agent")
builder.add_edge("planner", "finance_agent")
# LangGraph will run both before proceeding to the next joined node
```

### 6.3 Supervisor / Router

One agent reads intent and dispatches to the right specialist.

```
User → Supervisor → [Agent A | Agent B | Agent C]
                         │           │
                         └─── join ──┘
                               │
                           Supervisor
```

```python
from langgraph.graph import MessagesState
from langgraph.prebuilt import create_react_agent

# LangGraph has a built-in supervisor pattern
def supervisor_node(state):
    # Decide who should act next
    result = supervisor_chain.invoke(state)
    return {"next": result["next"]}  # "agent_a", "agent_b", or "FINISH"
```

### 6.4 Feedback / Retry Loop

A critic sends work back to a generator if quality is insufficient.

```
Writer → Critic → APPROVED → END
             │
          REJECTED
             │
           Writer (revised with critique in state)
```

Used in: Research pipeline (critic ↔ writer retry loop)

---

## 7. Memory Across Agents

By default, state is reset per invocation. For multi-turn conversations, use LangGraph's built-in checkpointing:

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()  # in-memory; use SqliteSaver / PostgresSaver for prod

graph_with_memory = builder.compile(checkpointer=checkpointer)

# Thread ID groups messages into the same conversation session
config = {"configurable": {"thread_id": "session_user_101"}}

# Turn 1
result1 = graph_with_memory.invoke(
    {"ticket": "Where is my order?", "customer_id": "C001", ...},
    config=config
)

# Turn 2 — agent remembers previous context from thread_id
result2 = graph_with_memory.invoke(
    {"ticket": "Can I get a refund for it?", "customer_id": "C001", ...},
    config=config
)
```

### Persistent Memory with PostgreSQL

```python
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost:5432/ragdb"
)
graph = builder.compile(checkpointer=checkpointer)
```

---

## 8. When to Use Multi-Agent vs Single Agent

| Signal | Use Single Agent | Use Multi-Agent |
|--------|-----------------|-----------------|
| Task complexity | Simple, one topic | Multiple domains, steps |
| Context size | Fits in one prompt | Would overflow context window |
| Specialization needed | No | Yes (search vs write vs validate) |
| Parallelism possible | No | Yes (search 3 sources at once) |
| Retry / quality loops | No | Yes (critic → writer loop) |
| Maintainability | Not a concern | Easier to swap one agent |

### Rule of thumb

> Start with a single agent. Move to multi-agent when:
> 1. The prompt exceeds ~4,000 tokens of instructions + context
> 2. You need two distinct skills (e.g., search AND structured writing)
> 3. You want to run steps in parallel
> 4. You need a validation/retry loop

---

## Summary

| Example | Pattern | Agents | Key LangGraph Feature |
|---------|---------|--------|----------------------|
| Customer Support | Sequential + routing | Classifier, Specialists, Drafter, QA | `add_conditional_edges` |
| Research Pipeline | Parallel fan-out + retry | Planner, Search×N, Finance, Sentiment, Writer, Critic | `Send` API, retry loop |
| Fraud Detection | Parallel + hard rules | Profile, Rules, ML, Decision, Action | Parallel edges, state merging |
