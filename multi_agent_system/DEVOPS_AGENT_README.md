# DevOps Lifecycle Automation — ReAct Multi-Agent System

> Autonomous incident detection, diagnosis, and remediation across GCP, Hetzner bare-metal, and Tailscale VPN mesh — powered by LangChain + LangGraph + GPT-4o.

---

## Table of Contents

1. [What This System Does](#1-what-this-system-does)
2. [Infrastructure Overview](#2-infrastructure-overview)
3. [Why Multi-Agent, Not One Agent](#3-why-multi-agent-not-one-agent)
4. [System Architecture](#4-system-architecture)
5. [The ReAct Loop — How Each Agent Thinks](#5-the-react-loop--how-each-agent-thinks)
6. [Agent Breakdown](#6-agent-breakdown)
7. [Solving the 15+ Tools Problem](#7-solving-the-15-tools-problem)
8. [Context Store — Redis State Management](#8-context-store--redis-state-management)
9. [LangGraph Workflow — The Wiring](#9-langgraph-workflow--the-wiring)
10. [Incident Lifecycle — End to End](#10-incident-lifecycle--end-to-end)
11. [Example Incident Walkthrough](#11-example-incident-walkthrough)
12. [Tech Stack](#12-tech-stack)
13. [Interview Q&A](#13-interview-qa)

---

## 1. What This System Does

When something breaks in production — a GCP instance goes OOM, a Nomad job crashes on a Hetzner GPU node, a Tailscale peer goes offline — a human engineer normally has to:

1. Get paged
2. Open Grafana / Prometheus
3. SSH into the node
4. Diagnose the issue
5. Run Terraform / Ansible / Nomad commands
6. Verify the fix

This system does all of that **autonomously**, in under 2 minutes, without waking anyone up.

---

## 2. Infrastructure Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      TAILSCALE VPN MESH                         │
│                                                                 │
│  ┌─────────────────────┐      ┌──────────────────────────────┐  │
│  │     GCP (Primary)   │      │   Hetzner (Bare Metal GPU)   │  │
│  │                     │      │                              │  │
│  │  ● GKE Cluster      │      │  ● hetzner-gpu-01 (A100)     │  │
│  │  ● Compute Engine   │◄────►│  ● hetzner-gpu-02 (A100)     │  │
│  │  ● Cloud SQL        │      │  ● hetzner-gpu-03 (H100)     │  │
│  │  ● Cloud Storage    │      │                              │  │
│  │  ● Terraform state  │      │  All nodes run:              │  │
│  │    (GCS backend)    │      │  - Nomad client              │  │
│  └─────────────────────┘      │  - Tailscale client          │  │
│                               │  - Ansible-managed config    │  │
│                               └──────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Monitoring Stack                                        │   │
│  │  Prometheus (scrapes all nodes) + Grafana (dashboards)   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**VM Bootstrap Flow (every new node):**
```
Instance starts → Ansible startup script runs →
  installs: Nomad client, Tailscale client, Prometheus node exporter,
            Docker, CUDA drivers (Hetzner GPU nodes) →
  registers with: Nomad server, Tailscale network →
  starts: all services + health checks
```

---

## 3. Why Multi-Agent, Not One Agent

### The Problem with a Single Agent

If you give one LLM agent all 26 tools across 7 domains:

- It picks wrong tools (asks Terraform to restart a Nomad job)
- It loses context mid-diagnosis in long chains
- Prompt becomes 8,000+ tokens of tool descriptions alone
- No parallelism — it checks one thing at a time
- Impossible to test, debug, or swap individual capabilities

### The Solution: Domain-Scoped Specialists

```
                       ┌─────────────────┐
                       │   SUPERVISOR    │
                       │  (GPT-4o brain) │
                       │  routes alerts  │
                       └────────┬────────┘
                                │
          ┌─────────────────────┼──────────────────────┐
          │                     │                      │
          ▼                     ▼                      ▼
  ┌──────────────┐     ┌──────────────┐      ┌──────────────┐
  │  Monitoring  │     │Infrastructure│      │  Machine Ops │
  │    Agent     │     │    Agent     │      │    Agent     │
  │              │     │              │      │              │
  │ - Prometheus │     │ - Terraform  │      │ - SSH exec   │
  │ - Grafana    │     │   GCP scale  │      │ - Reboot GCP │
  │ - Diagnose   │     │   Hetzner    │      │ - Reboot     │
  │   root cause │     │   scale      │      │   Hetzner    │
  └──────────────┘     └──────────────┘      └──────────────┘
          │                     │                      │
          ▼                     ▼                      ▼
  ┌──────────────┐     ┌──────────────┐      ┌──────────────┐
  │  Nomad Agent │     │ Ansible Agent│      │Tailscale Agent│
  │              │     │              │      │              │
  │ - Job mgmt   │     │ - Playbooks  │      │ - VPN status │
  │ - Scale alloc│     │ - Node setup │      │ - Node auth  │
  │ - Drain node │     │ - Config mgmt│      │ - Peer check │
  └──────────────┘     └──────────────┘      └──────────────┘
```

Each agent gets **only its domain's tools** — 3-5 tools max — so the LLM is never confused.

---

## 4. System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         ALERT INGESTION                              │
│                                                                      │
│  Prometheus Alert → AlertManager → Webhook → FastAPI endpoint        │
│  (or manual trigger for demo)                                        │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        LANGGRAPH WORKFLOW                            │
│                                                                      │
│  ┌──────────┐    ┌────────────┐    ┌────────────┐    ┌───────────┐  │
│  │Supervisor│───►│ Monitoring │───►│ Supervisor │───►│  Nomad /  │  │
│  │  Node    │    │   Node     │    │  (re-route)│    │  Infra /  │  │
│  └──────────┘    └────────────┘    └────────────┘    │  Ansible  │  │
│       ▲                                              └───────────┘  │
│       │                    Redis State                               │
│       └──────────────── (shared context) ──────────────────────────┘│
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       TOOL EXECUTION LAYER                           │
│                                                                      │
│  Each agent calls mocked tools that simulate real API calls:         │
│  GCP SDK / Terraform CLI / Nomad API / Ansible Runner /              │
│  Tailscale API / SSH / Prometheus HTTP API                           │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 5. The ReAct Loop — How Each Agent Thinks

ReAct = **Re**asoning + **Act**ing. The agent alternates between thinking and tool use until it reaches a conclusion.

```
ALERT RECEIVED: "hetzner-gpu-02 CPU at 97% for 10 minutes"
                        │
                        ▼
┌───────────────────────────────────────────────────────────┐
│  THINK: CPU is 97% sustained. Could be:                   │
│         1. Runaway training job                           │
│         2. Kernel bug / zombie process                    │
│         3. OOM causing swap thrash                        │
│         Let me check node metrics first.                  │
└───────────────────────────────────────────────────────────┘
                        │
                        ▼
  ACT: query_node_metrics(node="hetzner-gpu-02", window="15m")
                        │
                        ▼
┌───────────────────────────────────────────────────────────┐
│  OBSERVE: {                                               │
│    "cpu": 97.4, "memory": 91.2, "gpu_util": 99.8,        │
│    "top_process": "python3 train.py",                     │
│    "pid": 18432, "gpu_mem_used": "79GB / 80GB"            │
│  }                                                        │
└───────────────────────────────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────┐
│  THINK: GPU memory is nearly full (79/80GB). The training │
│         job is the culprit. Check if it's a Nomad job     │
│         so I can restart it gracefully vs kill the PID.   │
└───────────────────────────────────────────────────────────┘
                        │
                        ▼
  ACT: get_nomad_job_status(node="hetzner-gpu-02")
                        │
                        ▼
┌───────────────────────────────────────────────────────────┐
│  OBSERVE: {                                               │
│    "job_id": "llm-training-batch-47",                     │
│    "status": "running", "restarts": 8, "oom_kills": 3     │
│  }                                                        │
└───────────────────────────────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────┐
│  THINK: 3 OOM kills and 8 restarts = memory leak in the   │
│         job. Nomad is restarting it but it keeps leaking. │
│         I should: 1) stop the job, 2) check if infra      │
│         needs to scale another node to absorb workload,   │
│         3) flag for Ansible to check config limits.       │
└───────────────────────────────────────────────────────────┘
                        │
                        ▼
  ACT: restart_nomad_job(job_id="llm-training-batch-47", drain_first=True)
                        │
                        ▼
┌───────────────────────────────────────────────────────────┐
│  OBSERVE: {"success": true, "new_alloc": "alloc-a92f3b",  │
│            "node_reassigned": "hetzner-gpu-03"}           │
└───────────────────────────────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────┐
│  THINK: Job moved to gpu-03. CPU on gpu-02 should drop.   │
│         Let me verify before marking resolved.            │
└───────────────────────────────────────────────────────────┘
                        │
                        ▼
  ACT: query_node_metrics(node="hetzner-gpu-02", window="2m")
                        │
                        ▼
  OBSERVE: {"cpu": 12.3, "memory": 18.4, "gpu_util": 0.0}
                        │
                        ▼
  THINK: Metrics normalized. Incident resolved. Writing summary.
                        │
                        ▼
  FINAL ANSWER → passed to Supervisor → incident marked RESOLVED
```

The agent **never asks a human** — it reasons through follow-ups itself, picks the right tool for each sub-question, and loops until confident.

---

## 6. Agent Breakdown

### 6.1 Supervisor Agent

The orchestrator. Does not use domain tools. Uses GPT-4o to:
- Parse the incoming alert and determine incident type
- Route to the appropriate starting agent
- After each agent reports back, decide: escalate, delegate elsewhere, or close

```python
# Routing decision examples
alert_type → starting_agent:
  "high_cpu"           → monitoring_agent  (diagnose first)
  "node_unreachable"   → tailscale_agent   (check VPN first)
  "terraform_drift"    → infrastructure_agent
  "nomad_job_failing"  → nomad_agent
  "disk_full"          → machine_ops_agent
  "service_degraded"   → monitoring_agent  (diagnose first)
```

**Supervisor system prompt (key excerpt):**
```
You are a senior DevOps SRE. You receive incident alerts and coordinate
a team of specialized agents. Your job is to:
1. Identify the most likely root cause from the alert
2. Choose which agent investigates first
3. After each agent reports, decide next action
4. Mark incident resolved only when metrics confirm recovery

Available agents: monitoring, infrastructure, nomad, ansible, tailscale, machine_ops
Output as JSON: {"next_agent": "<name>", "reasoning": "<why>", "resolved": false}
```

---

### 6.2 Monitoring Agent

**Domain:** Prometheus + Grafana
**Tools available (4):**

| Tool | What it does |
|------|-------------|
| `query_prometheus` | Run PromQL — get CPU, memory, latency, error rates |
| `get_active_alerts` | List all firing Alertmanager alerts |
| `check_service_health` | HTTP health check against a service endpoint |
| `get_grafana_dashboard` | Pull current panel values from a Grafana dashboard |

**Responsibility:** Root cause diagnosis. Answers "what is broken and why?" before any action is taken.

---

### 6.3 Infrastructure Agent

**Domain:** Terraform (GCP + Hetzner)
**Tools available (4):**

| Tool | What it does |
|------|-------------|
| `terraform_plan_scale` | Preview scaling change (instance count, machine type) |
| `terraform_apply_scale` | Apply the scale change |
| `get_terraform_state` | Read current resource state |
| `list_gcp_instances` | List running GCP Compute Engine instances + status |

**Responsibility:** Scale up/down GCP VMs, add/remove Hetzner nodes via Terraform. All changes go through `plan → apply` to prevent accidents.

---

### 6.4 Nomad Agent

**Domain:** HashiCorp Nomad (workload orchestration)
**Tools available (5):**

| Tool | What it does |
|------|-------------|
| `list_nomad_jobs` | List all running jobs and their health |
| `get_job_status` | Detailed status — restarts, OOM kills, allocations |
| `restart_nomad_job` | Gracefully restart a job (drain → stop → start) |
| `scale_nomad_job` | Change job's replica count |
| `drain_nomad_node` | Mark a node as ineligible, migrate its jobs |

**Responsibility:** Manages workloads running on the Nomad cluster (both GCP and Hetzner nodes are Nomad clients).

---

### 6.5 Ansible Agent

**Domain:** Ansible playbook runner
**Tools available (3):**

| Tool | What it does |
|------|-------------|
| `run_playbook` | Execute a named playbook against a target host/group |
| `get_playbook_status` | Check if a running playbook succeeded or failed |
| `list_available_playbooks` | Show all playbooks and their purpose |

**Key playbooks in the system:**
```
bootstrap_node.yml     → Install Nomad client, Tailscale, exporters
restart_services.yml   → Restart all managed services on a node
rotate_credentials.yml → Rotate Tailscale auth key, service tokens
fix_disk_usage.yml     → Clean up logs, Docker images, tmp files
update_limits.yml      → Adjust OS-level memory/file descriptor limits
```

---

### 6.6 Tailscale Agent

**Domain:** Tailscale VPN mesh
**Tools available (3):**

| Tool | What it does |
|------|-------------|
| `get_tailscale_status` | List all peers, their IPs, last seen, online status |
| `check_node_connectivity` | Ping a specific node through the mesh |
| `authorize_new_node` | Approve a new node joining the tailnet |

**Why this matters:** GCP and Hetzner nodes communicate exclusively over Tailscale. If a node drops from the mesh, everything trying to reach it fails — this agent detects and remediates VPN-layer issues before assuming the node is dead.

---

### 6.7 Machine Ops Agent

**Domain:** Low-level instance operations
**Tools available (4):**

| Tool | What it does |
|------|-------------|
| `restart_gcp_instance` | Call GCP Compute API to reset a VM |
| `reboot_hetzner_server` | Call Hetzner API to reset a bare-metal server |
| `ssh_exec` | Run a shell command on a node via SSH (through Tailscale IP) |
| `get_instance_status` | Check if an instance is RUNNING / STOPPED / REPAIRING |

**This is the last resort agent.** Called only when all software-level fixes fail and the node itself needs to be recycled.

---

## 7. Solving the 15+ Tools Problem

### The Problem

With 26 tools across 7 domains, if you bind them all to one agent:

```
LLM context = system_prompt + all_tool_descriptions + conversation
            = ~2,000 + ~6,000 + ~3,000 = 11,000 tokens just to start

Result: LLM picks wrong tools, hallucinates tool names,
        loses focus across long chains
```

### Solution: Tool Registry with Domain Scoping

```python
# tools/registry.py

TOOL_REGISTRY = {
    "monitoring": [
        query_prometheus,
        get_active_alerts,
        check_service_health,
        get_grafana_dashboard,
    ],
    "infrastructure": [
        terraform_plan_scale,
        terraform_apply_scale,
        get_terraform_state,
        list_gcp_instances,
    ],
    "nomad": [
        list_nomad_jobs,
        get_job_status,
        restart_nomad_job,
        scale_nomad_job,
        drain_nomad_node,
    ],
    "ansible": [
        run_playbook,
        get_playbook_status,
        list_available_playbooks,
    ],
    "tailscale": [
        get_tailscale_status,
        check_node_connectivity,
        authorize_new_node,
    ],
    "machine_ops": [
        restart_gcp_instance,
        reboot_hetzner_server,
        ssh_exec,
        get_instance_status,
    ],
}

def get_tools_for_agent(domain: str) -> list:
    return TOOL_REGISTRY[domain]
```

**Each agent is initialized with only its domain tools:**

```python
# agents/nomad_agent.py
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from tools.registry import get_tools_for_agent

llm = ChatOpenAI(model="gpt-4o", temperature=0)

nomad_agent = create_react_agent(
    model=llm,
    tools=get_tools_for_agent("nomad"),   # only 5 tools, not 26
    prompt=NOMAD_SYSTEM_PROMPT,
)
```

**Result:**
| Agent | Tools in context | Token overhead |
|-------|-----------------|----------------|
| Single agent (naive) | 26 tools | ~6,000 tokens |
| Monitoring agent | 4 tools | ~800 tokens |
| Nomad agent | 5 tools | ~1,000 tokens |
| Each specialist | 3-5 tools | ~700-1,200 tokens |

**80% reduction in tool-related token usage. LLM accuracy improves significantly.**

---

## 8. Context Store — Redis State Management

### Why Redis (Not Just LangGraph State)

LangGraph's built-in state lives in memory and is scoped to one graph run. When multiple agents run sequentially — monitoring diagnoses, then nomad fixes, then supervisor verifies — they need to share what they learned **without re-reading from scratch**.

Redis acts as the **incident brain**: every agent reads from it and writes back.

### Incident Schema

```python
# state/incident.py

class IncidentState(BaseModel):
    incident_id: str               # "inc-20260406-hetzner-gpu-02-001"
    status: str                    # "investigating" | "remediating" | "verifying" | "resolved"
    severity: str                  # "critical" | "high" | "medium"

    # Alert context
    alert_name: str                # "HighCPUUsage"
    affected_node: str             # "hetzner-gpu-02"
    affected_service: str | None   # "llm-training-batch-47"
    cloud_provider: str            # "hetzner" | "gcp"

    # Diagnosis (written by monitoring agent, read by all others)
    root_cause: str | None         # "OOM leak in Nomad job"
    diagnosis_confidence: float    # 0.0 - 1.0

    # Actions
    actions_taken: list[str]       # append-only log of what was done
    current_agent: str             # which agent is active right now

    # Per-agent scratchpads (each agent writes its reasoning here)
    agent_notes: dict[str, str]    # {"monitoring": "...", "nomad": "..."}

    # Resolution
    resolved: bool
    resolved_at: str | None
    resolution_summary: str | None

    # Metadata
    created_at: str
    updated_at: str
```

### Redis Operations

```python
# state/redis_store.py

class RedisIncidentStore:

    def create_incident(self, alert: dict) -> IncidentState:
        """Called when alert fires. Creates incident in Redis."""
        incident = IncidentState(
            incident_id=f"inc-{datetime.now().strftime('%Y%m%d%H%M%S')}-{alert['node']}",
            alert_name=alert["alertname"],
            affected_node=alert["node"],
            status="investigating",
            ...
        )
        self.redis.setex(
            f"incident:{incident.incident_id}",
            ttl=86400,            # 24 hour TTL
            value=incident.json()
        )
        return incident

    def update_incident(self, incident_id: str, updates: dict):
        """Any agent can write back findings."""
        incident = self.get_incident(incident_id)
        for key, value in updates.items():
            setattr(incident, key, value)
        incident.updated_at = datetime.utcnow().isoformat()
        self.redis.set(f"incident:{incident_id}", incident.json())

    def append_action(self, incident_id: str, action: str):
        """Immutable audit log of every action taken."""
        self.redis.rpush(f"incident:{incident_id}:actions", action)

    def get_incident(self, incident_id: str) -> IncidentState:
        data = self.redis.get(f"incident:{incident_id}")
        return IncidentState.parse_raw(data)
```

### How Agents Use Shared Context

```
Monitoring Agent:
  reads → incident (to understand what alert fired)
  writes → root_cause, diagnosis_confidence, agent_notes["monitoring"]

Nomad Agent:
  reads → root_cause (already diagnosed, no need to re-check metrics)
  writes → actions_taken (appends "restarted job X"), agent_notes["nomad"]

Supervisor:
  reads → entire incident state after each agent
  writes → status, current_agent, resolved
```

No agent needs to re-ask "what was the original alert?" or "what did monitoring find?" — it's all in Redis.

---

## 9. LangGraph Workflow — The Wiring

### State (flows between graph nodes)

```python
# graph/state.py
from typing import TypedDict, Annotated
import operator

class DevOpsState(TypedDict):
    # Incident identification
    incident_id: str
    alert: dict

    # Message history (accumulates across all agent turns)
    messages: Annotated[list, operator.add]

    # Routing
    next_agent: str         # supervisor sets this after each step
    resolved: bool

    # Diagnosis from monitoring agent
    root_cause: str
    affected_component: str  # "node" | "job" | "network" | "infra"
```

### Graph Definition

```python
# graph/workflow.py
from langgraph.graph import StateGraph, END

builder = StateGraph(DevOpsState)

# Register all nodes
builder.add_node("supervisor",      supervisor_node)
builder.add_node("monitoring",      monitoring_node)
builder.add_node("infrastructure",  infrastructure_node)
builder.add_node("nomad",           nomad_node)
builder.add_node("ansible",         ansible_node)
builder.add_node("tailscale",       tailscale_node)
builder.add_node("machine_ops",     machine_ops_node)

# Entry: every incident starts with supervisor
builder.set_entry_point("supervisor")

# Supervisor routes to any agent
builder.add_conditional_edges(
    "supervisor",
    route_to_agent,           # reads state["next_agent"]
    {
        "monitoring":     "monitoring",
        "infrastructure": "infrastructure",
        "nomad":          "nomad",
        "ansible":        "ansible",
        "tailscale":      "tailscale",
        "machine_ops":    "machine_ops",
        "END":            END,
    }
)

# Every agent reports back to supervisor
for agent in ["monitoring", "infrastructure", "nomad", "ansible", "tailscale", "machine_ops"]:
    builder.add_edge(agent, "supervisor")

graph = builder.compile()
```

### Routing Logic

```python
def route_to_agent(state: DevOpsState) -> str:
    if state["resolved"]:
        return "END"
    return state["next_agent"]   # supervisor wrote this before handing off
```

### What the Graph Looks Like at Runtime

```
Incident: "hetzner-gpu-02 node unreachable"

START
  │
  ▼
[Supervisor]  →  "Node unreachable. Check VPN mesh first."
  │                next_agent = "tailscale"
  ▼
[Tailscale Agent]  →  "Node dropped from tailnet 8 min ago. Last seen IP: 10.x.x.x"
  │                    Writes finding to Redis
  ▼
[Supervisor]  →  "VPN issue confirmed. Try to reconnect or check node status."
  │                next_agent = "machine_ops"
  ▼
[Machine Ops Agent]  →  "GCP instance is RUNNING but unresponsive. Rebooting."
  │                      ssh_exec failed → restart_gcp_instance → instance back up
  ▼
[Supervisor]  →  "Instance rebooted. Check if it rejoined tailnet."
  │                next_agent = "tailscale"
  ▼
[Tailscale Agent]  →  "Node is back online. All peers can reach it."
  │
  ▼
[Supervisor]  →  "Node restored. Verify workloads resumed."
  │                next_agent = "nomad"
  ▼
[Nomad Agent]  →  "All jobs on gpu-02 running normally."
  │
  ▼
[Supervisor]  →  resolved = True
  │
  ▼
END  →  Incident closed in Redis. Summary written.
```

---

## 10. Incident Lifecycle — End to End

```
Phase 1: DETECTION
  Prometheus fires alert → AlertManager webhook → FastAPI receives alert JSON
  → RedisStore.create_incident() → LangGraph graph.invoke() starts

Phase 2: TRIAGE (Supervisor)
  GPT-4o reads alert, determines incident type
  Sets next_agent based on alert category
  Writes initial assessment to Redis

Phase 3: DIAGNOSIS (Monitoring Agent)
  Runs ReAct loop with Prometheus/Grafana tools
  Identifies root cause with confidence score
  Writes: root_cause, diagnosis_confidence, affected_component to Redis

Phase 4: REMEDIATION (Domain Agent)
  Reads root_cause from Redis (no re-diagnosis needed)
  Runs ReAct loop with domain tools
  Executes fix: restart job / scale infra / run playbook / reboot node
  Appends each action to Redis audit log

Phase 5: VERIFICATION (Monitoring Agent again)
  Supervisor routes back to monitoring
  Queries metrics to confirm recovery
  If metrics normalized → returns to supervisor with "verified"
  If not → supervisor routes back to domain agent for retry

Phase 6: RESOLUTION (Supervisor)
  Sets resolved=True in Redis
  Writes resolution_summary
  Updates incident status → "resolved"
  (Optional: posts to Slack / PagerDuty via webhook)
```

---

## 11. Example Incident Walkthrough

**Alert:** `InstanceDown` — `gcp-compute-worker-03` has been unreachable for 5 minutes

**Automatic flow:**

```
[Supervisor]
  THINK: Instance down alert for GCP node. Could be crash, network, or OOM.
         Start with monitoring to understand what happened before.

[Monitoring Agent]
  ACT: get_active_alerts()
  OBSERVE: 3 alerts — InstanceDown, HighMemoryUsage (was firing 10min ago), NomadClientDisconnected
  THINK: Memory was high before it went down — likely OOM killed the node or caused kernel panic.
  ACT: query_prometheus("up{instance='gcp-worker-03'}", window="30m")
  OBSERVE: up was 1 until 14:32 then dropped to 0. MemoryUsage was 98.7% at 14:30.
  CONCLUSION: Node OOM → kernel panic → instance crashed.
  → Writes to Redis: root_cause="OOM crash", confidence=0.92

[Supervisor]
  THINK: OOM crash confirmed. Machine ops should restart the instance.
         Then ansible should check and fix memory limits so it doesn't happen again.
  → next_agent = "machine_ops"

[Machine Ops Agent]
  ACT: get_instance_status(instance="gcp-compute-worker-03")
  OBSERVE: status="TERMINATED"
  ACT: restart_gcp_instance(instance="gcp-compute-worker-03")
  OBSERVE: {"status": "RUNNING", "started_at": "14:39:02"}
  ACT: ssh_exec(host="gcp-worker-03", cmd="systemctl is-active nomad")
  OBSERVE: "active"
  → Writes to Redis: actions=["restarted gcp-compute-worker-03", "verified nomad active"]

[Supervisor]
  THINK: Instance up, Nomad running. Now prevent recurrence.
  → next_agent = "ansible"

[Ansible Agent]
  ACT: list_available_playbooks()
  OBSERVE: [..., "update_limits.yml — adjust memory/file descriptor limits"]
  ACT: run_playbook(playbook="update_limits.yml", target="gcp-compute-worker-03")
  OBSERVE: {"status": "success", "changed": ["vm.overcommit_memory", "cgroup memory limits"]}
  → Writes to Redis: actions=["ran update_limits.yml on gcp-worker-03"]

[Supervisor]
  → next_agent = "monitoring" (verify)

[Monitoring Agent]
  ACT: check_service_health(node="gcp-compute-worker-03")
  OBSERVE: {"healthy": true, "latency_ms": 12, "nomad_jobs": 4, "cpu": 23.1, "mem": 41.2}
  CONCLUSION: Node fully recovered. Memory at 41% (well within limits).

[Supervisor]
  resolved = True
  resolution_summary = "GCP worker-03 crashed due to OOM at 14:32.
                        Restarted via GCP API. Ansible applied memory
                        limit fixes to prevent recurrence. All 4 Nomad
                        jobs resumed. Incident duration: 9 minutes."
```

**Total time: ~90 seconds. Zero human intervention.**

---

## 12. Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| LLM | GPT-4o | Reasoning, tool selection, ReAct loop |
| Agent Framework | LangChain | Tool binding, ReAct agent creation |
| Orchestration | LangGraph | Multi-agent state graph, routing |
| State | Redis | Shared incident context across agents |
| Monitoring | Prometheus + Grafana | Metrics source (mocked) |
| Infra | Terraform | GCP + Hetzner provisioning (mocked) |
| Workloads | HashiCorp Nomad | Job scheduling across all nodes |
| Config Mgmt | Ansible | Node bootstrap and remediation |
| Networking | Tailscale | VPN mesh between GCP and Hetzner |
| API Layer | FastAPI | Receives Prometheus webhooks |
| Language | Python 3.11+ | |

### Key Dependencies

```txt
langchain>=0.2.0
langgraph>=0.1.0
langchain-openai>=0.1.0
openai>=1.0.0
redis>=5.0.0
pydantic>=2.0.0
fastapi>=0.110.0
uvicorn>=0.29.0
python-dotenv>=1.0.0
```

---

## 13. Interview Q&A

**Q: Why LangGraph instead of just chaining LangChain agents?**

LangGraph gives you a **stateful graph** — state persists between nodes, you get conditional branching (supervisor can route to any of 6 agents based on runtime conditions), and built-in support for loops (monitoring → fix → monitoring to verify). LangChain chains are linear. Real incidents are not.

---

**Q: How do you prevent the agents from going in infinite loops?**

Two mechanisms:
1. The incident object in Redis has a `max_remediation_attempts` counter. If it hits 3, supervisor escalates to human (PagerDuty) and stops.
2. LangGraph supports `recursion_limit` on the graph — hard cap on total node executions.

---

**Q: What if two alerts fire at the same time for the same node?**

Each alert creates its own `incident_id` in Redis. LangGraph runs are independent Python async tasks. Redis keys are namespaced by incident ID so there's no collision. The supervisor is stateless — it reads the alert, not a shared singleton.

---

**Q: Why not just use Kubernetes instead of Nomad?**

Nomad is lighter and works natively across heterogeneous environments — GCP VMs and Hetzner bare metal in the same cluster without the overhead of kubelets. It handles GPU workloads (training jobs) alongside regular services without needing complex node selectors or device plugins.

---

**Q: How does Tailscale factor into the agent logic?**

All SSH commands and internal API calls route through Tailscale IPs, not public IPs. If a node disappears from the mesh, every other communication to it fails. The Tailscale agent specifically checks VPN-layer reachability before the Machine Ops agent wastes time trying SSH — it prevents false "node is dead" conclusions when it's actually just a VPN key rotation issue.

---

**Q: How do you manage context window limits across a long incident?**

Three strategies:
1. **Tool domain scoping** — each agent only sees 3-5 tools, not 26
2. **Redis as memory** — agents don't carry full history, they query Redis for what they need
3. **Message summarization** — after 10 ReAct steps, the agent's message history is summarized before continuing. LangGraph supports this via custom reducers on the messages field.

---

**Q: Why Redis over just using LangGraph's built-in checkpointer?**

LangGraph checkpointers persist graph state for **replay and resumability** — great for debugging. Redis stores **business state** — the incident, the audit log, cross-agent notes — which needs to be queryable by external systems (dashboards, alerting, post-mortems). They serve different purposes and we use both.

---

**Q: How would you productionize this beyond mocks?**

1. Replace mock tools with real SDK calls (GCP Python SDK, Hetzner API, Nomad HTTP API, Ansible Runner)
2. Add human-in-the-loop for destructive ops (`restart_gcp_instance`, `terraform_apply`) using LangGraph's interrupt/approve pattern
3. Replace in-memory Redis with Redis Cluster for HA
4. Add Langfuse or LangSmith for LLM call tracing and cost tracking
5. Rate limit OpenAI calls with exponential backoff
6. Add incident runbook storage — agents can look up known fixes for known alerts before calling GPT-4o

---

## 14. How Tools Are Built — Deep Dive

This section explains exactly how to create a LangChain tool, from a simple function all the way to a production-grade tool with error handling and mocking. Then covers how MCP fits in.

---

### 14.1 What Is a Tool?

A **tool** is just a Python function that:
1. Has a clear name and description (the LLM reads this to decide when to call it)
2. Takes typed inputs (the LLM generates these as JSON)
3. Returns a string (the LLM reads this as the "observation")

The LLM never sees your Python code — it only sees the **name**, **description**, and **input schema**.

```
LLM sees:                          Python runs:
─────────────────────────────      ────────────────────────────────
Tool: restart_nomad_job            def restart_nomad_job(job_id, drain_first):
Desc: Restart a running Nomad          # actual HTTP call to Nomad API
      job. Drains node first           response = requests.post(...)
      if drain_first=True.             return json.dumps(response)
Input: {
  "job_id": "string",
  "drain_first": "boolean"
}
```

---

### 14.2 Three Ways to Define a Tool in LangChain

#### Method 1: `@tool` decorator (simplest)

```python
from langchain_core.tools import tool

@tool
def get_nomad_job_status(job_id: str) -> str:
    """
    Get the current status of a Nomad job including restart count,
    OOM kills, and allocation details.

    Args:
        job_id: The Nomad job identifier (e.g. 'llm-training-batch-47')
    """
    # MOCKED — replace with real Nomad HTTP API call
    mock_data = {
        "llm-training-batch-47": {
            "status": "running",
            "restarts": 8,
            "oom_kills": 3,
            "allocations": [
                {"id": "alloc-a1b2", "node": "hetzner-gpu-02", "status": "running"}
            ]
        }
    }
    result = mock_data.get(job_id, {"error": f"Job {job_id} not found"})
    return json.dumps(result, indent=2)
```

LangChain automatically:
- Uses the function name as the tool name
- Uses the docstring as the tool description
- Infers the input schema from type hints

---

#### Method 2: `StructuredTool` with Pydantic schema (recommended for complex inputs)

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

# Step 1: Define the input schema
class RestartNomadJobInput(BaseModel):
    job_id: str = Field(description="The Nomad job ID to restart")
    drain_first: bool = Field(
        default=True,
        description="If True, drain the node before restarting to migrate jobs gracefully"
    )
    timeout_seconds: int = Field(
        default=120,
        description="Max seconds to wait for drain to complete"
    )

# Step 2: Write the actual function
def _restart_nomad_job(job_id: str, drain_first: bool, timeout_seconds: int) -> str:
    """Core logic — separated from schema for testability."""

    steps = []

    if drain_first:
        # MOCKED: In real impl → POST /v1/node/{node_id}/drain
        steps.append(f"Draining node hosting {job_id}...")
        steps.append(f"Drain complete after 12s (timeout was {timeout_seconds}s)")

    # MOCKED: In real impl → POST /v1/job/{job_id}/revert or DELETE + re-register
    steps.append(f"Stopping job {job_id}...")
    steps.append(f"Starting job {job_id}...")

    result = {
        "success": True,
        "job_id": job_id,
        "new_allocation": "alloc-x92f3b",
        "node_reassigned": "hetzner-gpu-03",   # Nomad picked a healthy node
        "steps": steps,
    }
    return json.dumps(result, indent=2)


# Step 3: Wrap as StructuredTool
restart_nomad_job = StructuredTool.from_function(
    func=_restart_nomad_job,
    name="restart_nomad_job",
    description="""Gracefully restart a Nomad job.
    Optionally drains the node first to migrate allocations before stopping.
    Use this when a job is crash-looping or has OOM kill history.
    Do NOT use for jobs that are healthy — only for failed/degraded jobs.""",
    args_schema=RestartNomadJobInput,
    return_direct=False,   # False = LLM continues reasoning after seeing output
)
```

The extra description detail is important — `"Do NOT use for jobs that are healthy"` teaches the LLM when **not** to call it, reducing hallucinated tool calls.

---

#### Method 3: `BaseTool` class (most control, for stateful tools)

Use this when your tool needs to hold state (e.g., a connection pool, a session token, a mock data store that persists between calls).

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

class SSHExecInput(BaseModel):
    host: str = Field(description="Tailscale IP or hostname of the target node")
    command: str = Field(description="Shell command to execute")
    timeout: int = Field(default=30, description="Command timeout in seconds")

class SSHExecTool(BaseTool):
    name: str = "ssh_exec"
    description: str = """Execute a shell command on a remote node via SSH over the Tailscale VPN.
    Returns stdout, stderr, and exit code.
    Use for: checking process status, reading logs, killing PIDs, checking disk usage.
    Do NOT use for: actions covered by other tools (use restart_nomad_job, not 'systemctl restart nomad')."""

    args_schema: Type[BaseModel] = SSHExecInput

    # Stateful: mock SSH session store
    mock_responses: dict = {
        "systemctl is-active nomad": "active",
        "df -h /": "Filesystem  Size  Used  Avail  Use%\n/dev/sda1   200G   45G   155G   23%",
        "ps aux | grep python": "root  18432  97.4  91.2  python3 train.py",
        "free -m": "total: 80384  used: 73481  free: 6903",
    }

    def _run(self, host: str, command: str, timeout: int = 30) -> str:
        """Synchronous execution — called by the agent."""

        # MOCKED: real impl uses paramiko or asyncssh
        # In prod: ssh_client = paramiko.SSHClient()
        #          ssh_client.connect(host, key_filename=TAILSCALE_KEY)
        #          stdin, stdout, stderr = ssh_client.exec_command(command)

        output = self.mock_responses.get(command, f"[mock] Command '{command}' executed on {host}")
        result = {
            "host": host,
            "command": command,
            "stdout": output,
            "stderr": "",
            "exit_code": 0,
        }
        return json.dumps(result, indent=2)

    async def _arun(self, host: str, command: str, timeout: int = 30) -> str:
        """Async version — used when agent runs in async context."""
        return self._run(host, command, timeout)
```

---

### 14.3 Building the Full Nomad Tool Suite

Here's the complete set of 5 Nomad tools showing different patterns:

```python
# tools/nomad_tools.py

import json
from langchain_core.tools import tool, StructuredTool
from pydantic import BaseModel, Field


# ── Tool 1: Simple query (no side effects) ──────────────────────────

@tool
def list_nomad_jobs() -> str:
    """List all Nomad jobs across the cluster with their current status.
    Use this first when investigating workload issues to get an overview."""
    mock_jobs = [
        {"id": "llm-training-batch-47", "status": "running",  "node": "hetzner-gpu-02", "restarts": 8},
        {"id": "api-server",            "status": "running",  "node": "gcp-worker-01",  "restarts": 0},
        {"id": "data-pipeline",         "status": "dead",     "node": "gcp-worker-02",  "restarts": 15},
        {"id": "embedding-service",     "status": "running",  "node": "hetzner-gpu-03", "restarts": 1},
    ]
    return json.dumps(mock_jobs, indent=2)


# ── Tool 2: Parameterized query ──────────────────────────────────────

@tool
def get_nomad_job_status(job_id: str) -> str:
    """Get detailed status of a specific Nomad job including restart count,
    OOM kill history, allocation node, and last failure reason.

    Args:
        job_id: Exact job ID as shown in list_nomad_jobs
    """
    mock_detail = {
        "llm-training-batch-47": {
            "status": "running",
            "restarts": 8,
            "oom_kills": 3,
            "last_failure": "OOM: memory limit exceeded (80GB / 80GB)",
            "allocations": [{"id": "alloc-a1b2", "node": "hetzner-gpu-02", "cpu_pct": 97.4, "mem_pct": 91.2}],
            "uptime_seconds": 847,
        },
        "data-pipeline": {
            "status": "dead",
            "restarts": 15,
            "oom_kills": 0,
            "last_failure": "exit code 1: connection refused to upstream DB",
            "allocations": [],
            "uptime_seconds": 0,
        }
    }
    result = mock_detail.get(job_id, {"error": f"Job '{job_id}' not found in cluster"})
    return json.dumps(result, indent=2)


# ── Tool 3: Stateful action with schema ─────────────────────────────

class RestartJobInput(BaseModel):
    job_id: str = Field(description="Nomad job ID to restart")
    drain_first: bool = Field(default=True, description="Drain the node before restarting")

def _restart_job(job_id: str, drain_first: bool) -> str:
    steps = []
    if drain_first:
        steps.append(f"[drain] Marked node ineligible, migrating {job_id} allocations")
    steps.append(f"[stop]  Job {job_id} stopped")
    steps.append(f"[start] Job {job_id} scheduled on hetzner-gpu-03")
    steps.append(f"[check] Job {job_id} healthy after 8s")
    return json.dumps({"success": True, "steps": steps, "new_node": "hetzner-gpu-03"}, indent=2)

restart_nomad_job = StructuredTool.from_function(
    func=_restart_job,
    name="restart_nomad_job",
    description="""Restart a Nomad job. Optionally drains the current node first.
    Use when: job is OOM-looping, crash-looping, or stuck in pending state.
    Do NOT use on: healthy jobs, or jobs that are intentionally stopped.""",
    args_schema=RestartJobInput,
)


# ── Tool 4: Scale action ─────────────────────────────────────────────

class ScaleJobInput(BaseModel):
    job_id: str = Field(description="Job ID to scale")
    count: int = Field(description="New replica count (1-10)")

def _scale_job(job_id: str, count: int) -> str:
    return json.dumps({
        "success": True,
        "job_id": job_id,
        "previous_count": 1,
        "new_count": count,
        "message": f"Scaling {job_id} from 1 to {count} replicas across available GPU nodes"
    }, indent=2)

scale_nomad_job = StructuredTool.from_function(
    func=_scale_job,
    name="scale_nomad_job",
    description="Scale a Nomad job to a specified replica count. Use when load requires more instances.",
    args_schema=ScaleJobInput,
)


# ── Tool 5: Node-level operation ─────────────────────────────────────

@tool
def drain_nomad_node(node_id: str) -> str:
    """Mark a Nomad node as ineligible and migrate all its jobs to other nodes.
    Use before rebooting a node to prevent job loss.
    The node will no longer receive new allocations until undrained.

    Args:
        node_id: The Nomad node ID (e.g. 'hetzner-gpu-02')
    """
    return json.dumps({
        "success": True,
        "node": node_id,
        "jobs_migrated": ["llm-training-batch-47"],
        "migrated_to": ["hetzner-gpu-03"],
        "node_status": "ineligible (drained)",
        "message": f"Node {node_id} safely drained. Safe to reboot."
    }, indent=2)
```

---

### 14.4 Building the Terraform Tools

Terraform tools are different — they always follow a **plan → apply** pattern to prevent accidents. The plan tool has no side effects; the apply tool does.

```python
# tools/terraform_tools.py

import json
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class TerraformPlanInput(BaseModel):
    resource_type: str = Field(
        description="What to scale: 'gcp_instance' or 'hetzner_server'"
    )
    action: str = Field(
        description="'scale_up' to add instances, 'scale_down' to remove"
    )
    count: int = Field(
        description="Number of instances to add or remove"
    )
    instance_type: str = Field(
        default="n2-standard-8",
        description="GCP machine type or Hetzner server type (e.g. 'AX102' for GPU)"
    )

def _terraform_plan(resource_type: str, action: str, count: int, instance_type: str) -> str:
    """
    REAL IMPL would run:
        subprocess.run(["terraform", "plan", "-var", f"count={count}", "-out=tfplan"])
    and parse the output.

    Always call this BEFORE terraform_apply_scale. Never skip the plan step.
    """
    direction = "+" if action == "scale_up" else "-"
    mock_plan = {
        "plan_id": "tfplan-20260406-142301",
        "changes": [
            {
                "action": "create" if action == "scale_up" else "destroy",
                "resource": f"{resource_type}.worker_{count}",
                "type": instance_type,
            }
        ],
        "summary": f"Plan: {count} to add, 0 to change, 0 to destroy" if action == "scale_up"
                   else f"Plan: 0 to add, 0 to change, {count} to destroy",
        "estimated_cost_delta": f"+${count * 0.38:.2f}/hr" if action == "scale_up" else f"-${count * 0.38:.2f}/hr",
        "safe_to_apply": True,
    }
    return json.dumps(mock_plan, indent=2)

terraform_plan_scale = StructuredTool.from_function(
    func=_terraform_plan,
    name="terraform_plan_scale",
    description="""Preview a Terraform scaling change WITHOUT applying it.
    Always call this before terraform_apply_scale.
    Returns what will be created/destroyed and estimated cost impact.
    Use when: CPU/memory is consistently high and more capacity is needed.""",
    args_schema=TerraformPlanInput,
)


class TerraformApplyInput(BaseModel):
    plan_id: str = Field(description="The plan_id returned by terraform_plan_scale")

def _terraform_apply(plan_id: str) -> str:
    """
    REAL IMPL would run:
        subprocess.run(["terraform", "apply", plan_id])
    Terraform reads the saved plan file and executes it.
    """
    return json.dumps({
        "success": True,
        "plan_id": plan_id,
        "resources_created": ["gcp_instance.worker_4"],
        "instance_ips": {
            "gcp_instance.worker_4": {
                "internal_ip": "10.128.0.14",
                "tailscale_ip": "100.64.0.14",
            }
        },
        "nomad_registered": True,    # Ansible startup script ran on boot
        "tailscale_registered": True,
        "elapsed_seconds": 47,
    }, indent=2)

terraform_apply_scale = StructuredTool.from_function(
    func=_terraform_apply,
    name="terraform_apply_scale",
    description="""Apply a previously generated Terraform plan to actually provision or destroy infrastructure.
    REQUIRES a plan_id from terraform_plan_scale — never call without planning first.
    This is irreversible in the short term. Only call when plan output looks correct.""",
    args_schema=TerraformApplyInput,
)


@tool
def get_terraform_state() -> str:
    """Read current Terraform state — list all provisioned resources, their IPs,
    and current counts. Use to understand what infrastructure currently exists
    before deciding to scale."""
    mock_state = {
        "gcp_instances": [
            {"name": "gcp-worker-01", "type": "n2-standard-8", "status": "RUNNING", "zone": "us-central1-a"},
            {"name": "gcp-worker-02", "type": "n2-standard-8", "status": "RUNNING", "zone": "us-central1-b"},
            {"name": "gcp-worker-03", "type": "n2-standard-8", "status": "TERMINATED", "zone": "us-central1-c"},
        ],
        "hetzner_servers": [
            {"name": "hetzner-gpu-01", "type": "AX102", "status": "running", "location": "fsn1"},
            {"name": "hetzner-gpu-02", "type": "AX102", "status": "running", "location": "fsn1"},
            {"name": "hetzner-gpu-03", "type": "GX2",   "status": "running", "location": "hel1"},
        ],
        "total_vcpus": 96,
        "total_gpu_nodes": 3,
    }
    return json.dumps(mock_state, indent=2)
```

**Key principle:** The tool description teaches the LLM the **order of operations**. Notice `terraform_apply_scale` says "REQUIRES a plan_id from terraform_plan_scale" — the LLM will naturally call plan first because you told it to.

---

### 14.5 Building Machine Ops Tools

Machine Ops tools are the most dangerous — they reboot production nodes. The descriptions are deliberately cautious.

```python
# tools/machine_ops_tools.py

import json
from langchain_core.tools import tool, StructuredTool, BaseTool
from pydantic import BaseModel, Field
from typing import Type


@tool
def get_instance_status(instance_name: str) -> str:
    """Check the current power/run state of a GCP or Hetzner instance.
    Always call this BEFORE restart_gcp_instance or reboot_hetzner_server.
    Returns: RUNNING, TERMINATED, STAGING, STOPPING, or REPAIRING.

    Args:
        instance_name: Full instance name (e.g. 'gcp-worker-03' or 'hetzner-gpu-02')
    """
    mock_states = {
        "gcp-worker-03":   {"status": "TERMINATED", "provider": "gcp",     "uptime_hours": 0},
        "gcp-worker-01":   {"status": "RUNNING",    "provider": "gcp",     "uptime_hours": 312},
        "hetzner-gpu-02":  {"status": "running",    "provider": "hetzner", "uptime_hours": 2},
        "hetzner-gpu-01":  {"status": "running",    "provider": "hetzner", "uptime_hours": 891},
    }
    result = mock_states.get(instance_name, {"error": f"Instance '{instance_name}' not found"})
    return json.dumps(result, indent=2)


class RestartGCPInput(BaseModel):
    instance_name: str = Field(description="GCP Compute Engine instance name")
    zone: str = Field(default="us-central1-a", description="GCP zone the instance is in")
    wait_for_ready: bool = Field(default=True, description="Block until instance is RUNNING again")

def _restart_gcp_instance(instance_name: str, zone: str, wait_for_ready: bool) -> str:
    """
    REAL IMPL:
        from google.cloud import compute_v1
        client = compute_v1.InstancesClient()
        client.reset(project=GCP_PROJECT, zone=zone, instance=instance_name)

    This is a HARD RESET — equivalent to pulling the power cord.
    Triggers the Ansible startup script on next boot.
    """
    return json.dumps({
        "success": True,
        "instance": instance_name,
        "zone": zone,
        "action": "HARD_RESET",
        "status_after": "RUNNING" if wait_for_ready else "STAGING",
        "boot_time_seconds": 43,
        "tailscale_reconnected": True,
        "nomad_registered": True,
        "warning": "All in-memory state on this instance was lost"
    }, indent=2)

restart_gcp_instance = StructuredTool.from_function(
    func=_restart_gcp_instance,
    name="restart_gcp_instance",
    description="""Hard reset a GCP Compute Engine instance (equivalent to power cycle).
    Use as LAST RESORT when: instance is TERMINATED and won't start, or completely
    unresponsive to SSH and all service-level fixes have failed.
    WARNING: Causes ~45s downtime. All in-memory state is lost.
    PREFER: restart_nomad_job or ssh_exec before calling this.""",
    args_schema=RestartGCPInput,
)


# ── SSH Exec using BaseTool (stateful mock store) ────────────────────

class SSHExecInput(BaseModel):
    host: str = Field(description="Tailscale IP or hostname (e.g. 'hetzner-gpu-02' resolves via tailscale)")
    command: str = Field(description="Shell command to run on the remote node")

class SSHExecTool(BaseTool):
    name: str = "ssh_exec"
    description: str = """Run a shell command on any node in the Tailscale VPN mesh via SSH.
    Use for: reading logs, checking process status, killing PIDs, checking disk/memory.
    Returns: stdout, stderr, exit_code.
    Do NOT use for actions that have dedicated tools (use restart_nomad_job, not 'systemctl restart nomad')."""

    args_schema: Type[BaseModel] = SSHExecInput

    # Mock SSH response store — simulates what a real node would return
    _mock_responses: dict = {
        "systemctl is-active nomad": "active",
        "systemctl is-active tailscaled": "active",
        "df -h /": "/dev/sda1  200G  189G  11G  95%  /",   # disk almost full!
        "free -m": "Mem:  80384  73900  6484  0  2847  5200",
        "ps aux | grep python": "root 18432 97.4 91.2 python3 train.py --batch 512",
        "journalctl -u nomad --since '5 minutes ago'": "[ERROR] OOM kill received for alloc alloc-a1b2",
        "nvidia-smi --query-gpu=memory.used,memory.total --format=csv": "79123 MiB, 80936 MiB",
    }

    def _run(self, host: str, command: str) -> str:
        output = self._mock_responses.get(command, f"[mock-ssh {host}] $ {command}\nok")
        return json.dumps({
            "host": host,
            "command": command,
            "stdout": output,
            "stderr": "",
            "exit_code": 0,
        }, indent=2)

    async def _arun(self, host: str, command: str) -> str:
        return self._run(host, command)

ssh_exec = SSHExecTool()
```

---

### 14.6 The Tool Description Is Everything

The LLM **cannot see your code**. It makes decisions purely based on the tool description. This is where most people go wrong.

| Bad description | Why it's bad |
|----------------|-------------|
| `"Restart a job"` | LLM doesn't know when to use it vs ssh_exec or reboot |
| `"Get metrics"` | LLM doesn't know which metrics or what format to expect |
| `"Apply terraform"` | Doesn't warn LLM to plan first |

| Good description | Why it works |
|----------------|-------------|
| `"Restart a Nomad job. Use when: OOM-looping. Do NOT use: on healthy jobs"` | Teaches when AND when not to call |
| `"Always call terraform_plan_scale FIRST before calling this"` | Enforces ordering |
| `"Last resort — prefer ssh_exec before calling this"` | Creates a preference hierarchy |

**Rule of thumb:** Write the description as if you're writing instructions for a junior engineer who can't ask questions.

---

### 14.7 MCP — Model Context Protocol

MCP is an **open standard** (created by Anthropic) that lets you expose tools as a **server** that any LLM application can connect to — not just LangChain.

#### Without MCP (what we've built so far)
```
LangChain Agent ──── calls ────► Python functions (in same process)
```

#### With MCP
```
LangChain Agent ──── MCP client ────► MCP Server (separate process/service)
                                              │
                                              ├── Nomad tools
                                              ├── Terraform tools
                                              └── Machine ops tools
```

The MCP server exposes your tools over a standard protocol (JSON-RPC over stdio or HTTP). Any MCP-compatible client (Claude Desktop, LangChain, custom agents) can consume them.

---

#### Building an MCP Server for Nomad Tools

```python
# mcp_servers/nomad_mcp_server.py

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
import json

# Initialize the MCP server
app = Server("nomad-devops-server")


# ── Register tools via @app.list_tools ────────────────────────────────

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """Tell the MCP client what tools this server offers."""
    return [
        types.Tool(
            name="list_nomad_jobs",
            description="List all Nomad jobs across the cluster with status",
            inputSchema={
                "type": "object",
                "properties": {},   # no inputs needed
                "required": [],
            },
        ),
        types.Tool(
            name="get_nomad_job_status",
            description="Get detailed status of a specific Nomad job including OOM kills and restarts",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The Nomad job ID"
                    }
                },
                "required": ["job_id"],
            },
        ),
        types.Tool(
            name="restart_nomad_job",
            description="Gracefully restart a Nomad job, optionally draining the node first",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Job ID to restart"},
                    "drain_first": {"type": "boolean", "description": "Drain node before restarting", "default": True},
                },
                "required": ["job_id"],
            },
        ),
    ]


# ── Handle tool calls ─────────────────────────────────────────────────

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Execute the tool and return the result."""

    if name == "list_nomad_jobs":
        result = [
            {"id": "llm-training-batch-47", "status": "running", "restarts": 8},
            {"id": "api-server",            "status": "running", "restarts": 0},
        ]
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "get_nomad_job_status":
        job_id = arguments["job_id"]
        result = {"job_id": job_id, "status": "running", "oom_kills": 3, "restarts": 8}
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "restart_nomad_job":
        job_id = arguments["job_id"]
        drain  = arguments.get("drain_first", True)
        result = {"success": True, "job_id": job_id, "drained": drain, "new_node": "hetzner-gpu-03"}
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    else:
        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


# ── Run the server ────────────────────────────────────────────────────

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

Run this as a standalone process:
```bash
python mcp_servers/nomad_mcp_server.py
```

---

#### Connecting LangChain to the MCP Server

```python
# agents/nomad_agent_mcp.py

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

async def build_nomad_agent_with_mcp():
    # Connect to one or more MCP servers
    client = MultiServerMCPClient({
        "nomad": {
            "command": "python",
            "args": ["mcp_servers/nomad_mcp_server.py"],
            "transport": "stdio",
        },
        # Could connect to more servers here:
        # "terraform": { "command": "python", "args": ["mcp_servers/terraform_mcp_server.py"] }
    })

    # Get tools from the MCP server — same interface as regular LangChain tools
    tools = await client.get_tools()

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    agent = create_react_agent(
        model=llm,
        tools=tools,      # tools came from MCP server, not local Python functions
        prompt=NOMAD_SYSTEM_PROMPT,
    )

    return agent
```

---

#### MCP vs Direct Tools — When to Use Which

| | Direct LangChain Tools | MCP Server |
|---|---|---|
| **Best for** | Single project, tools in same codebase | Shared tools across multiple agents/apps |
| **Reusability** | Tied to this project | Any MCP client can use it (Claude Desktop, other agents) |
| **Deployment** | Same process as agent | Separate process, can be on different machine |
| **Latency** | Near-zero (function call) | Small overhead (inter-process communication) |
| **Auth/security** | Handled in Python code | Can add at MCP layer |
| **Use case** | This DevOps project | "I want Claude Desktop to also use these tools" |

**In this project:** We use direct LangChain tools (simpler, same codebase).
**When to switch to MCP:** When you want the same Nomad/Terraform tools to be accessible from Claude Desktop, a different agent framework, or a different service entirely.

---

### 14.8 Tool Summary

```
3 ways to define tools:
  @tool decorator           → Simple. Best for quick queries with minimal inputs.
  StructuredTool            → Recommended. Explicit Pydantic schema, clear validation.
  BaseTool class            → Full control. Use for stateful tools or complex logic.

The LLM uses:
  tool.name                 → to identify which tool to call
  tool.description          → to decide WHEN to call it
  tool.args_schema          → to know WHAT inputs to generate

2 ways to serve tools:
  Direct (in-process)       → Python function, called directly by LangChain
  MCP Server (out-of-process) → JSON-RPC server, any MCP client can connect

Key rule:
  Write descriptions like instructions to a junior engineer.
  Tell the LLM: WHEN to use it, WHEN NOT to, and what ORDER to follow.
```
