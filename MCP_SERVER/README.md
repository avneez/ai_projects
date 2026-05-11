# Model Context Protocol (MCP) Server — In-Depth Guide

## Table of Contents
1. [What is MCP?](#what-is-mcp)
2. [Why Use MCP?](#why-use-mcp)
3. [Core Concepts](#core-concepts)
4. [Architecture](#architecture)
5. [Creating an MCP Server](#creating-an-mcp-server)
6. [Tools, Resources, and Prompts](#tools-resources-and-prompts)
7. [Real-World Example: File System + DB MCP Server](#real-world-example)
8. [Connecting to Claude](#connecting-to-claude)
9. [Transport Layers](#transport-layers)
10. [Best Practices](#best-practices)

---

## What is MCP?

**Model Context Protocol (MCP)** is an open standard developed by Anthropic that defines how AI models (like Claude) communicate with external tools, data sources, and services.

Think of MCP as a **USB-C port for AI** — a universal plug that lets any LLM connect to any tool without custom integrations per tool per model.

```
Without MCP:                        With MCP:
┌─────────┐  custom code  ┌──────┐  ┌─────────┐  MCP  ┌──────────────┐
│  Claude │ ────────────> │ Tool │  │  Claude │ ────> │  MCP Server  │
└─────────┘               └──────┘  └─────────┘       │  (any tool)  │
                                                        └──────────────┘
```

MCP follows a **client-server** model:
- **MCP Host** — the AI app (e.g. Claude Desktop, Claude Code)
- **MCP Client** — lives inside the host, manages connections
- **MCP Server** — lightweight process exposing tools/data to the model

---

## Why Use MCP?

| Problem (Before MCP) | Solution (With MCP) |
|---|---|
| Every tool needs custom integration per model | One MCP server works with any MCP-compatible model |
| AI can only see data in its context window | MCP servers stream real-time, structured data |
| Tools tightly coupled to LLM code | Tools are independent processes — isolated & reusable |
| No standard for permissions/security | MCP defines capability negotiation and scoping |
| Hard to test/debug LLM–tool interactions | Servers are plain processes — debuggable independently |

**Real benefits:**
- Give Claude access to your database, APIs, filesystem, or internal services
- Build once, use across Claude Desktop, Claude Code, any MCP client
- Each server runs in its own process — no shared state, safer execution
- Composable: Claude can call multiple MCP servers in one conversation

---

## Core Concepts

### 1. Tools
Functions the LLM can **call** (like function calling in OpenAI).
```
Claude → "search_database(query='users created today')" → MCP Server → DB → result
```

### 2. Resources
Data the LLM can **read** — files, DB rows, API responses. Identified by URI.
```
resource://myserver/users/42  →  { id: 42, name: "Alice", ... }
```

### 3. Prompts
Pre-defined prompt templates the LLM or user can **invoke**.
```
prompt: "summarize_ticket" with argument ticket_id=123
```

### 4. Sampling
The server can ask the LLM to **generate text** (server → model, not just model → server).

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    MCP Host (Claude Desktop)              │
│                                                           │
│   ┌─────────────┐     JSON-RPC 2.0      ┌─────────────┐  │
│   │  MCP Client │ <──────────────────> │  MCP Server │  │
│   │  (built-in) │   stdio / SSE / HTTP │  (your code)│  │
│   └─────────────┘                       └──────┬──────┘  │
│                                                │          │
└────────────────────────────────────────────────│──────────┘
                                                 │
                                    ┌────────────▼──────────┐
                                    │  External Systems      │
                                    │  (DB, APIs, FS, etc.) │
                                    └───────────────────────┘

Communication: JSON-RPC 2.0 over stdio (local) or SSE/HTTP (remote)
```

---

## Creating an MCP Server

### Prerequisites
```bash
# Python SDK
pip install mcp

# OR Node.js SDK
npm install @modelcontextprotocol/sdk
```

---

### Minimal MCP Server (Python)

```python
# server.py
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import asyncio

# 1. Create the server instance
app = Server("my-first-mcp-server")

# 2. Register a tool
@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="add_numbers",
            description="Add two numbers together",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        )
    ]

# 3. Handle tool calls
@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "add_numbers":
        result = arguments["a"] + arguments["b"]
        return [TextContent(type="text", text=str(result))]
    raise ValueError(f"Unknown tool: {name}")

# 4. Run over stdio
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:
```bash
python server.py
# Server is now listening on stdio, ready for MCP client connections
```

---

### Minimal MCP Server (Node.js / TypeScript)

```typescript
// server.ts
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

// 1. Create server
const server = new Server(
  { name: "my-first-mcp-server", version: "1.0.0" },
  { capabilities: { tools: {} } }
);

// 2. List available tools
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "add_numbers",
      description: "Add two numbers together",
      inputSchema: {
        type: "object",
        properties: {
          a: { type: "number", description: "First number" },
          b: { type: "number", description: "Second number" },
        },
        required: ["a", "b"],
      },
    },
  ],
}));

// 3. Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  if (request.params.name === "add_numbers") {
    const { a, b } = request.params.arguments as { a: number; b: number };
    return { content: [{ type: "text", text: String(a + b) }] };
  }
  throw new Error(`Unknown tool: ${request.params.name}`);
});

// 4. Start over stdio
const transport = new StdioServerTransport();
await server.connect(transport);
```

---

## Tools, Resources, and Prompts

### Exposing a Resource (Python)

```python
from mcp.types import Resource
from urllib.parse import urlparse

@app.list_resources()
async def list_resources() -> list[Resource]:
    return [
        Resource(
            uri="resource://myserver/config",
            name="App Config",
            description="Current application configuration",
            mimeType="application/json",
        )
    ]

@app.read_resource()
async def read_resource(uri: str) -> str:
    if uri == "resource://myserver/config":
        import json
        return json.dumps({"debug": True, "version": "1.2.3"})
    raise ValueError(f"Unknown resource: {uri}")
```

### Exposing a Prompt Template (Python)

```python
from mcp.types import Prompt, PromptArgument, PromptMessage

@app.list_prompts()
async def list_prompts() -> list[Prompt]:
    return [
        Prompt(
            name="code_review",
            description="Review code for a specific language",
            arguments=[
                PromptArgument(name="language", description="Programming language", required=True),
                PromptArgument(name="code", description="Code to review", required=True),
            ],
        )
    ]

@app.get_prompt()
async def get_prompt(name: str, arguments: dict) -> list[PromptMessage]:
    if name == "code_review":
        lang = arguments["language"]
        code = arguments["code"]
        return [
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"Please review this {lang} code for bugs, style, and performance:\n\n```{lang}\n{code}\n```",
                ),
            )
        ]
    raise ValueError(f"Unknown prompt: {name}")
```

---

## Real-World Example

### SQLite + Filesystem MCP Server (Python)

This server gives Claude the ability to:
- Query a SQLite database
- Read files from a directory
- Write new files

```python
# sqlite_fs_server.py
import asyncio
import sqlite3
import os
from pathlib import Path
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, Resource, TextContent

BASE_DIR = Path("/tmp/mcp_workspace")
DB_PATH = BASE_DIR / "data.db"
BASE_DIR.mkdir(exist_ok=True)

# Seed the database
conn = sqlite3.connect(DB_PATH)
conn.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)")
conn.execute("INSERT OR IGNORE INTO users VALUES (1,'Alice','alice@example.com')")
conn.execute("INSERT OR IGNORE INTO users VALUES (2,'Bob','bob@example.com')")
conn.commit()
conn.close()

app = Server("sqlite-fs-server")

# ── TOOLS ─────────────────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="query_db",
            description="Run a SELECT query on the SQLite database",
            inputSchema={
                "type": "object",
                "properties": {"sql": {"type": "string", "description": "SQL SELECT query"}},
                "required": ["sql"],
            },
        ),
        Tool(
            name="write_file",
            description="Write content to a file in the workspace",
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["filename", "content"],
            },
        ),
        Tool(
            name="read_file",
            description="Read a file from the workspace",
            inputSchema={
                "type": "object",
                "properties": {"filename": {"type": "string"}},
                "required": ["filename"],
            },
        ),
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "query_db":
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.execute(arguments["sql"])
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        conn.close()
        result = [dict(zip(cols, row)) for row in rows]
        return [TextContent(type="text", text=str(result))]

    elif name == "write_file":
        # Prevent path traversal
        safe_path = BASE_DIR / Path(arguments["filename"]).name
        safe_path.write_text(arguments["content"])
        return [TextContent(type="text", text=f"Written to {safe_path}")]

    elif name == "read_file":
        safe_path = BASE_DIR / Path(arguments["filename"]).name
        if not safe_path.exists():
            return [TextContent(type="text", text="File not found")]
        return [TextContent(type="text", text=safe_path.read_text())]

    raise ValueError(f"Unknown tool: {name}")

# ── RESOURCES ─────────────────────────────────────────────────────────────────

@app.list_resources()
async def list_resources() -> list[Resource]:
    files = list(BASE_DIR.glob("*"))
    return [
        Resource(uri=f"file://workspace/{f.name}", name=f.name, mimeType="text/plain")
        for f in files if f.is_file()
    ]

@app.read_resource()
async def read_resource(uri: str) -> str:
    filename = uri.split("/")[-1]
    path = BASE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(uri)
    return path.read_text()

# ── MAIN ──────────────────────────────────────────────────────────────────────

async def main():
    async with stdio_server() as (r, w):
        await app.run(r, w, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
```

**What Claude can now do with this server:**
```
User: "How many users are in the database? Write a summary to a file."

Claude:
  1. Calls query_db("SELECT COUNT(*) as total FROM users")  → { total: 2 }
  2. Calls query_db("SELECT * FROM users")                  → [Alice, Bob]
  3. Calls write_file("summary.txt", "2 users: Alice, Bob") → Written
```

---

## Connecting to Claude

### Claude Desktop (`~/Library/Application Support/Claude/claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "sqlite-fs": {
      "command": "python",
      "args": ["/path/to/sqlite_fs_server.py"]
    },
    "my-node-server": {
      "command": "node",
      "args": ["/path/to/server.js"]
    }
  }
}
```

### Claude Code (CLI)

```bash
# Add a server
claude mcp add my-server python /path/to/server.py

# List servers
claude mcp list

# Remove a server
claude mcp remove my-server
```

After connecting, Claude automatically discovers your server's tools/resources and uses them during conversations.

---

## Transport Layers

| Transport | Use Case | How |
|---|---|---|
| **stdio** (default) | Local servers, CLI tools | Server reads/writes stdin/stdout |
| **SSE** (Server-Sent Events) | Remote/web servers | HTTP with event streaming |
| **HTTP Streamable** | Modern remote (MCP 2025-03) | Stateless HTTP with streaming |

### SSE Transport (for remote servers)

```python
# server_sse.py
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route
import uvicorn

app = Server("remote-server")
# ... register tools same as before ...

sse = SseServerTransport("/messages/")

async def handle_sse(request):
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await app.run(streams[0], streams[1], app.create_initialization_options())

starlette_app = Starlette(routes=[Route("/sse", endpoint=handle_sse)])

if __name__ == "__main__":
    uvicorn.run(starlette_app, host="0.0.0.0", port=8000)
```

---

## Best Practices

### 1. Input Validation & Security
```python
import re

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "query_db":
        sql = arguments["sql"].strip()
        # Only allow SELECT — never allow writes through this tool
        if not re.match(r"^\s*SELECT\b", sql, re.IGNORECASE):
            return [TextContent(type="text", text="Error: Only SELECT queries are allowed")]
        # Use parameterized queries when you have user values
        ...
```

### 2. Descriptive Tool Schemas
```python
# Bad — Claude won't know what to pass
Tool(name="process", inputSchema={"type": "object", "properties": {"x": {"type": "string"}}})

# Good — Claude understands exactly what to send
Tool(
    name="search_users",
    description="Search users by name or email. Returns matching user records.",
    inputSchema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Partial name or email to search for (case-insensitive)",
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return (default 10, max 100)",
                "default": 10,
            },
        },
        "required": ["query"],
    },
)
```

### 3. Error Handling
```python
@app.call_tool()
async def call_tool(name: str, arguments: dict):
    try:
        if name == "query_db":
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.execute(arguments["sql"])
            rows = cursor.fetchall()
            conn.close()
            return [TextContent(type="text", text=str(rows))]
    except sqlite3.OperationalError as e:
        # Return errors as text — Claude will handle them gracefully
        return [TextContent(type="text", text=f"Database error: {e}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Unexpected error: {e}")]
```

### 4. Project Structure

```
my-mcp-server/
├── server.py          # Entry point + server setup
├── tools/
│   ├── database.py    # DB tool handlers
│   ├── filesystem.py  # FS tool handlers
│   └── api.py         # External API tools
├── resources/
│   └── handlers.py    # Resource read handlers
├── requirements.txt
└── README.md
```

---

## Quick Reference

```bash
# Install Python SDK
pip install mcp

# Install Node SDK
npm install @modelcontextprotocol/sdk

# Test your server manually (send JSON-RPC over stdin)
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | python server.py

# Add to Claude Code
claude mcp add myserver python /path/to/server.py

# Inspect available tools after connecting
# Just ask Claude: "What tools do you have available?"
```

---

## Further Reading

- [MCP Specification](https://spec.modelcontextprotocol.io)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk)
- [Official MCP Servers (reference implementations)](https://github.com/modelcontextprotocol/servers)
