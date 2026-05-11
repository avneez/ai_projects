# Interactive Chatbot for Stock News Queries

## Purpose
Allow users to query stock news and insights using natural language. Powered by RAG (same pipeline as prediction system).

## User Interface Options
1. **Streamlit** (Rapid prototyping, internal use)
2. **FastAPI + React** (Production, external users)

## Libraries Used
- **Streamlit** or **FastAPI** - Web framework
- **LangChain** - Conversation management
- **pymilvus** - Vector search
- **vLLM** - LLM inference (shared with prediction system)

## Example Queries

```
User: "What's the latest news about TSLA?"
Bot: [Retrieves top 5 recent TSLA articles]
     "Here are the latest Tesla news:
      1. Tesla Q3 earnings beat expectations (+15% YoY)
      2. New Gigafactory announced in Mexico
      3. Cybertruck production ramp-up delays
      ..."

User: "Why did NVDA drop 5% today?"
Bot: [Searches for NVDA + today + price drop]
     "NVDA dropped due to:
      - Regulatory concerns about China chip exports
      - Profit-taking after 40% rally
      - Sector rotation out of tech
      ..."

User: "Compare sentiment for AAPL vs MSFT this week"
Bot: [Retrieves news for both, generates comparison]
     "AAPL: Mostly positive (sentiment: +0.6)
      - Strong iPhone 15 sales
      MSFT: Mixed (sentiment: +0.2)
      - Azure growth slowing
      ..."
```

## Architecture

### Streamlit Version (Simple)
```
User Input (text box)
  ↓
Query Understanding (extract ticker, time window)
  ↓
Milvus Vector Search (RAG)
  ↓
LLM Response Generation (vLLM)
  ↓
Display to User (markdown + charts)
```

### FastAPI + React Version (Production)
```
React Frontend
  ↓ (HTTP POST)
FastAPI Backend
  ↓
Query Processing
  ↓
RAG Pipeline (Milvus + LangChain)
  ↓
Streaming Response (Server-Sent Events)
  ↓ (Real-time)
React displays token-by-token
```

## Query Processing Logic

### 1. Entity Extraction
```python
# Extract tickers from query
query = "What's happening with Apple and Tesla?"

# Use regex or NER
tickers = extract_tickers(query)  # ["AAPL", "TSLA"]

# Time window detection
time_window = detect_time_window(query)  # Default: "24h"
# Patterns: "today", "this week", "last month"
```

### 2. Milvus Query Construction
```python
def build_search_query(tickers, time_window, user_query):
    # Embedding
    query_embedding = embedder.encode(user_query)

    # Filters
    filters = {
        "tickers": {"$in": tickers},  # Match any ticker
        "timestamp": {"$gte": time_window_start},
        "credibility_score": {"$gte": 0.5}  # Quality threshold
    }

    # Search
    results = milvus_collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "IP", "nprobe": 16},
        limit=10,
        expr=build_filter_expr(filters)  # Milvus filter syntax
    )

    return results
```

### 3. LLM Prompt for Chatbot
```
System: You are a helpful financial news assistant. Provide accurate, concise answers based on the retrieved news articles.

Retrieved Articles:
{article_summaries}

User Question: {user_query}

Instructions:
- Cite article sources when possible
- Be conversational but professional
- If information is outdated, mention it
- If you don't have relevant information, say so

Response:
```

### 4. Conversational Memory (LangChain)
```python
# Maintain context across multiple exchanges
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()

# First exchange
User: "Tell me about Tesla"
Bot: [Provides Tesla news]
memory.save_context({"input": query}, {"output": response})

# Follow-up (uses context)
User: "What about their earnings?"
Bot: [Understands "their" = Tesla from context]
```

## Streamlit Implementation Concept

```python
import streamlit as st
from rag_service import RAGPipeline

# Initialize
st.title("Stock News Chatbot")
rag = RAGPipeline()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask about stock news..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get bot response
    with st.spinner("Thinking..."):
        response = rag.query(prompt)

    # Add bot response
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Rerun to update UI
    st.rerun()
```

## FastAPI Implementation Concept

### Backend Endpoint
```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()
rag = RAGPipeline()

@app.post("/chat")
async def chat(query: str):
    # Streaming response
    def generate():
        for token in rag.query_stream(query):
            yield f"data: {token}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

### React Frontend (Simplified)
```javascript
const [messages, setMessages] = useState([]);
const [input, setInput] = useState("");

const sendMessage = async () => {
  // Add user message
  setMessages([...messages, {role: "user", content: input}]);

  // Fetch response (streaming)
  const response = await fetch("/chat", {
    method: "POST",
    body: JSON.stringify({query: input})
  });

  const reader = response.body.getReader();
  let botMessage = "";

  while (true) {
    const {done, value} = await reader.read();
    if (done) break;

    const token = new TextDecoder().decode(value);
    botMessage += token;

    // Update UI in real-time
    setMessages([...messages, {role: "bot", content: botMessage}]);
  }
};
```

## Advanced Features

### 1. Auto-Suggestions
```
User types: "Tell me about..."
Suggestions:
  - "Tell me about AAPL earnings"
  - "Tell me about market trends"
  - "Tell me about tech sector news"
```

### 2. Chart Integration
```python
# When user asks about price
if "price" in query:
    # Fetch OHLCV from TimescaleDB
    chart_data = get_price_history(ticker)

    # Streamlit: st.line_chart(chart_data)
    # FastAPI: Return JSON for frontend charting library
```

### 3. News Timeline
```
User: "Show me Tesla news timeline this week"

Output:
Monday: Earnings announcement
Tuesday: Stock up 8%
Wednesday: New factory news
Thursday: Analyst upgrade
Friday: CEO interview
```

### 4. Sentiment Trends
```
User: "How has AAPL sentiment changed over time?"

Bot: [Queries Milvus with time bucketing]
     "AAPL sentiment trend:
      Last week: +0.7 (bullish)
      This week: +0.3 (neutral)
      Shift driven by supply chain concerns"
```

## Performance Optimization

### Caching
```python
# Cache frequent queries
@lru_cache(maxsize=1000)
def cached_query(query_hash):
    return rag.query(query)

# Cache key: hash(query + time_window)
# TTL: 5 minutes for real-time queries
```

### Pre-computed Summaries
```
For popular stocks (AAPL, TSLA, NVDA):
- Pre-generate daily summaries
- Store in Redis
- Serve instantly (no RAG latency)
```

## Security & Rate Limiting

### Authentication
```python
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/chat")
async def chat(query: str, token: str = Depends(security)):
    user = verify_token(token)
    # ... process query
```

### Rate Limiting
```python
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)

@app.post("/chat")
@limiter.limit("20/minute")  # 20 queries per minute per IP
async def chat(query: str):
    # ... process
```

### Input Sanitization
```python
def sanitize_input(query):
    # Prevent prompt injection
    query = query.strip()
    query = re.sub(r'<[^>]*>', '', query)  # Remove HTML
    query = query[:500]  # Max length

    # Check for malicious patterns
    forbidden_patterns = ['DROP TABLE', 'DELETE FROM', '<script>']
    for pattern in forbidden_patterns:
        if pattern.lower() in query.lower():
            raise ValueError("Invalid input")

    return query
```

## Deployment

### Streamlit (Internal Use)
```bash
streamlit run chatbot_app.py --server.port 8501
```

### FastAPI (Production)
```bash
# With Gunicorn (production WSGI server)
gunicorn -w 4 -k uvicorn.workers.UvicornWorker chatbot_api:app --bind 0.0.0.0:8000
```

### Docker Compose
```yaml
chatbot:
  build: ./chatbot
  ports:
    - "8000:8000"
  environment:
    - MILVUS_HOST=milvus
    - VLLM_HOST=vllm-server
    - REDIS_HOST=redis
  depends_on:
    - milvus
    - vllm-server
```

## Monitoring

### Metrics
```
- Queries per minute
- Average response time
- User satisfaction (thumbs up/down)
- Most queried stocks
- Cache hit rate
```

### Logging
```python
logger.info(f"Query: {query}, User: {user_id}, Response time: {latency_ms}ms")
```

## Interview Q&A

**Q: How do you handle ambiguous queries?**
A: Ask clarifying questions. "Did you mean AAPL (Apple) or APPL?"  Or show top matches: "I found news for: 1) AAPL, 2) MSFT. Which one?"

**Q: What if user asks about a stock not in the database?**
A: "I don't have recent news for XYZ. It might be a small-cap or delisted stock."  Optionally fetch from external API (e.g., News API).

**Q: How do you prevent chatbot from making investment advice?**
A: System prompt disclaimer: "I provide news summaries, not investment advice. Consult a financial advisor."  Plus legal disclaimer in UI.

**Q: Streamlit vs FastAPI + React trade-off?**
A:
- **Streamlit**: Faster development (1 day), limited customization, internal use
- **FastAPI + React**: More work (1-2 weeks), full control, production-ready
- Start with Streamlit, migrate if needed

**Q: How do you handle real-time updates (e.g., breaking news)?**
A:
- WebSocket connection (FastAPI supports)
- Push notifications when new high-impact news arrives
- Or polling every 30 seconds

**Q: Can users save queries/favorites?**
A: Yes. Store in PostgreSQL:
```sql
CREATE TABLE user_queries (
    user_id INT,
    query TEXT,
    timestamp TIMESTAMP,
    is_favorite BOOLEAN
);
```

**Q: How to evaluate chatbot quality?**
A:
1. User feedback (thumbs up/down)
2. A/B testing different prompts
3. Sample 100 responses, human review
4. Metrics: answer relevance, factual accuracy, latency
