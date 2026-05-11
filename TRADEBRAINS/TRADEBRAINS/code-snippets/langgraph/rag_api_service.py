"""
FastAPI Service for Adaptive RAG State Machine
Exposes REST API endpoint for sentiment analysis with state transition logging
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from adaptive_rag_graph import run_adaptive_rag
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LangGraph Adaptive RAG Service")


class SentimentRequest(BaseModel):
    symbol: str
    query: str = None


class SentimentResponse(BaseModel):
    sentiment: dict
    confidence: str
    articles_used: int
    state_transitions: list


@app.post("/analyze_sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """
    Analyze market sentiment for a stock symbol using adaptive RAG

    The state machine will:
    1. Try to retrieve news from last 24h
    2. If insufficient (<5 articles), expand to 72h
    3. If still insufficient (<3 articles), fall back to cached sentiment
    4. Filter by credibility score > 0.7
    5. Generate sentiment using Sentiment LLM
    """
    try:
        logger.info(f"[API] Received sentiment request for {request.symbol}")

        # Run adaptive RAG state machine
        result = await run_adaptive_rag(request.symbol)

        # Mock state transitions for demo (in production, extract from logs)
        state_transitions = [
            f"retrieve (24h) → found {result['articles_used']} articles",
            f"validate → {result['articles_used']} passed credibility check",
            f"generate_sentiment → confidence={result['confidence']}"
        ]

        return SentimentResponse(
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            articles_used=result["articles_used"],
            state_transitions=state_transitions
        )

    except Exception as e:
        logger.error(f"[API] Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "langgraph-rag"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8003))
    uvicorn.run(app, host="0.0.0.0", port=port)
