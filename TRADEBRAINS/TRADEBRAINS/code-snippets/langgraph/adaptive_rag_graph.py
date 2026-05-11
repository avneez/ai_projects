"""
LangGraph Adaptive RAG State Machine
Handles news retrieval with dynamic time window expansion and fallback strategies
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define state
class RAGState(TypedDict):
    symbol: str
    query: str
    time_window: int  # hours
    articles: list
    credibility_filtered: list
    sentiment_scores: dict
    confidence: str  # "high", "medium", "low"
    attempt: int


# Define nodes (states)
def retrieve_news(state: RAGState) -> RAGState:
    """
    State 1: Retrieve news from Milvus vector DB
    """
    from milvus_client import search_news

    articles = search_news(
        query=state["query"],
        time_window_hours=state["time_window"],
        top_k=10 if state["time_window"] == 24 else 15
    )

    state["articles"] = articles
    logger.info(f"[Retrieve] Found {len(articles)} articles (window={state['time_window']}h)")
    return state


def validate_credibility(state: RAGState) -> RAGState:
    """
    State 2: Filter news by credibility score
    """
    filtered = [
        article for article in state["articles"]
        if article.get("credibility_score", 0) > 0.7
    ]

    state["credibility_filtered"] = filtered
    logger.info(f"[Validate] {len(filtered)}/{len(state['articles'])} articles passed credibility check")
    return state


def generate_sentiment(state: RAGState) -> RAGState:
    """
    State 3: Generate sentiment scores using Sentiment LLM
    """
    from vllm_client import call_sentiment_llm

    # Build context from articles
    context = "\n\n".join([
        f"[{article['source']}] {article['title']}: {article['text'][:500]}"
        for article in state["credibility_filtered"]
    ])

    # Call Sentiment LLM
    sentiment = call_sentiment_llm(
        symbol=state["symbol"],
        news_context=context
    )

    state["sentiment_scores"] = sentiment

    # Set confidence based on article count
    if len(state["credibility_filtered"]) >= 5:
        state["confidence"] = "high"
    elif len(state["credibility_filtered"]) >= 3:
        state["confidence"] = "medium"
    else:
        state["confidence"] = "low"

    logger.info(f"[Sentiment] Generated scores with confidence={state['confidence']}")
    return state


def use_fallback_sentiment(state: RAGState) -> RAGState:
    """
    State 3b: Fallback when no news available - use cached sentiment + fundamentals
    """
    from cache import get_cached_sentiment, get_company_fundamentals

    cached = get_cached_sentiment(state["symbol"], max_age_hours=72)
    fundamentals = get_company_fundamentals(state["symbol"])

    # Generate conservative sentiment from fundamentals
    state["sentiment_scores"] = {
        "market_sentiment_score": cached.get("market_sentiment_score", 0.0),
        "fear_greed_score": 50,  # Neutral
        "upside_catalyst_rating": fundamentals.get("analyst_rating", 5),
        "downside_risk_rating": 5,  # Neutral
        "event_importance_score": 2,  # Low (no recent news)
        "sector_impact": fundamentals.get("sector_momentum", 5)
    }
    state["confidence"] = "low"

    logger.info(f"[Fallback] Using cached sentiment + fundamentals (confidence=low)")
    return state


# Define conditional routing
def should_expand_window(state: RAGState) -> Literal["expand", "validate"]:
    """
    Route: If insufficient articles, expand time window
    """
    if len(state["articles"]) < 5 and state["attempt"] == 0:
        return "expand"
    return "validate"


def should_use_fallback(state: RAGState) -> Literal["fallback", "generate"]:
    """
    Route: If still insufficient after expansion, use fallback
    """
    if len(state["credibility_filtered"]) < 3:
        return "fallback"
    return "generate"


def expand_time_window(state: RAGState) -> RAGState:
    """
    State 2b: Expand retrieval time window from 24h → 72h
    """
    state["time_window"] = 72
    state["attempt"] = 1
    logger.info(f"[Expand] Expanding time window to {state['time_window']}h")
    return state


# Build the graph
def build_adaptive_rag_graph():
    workflow = StateGraph(RAGState)

    # Add nodes
    workflow.add_node("retrieve", retrieve_news)
    workflow.add_node("expand_window", expand_time_window)
    workflow.add_node("validate", validate_credibility)
    workflow.add_node("generate_sentiment", generate_sentiment)
    workflow.add_node("fallback_sentiment", use_fallback_sentiment)

    # Add edges
    workflow.set_entry_point("retrieve")

    # Conditional routing after retrieval
    workflow.add_conditional_edges(
        "retrieve",
        should_expand_window,
        {
            "expand": "expand_window",
            "validate": "validate"
        }
    )

    # If expanded, retry retrieval
    workflow.add_edge("expand_window", "retrieve")

    # Conditional routing after validation
    workflow.add_conditional_edges(
        "validate",
        should_use_fallback,
        {
            "fallback": "fallback_sentiment",
            "generate": "generate_sentiment"
        }
    )

    # Both paths lead to END
    workflow.add_edge("generate_sentiment", END)
    workflow.add_edge("fallback_sentiment", END)

    return workflow.compile()


# Usage
async def run_adaptive_rag(symbol: str):
    """
    Run adaptive RAG state machine for a stock symbol
    """
    graph = build_adaptive_rag_graph()

    # Initial state
    initial_state = {
        "symbol": symbol,
        "query": f"{symbol} stock news earnings market",
        "time_window": 24,
        "articles": [],
        "credibility_filtered": [],
        "sentiment_scores": {},
        "confidence": "unknown",
        "attempt": 0
    }

    # Execute state machine
    result = await graph.ainvoke(initial_state)

    return {
        "sentiment": result["sentiment_scores"],
        "confidence": result["confidence"],
        "articles_used": len(result["credibility_filtered"])
    }


# Example execution
if __name__ == "__main__":
    result = asyncio.run(run_adaptive_rag("AAPL"))
    print(result)
    # Output: {'sentiment': {...}, 'confidence': 'high', 'articles_used': 7}
