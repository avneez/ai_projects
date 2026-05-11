"""
LangGraph Prediction Pipeline State Machine
Handles feature fetching, risk checks, prediction, and confidence gating with error recovery
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionState(TypedDict):
    symbol: str
    technical_features: Optional[dict]
    sentiment_scores: Optional[dict]
    sentiment_confidence: str
    gnn_embeddings: Optional[list]
    risk_check_passed: bool
    prediction: Optional[dict]
    final_decision: str
    reasoning: str
    retry_count: int


def fetch_features(state: PredictionState) -> PredictionState:
    """
    State 1: Fetch technical indicators from TimescaleDB
    """
    from timescaledb_client import fetch_technical_indicators

    try:
        features = fetch_technical_indicators(
            symbol=state["symbol"],
            lookback_periods=100
        )
        state["technical_features"] = features
        logger.info(f"[Fetch] Loaded {len(features)} technical indicators")
    except Exception as e:
        if state["retry_count"] < 3:
            state["retry_count"] += 1
            logger.warning(f"[Fetch] Error: {e}. Retry {state['retry_count']}/3")
            raise  # Will trigger retry
        else:
            state["technical_features"] = None
            logger.error(f"[Fetch] Failed after 3 retries. Using cached features.")

    return state


def risk_precheck(state: PredictionState) -> PredictionState:
    """
    State 2: Pre-check risk limits before expensive LLM call
    """
    from portfolio_manager import check_position_limits, get_current_position, get_margin_available

    risk_ok = check_position_limits(
        symbol=state["symbol"],
        current_position=get_current_position(state["symbol"]),
        margin_available=get_margin_available()
    )

    state["risk_check_passed"] = risk_ok

    if not risk_ok:
        state["final_decision"] = "HOLD"
        state["reasoning"] = "Position limit exceeded or insufficient margin"
        logger.info(f"[Risk] Pre-check failed: {state['reasoning']}")

    return state


def run_price_prediction(state: PredictionState) -> PredictionState:
    """
    State 3: Run Price Prediction LLM
    """
    from vllm_client import call_price_prediction_llm

    prediction = call_price_prediction_llm(
        symbol=state["symbol"],
        technical_features=state["technical_features"],
        sentiment_scores=state["sentiment_scores"],
        gnn_embeddings=state["gnn_embeddings"]
    )

    state["prediction"] = prediction
    logger.info(f"[Predict] {prediction['action']} (confidence={prediction['confidence']:.2f})")

    return state


def apply_confidence_gating(state: PredictionState) -> PredictionState:
    """
    State 4: Apply confidence thresholds and sentiment reliability checks
    """
    pred_conf = state["prediction"]["confidence"]
    sent_conf = state["sentiment_confidence"]

    # Low prediction confidence → HOLD
    if pred_conf < 0.6:
        state["final_decision"] = "HOLD"
        state["reasoning"] = f"Low prediction confidence ({pred_conf:.2f})"

    # Medium confidence + unreliable sentiment → HOLD
    elif 0.6 <= pred_conf < 0.75 and sent_conf == "low":
        state["final_decision"] = "HOLD"
        state["reasoning"] = f"Unreliable news data (sentiment_conf={sent_conf})"

    # High confidence or medium + good sentiment → Execute
    else:
        state["final_decision"] = state["prediction"]["action"]
        state["reasoning"] = f"High confidence trade (pred={pred_conf:.2f}, sent={sent_conf})"

    logger.info(f"[Gate] Final decision: {state['final_decision']} - {state['reasoning']}")

    return state


# Conditional routing
def should_run_prediction(state: PredictionState) -> Literal["predict", "end"]:
    """
    Route: Only run prediction if risk check passed
    """
    if state["risk_check_passed"]:
        return "predict"
    return "end"


def should_retry_fetch(state: PredictionState) -> Literal["retry", "risk"]:
    """
    Route: Retry fetch if failed and retries remaining
    """
    if state["technical_features"] is None and state["retry_count"] < 3:
        return "retry"
    return "risk"


# Build graph
def build_prediction_pipeline_graph():
    workflow = StateGraph(PredictionState)

    workflow.add_node("fetch", fetch_features)
    workflow.add_node("risk", risk_precheck)
    workflow.add_node("predict", run_price_prediction)
    workflow.add_node("gate", apply_confidence_gating)

    workflow.set_entry_point("fetch")

    # Conditional retry on fetch failure
    workflow.add_conditional_edges(
        "fetch",
        should_retry_fetch,
        {
            "retry": "fetch",  # Loop back
            "risk": "risk"
        }
    )

    workflow.add_conditional_edges(
        "risk",
        should_run_prediction,
        {
            "predict": "predict",
            "end": END
        }
    )

    workflow.add_edge("predict", "gate")
    workflow.add_edge("gate", END)

    return workflow.compile()


# Usage
async def run_prediction_pipeline(symbol: str, sentiment_result: dict, gnn_embeddings: list):
    """
    Run prediction pipeline state machine
    """
    graph = build_prediction_pipeline_graph()

    initial_state = {
        "symbol": symbol,
        "technical_features": None,
        "sentiment_scores": sentiment_result["sentiment"],
        "sentiment_confidence": sentiment_result["confidence"],
        "gnn_embeddings": gnn_embeddings,
        "risk_check_passed": False,
        "prediction": None,
        "final_decision": "HOLD",
        "reasoning": "",
        "retry_count": 0
    }

    result = await graph.ainvoke(initial_state)

    return {
        "action": result["final_decision"],
        "confidence": result["prediction"]["confidence"] if result["prediction"] else 0.0,
        "reasoning": result["reasoning"]
    }


# Example execution
if __name__ == "__main__":
    import asyncio

    # Mock sentiment result
    sentiment_result = {
        "sentiment": {
            "market_sentiment_score": 0.72,
            "fear_greed_score": 65,
            "upside_catalyst_rating": 7,
            "downside_risk_rating": 3,
            "event_importance_score": 8,
            "sector_impact": 6
        },
        "confidence": "high"
    }

    # Mock GNN embeddings
    gnn_embeddings = [0.1] * 128

    result = asyncio.run(run_prediction_pipeline("AAPL", sentiment_result, gnn_embeddings))
    print(result)
