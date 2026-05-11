"""
LangGraph Chatbot with Conversation Memory
Maintains persistent chat history across sessions using checkpointing
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatbotState(TypedDict):
    messages: Annotated[list, "conversation history"]
    user_query: str
    context: str
    response: str


def retrieve_context(state: ChatbotState) -> ChatbotState:
    """
    Retrieve relevant context from vector DB based on conversation history
    """
    from milvus_client import search_news

    # Extract key terms from conversation
    recent_messages = state["messages"][-3:]  # Last 3 exchanges
    query = " ".join([msg.content for msg in recent_messages if isinstance(msg, HumanMessage)])

    # If no conversation history, use current query
    if not query:
        query = state["user_query"]

    articles = search_news(query=query, time_window_hours=48, top_k=5)

    state["context"] = "\n\n".join([
        f"[{a['timestamp']}] {a['title']}: {a['text'][:300]}"
        for a in articles
    ])

    logger.info(f"[Retrieve] Found {len(articles)} relevant articles for context")
    return state


def generate_response(state: ChatbotState) -> ChatbotState:
    """
    Generate response using Sentiment LLM with conversation memory
    """
    from vllm_client import call_chatbot_llm

    # Build prompt with conversation history
    conversation_context = "\n".join([
        f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
        for msg in state["messages"]
    ])

    prompt = f"""Conversation History:
{conversation_context}

Relevant News Context:
{state['context']}

User Query: {state['user_query']}

Provide a helpful response about market sentiment and trading insights."""

    response = call_chatbot_llm(prompt)

    state["response"] = response
    state["messages"].append(HumanMessage(content=state["user_query"]))
    state["messages"].append(AIMessage(content=response))

    logger.info(f"[Response] Generated response ({len(response)} chars)")
    return state


def build_chatbot_graph(db_path: str = ":memory:"):
    """
    Build chatbot graph with persistent memory

    Args:
        db_path: Path to SQLite database for conversation memory
                 Use ":memory:" for in-memory (testing)
                 Use file path for persistent storage (production)
    """
    workflow = StateGraph(ChatbotState)

    workflow.add_node("retrieve", retrieve_context)
    workflow.add_node("respond", generate_response)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "respond")
    workflow.add_edge("respond", END)

    # Add memory checkpointing
    memory = SqliteSaver.from_conn_string(db_path)

    return workflow.compile(checkpointer=memory)


# Global chatbot instance
CHATBOT_DB_PATH = os.getenv("SQLITE_DB_PATH", "/data/chat_memory.db")
chatbot = build_chatbot_graph(CHATBOT_DB_PATH)


async def chat(user_query: str, thread_id: str = "default"):
    """
    Chat with memory persistence across sessions

    Args:
        user_query: User's question
        thread_id: Unique ID for conversation thread (e.g., user ID)

    Returns:
        str: Assistant's response
    """
    config = {"configurable": {"thread_id": thread_id}}

    # Load conversation history automatically from checkpointer
    result = await chatbot.ainvoke(
        {
            "user_query": user_query,
            "messages": [],
            "context": "",
            "response": ""
        },
        config=config
    )

    return result["response"]


# Example multi-turn conversation
if __name__ == "__main__":
    import asyncio

    async def test_conversation():
        """Test multi-turn conversation with memory"""

        # Turn 1
        response1 = await chat("What's the sentiment on AAPL?", thread_id="user123")
        print(f"User: What's the sentiment on AAPL?")
        print(f"Bot: {response1}\n")

        # Turn 2 - Bot remembers AAPL context
        response2 = await chat("How does that compare to yesterday?", thread_id="user123")
        print(f"User: How does that compare to yesterday?")
        print(f"Bot: {response2}\n")

        # Turn 3 - Bot still remembers AAPL
        response3 = await chat("What about TSLA?", thread_id="user123")
        print(f"User: What about TSLA?")
        print(f"Bot: {response3}\n")

        # Different user - fresh conversation
        response4 = await chat("What's trending today?", thread_id="user456")
        print(f"User (different thread): What's trending today?")
        print(f"Bot: {response4}\n")

    asyncio.run(test_conversation())
