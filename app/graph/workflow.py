from typing import List
from langgraph.graph import END, StateGraph

from app.core.logger import get_logger
from app.graph.nodes import generate_node, retrieve_node
from app.graph.state import AgentState
from app.models.schemas import Message

logger = get_logger(__name__)

def _should_generate(state: AgentState) -> str:
    """Conditional edge: skip generation if retrieval errored OR if intent was resolved."""
    if state.get("error"):
        logger.warning("Skipping generation due to retrieval error")
        return "end"
    
    # If we already generated a reply in the retrieve node, skip the generate node!
    if state.get("current_intent") in ["refuse"]:
        logger.info("Skipping generation; intent handled immediately.")
        return "end"
        
    return "generate"

def build_graph():
    """Build and compile the LangGraph RAG workflow."""
    logger.info("Building LangGraph workflow")

    graph = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.set_entry_point("retrieve")

    # ── Edges ─────────────────────────────────────────────────────────────────
    graph.add_conditional_edges(
        "retrieve",
        _should_generate,
        {"generate": "generate", "end": END},
    )
    graph.add_edge("generate", END)

    compiled = graph.compile()
    logger.info("LangGraph workflow compiled")
    return compiled


async def run_rag_agent(messages: List[Message]) -> AgentState:
    """Execute the RAG workflow and return the final state."""
    app = build_graph()

    # Convert Pydantic Message models to raw dicts for LangGraph state
    dict_messages = [{"role": m.role, "content": m.content} for m in messages]

    initial_state: AgentState = {
        "messages": dict_messages,
        "current_intent": None,
        "retrieved_docs": [],
        "reply": "",
        "recommendations": [],
        "end_of_conversation": False,
        "steps": ["start: messages received"],
        "error": None,
    }

    logger.info("Running RAG agent with %d messages", len(dict_messages))
    final_state = await app.ainvoke(initial_state)
    logger.info("RAG agent complete. Steps: %s", final_state.get("steps"))
    
    return final_state