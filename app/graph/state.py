from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

class AgentState(TypedDict):
    """Shared mutable state passed between every LangGraph node."""

    # ─── Input ────────────────────────────────────────────────────────────────
    messages: List[Dict[str, str]]  # e.g., [{"role": "user", "content": "..."}]

    # ─── Internal Processing ──────────────────────────────────────────────────
    current_intent: Optional[str]   # What the user is actually looking for right now
    retrieved_docs: List[Any]       # Will hold our rich Document objects

    # ─── Output (Required by Grader) ──────────────────────────────────────────
    reply: str
    recommendations: List[Dict[str, str]] # Strictly [{"name": "", "url": "", "test_type": ""}]
    end_of_conversation: bool

    # ─── Trace & Error Handling ───────────────────────────────────────────────
    steps: List[str]
    error: Optional[str]