from fastapi import APIRouter, HTTPException, status
from app.core.logger import get_logger
from app.graph.workflow import run_rag_agent
from app.models.schemas import ChatRequest, ChatResponse, HealthResponse

logger = get_logger(__name__)
router = APIRouter()

# ─── Health Check ─────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    tags=["Utility"],
)
async def health():
    """Returns exact schema expected by the SHL automated evaluator."""
    return HealthResponse(status="ok")


# ─── Chat Endpoint ────────────────────────────────────────────────────────────

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Conversational endpoint for SHL assessment recommendations",
    tags=["Conversational Agent"],
)
async def chat(body: ChatRequest):
    """
    Takes a stateless conversation history and returns the next agent reply, 
    plus a structured shortlist of recommendations when appropriate.
    """
    logger.info("POST /chat received %d messages", len(body.messages))

    try:
        # Pass the whole message history to LangGraph
        final_state = await run_rag_agent(messages=body.messages)
    except Exception as exc:
        logger.exception("Unhandled error in RAG agent: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent failed: {exc}",
        )

    return ChatResponse(
        reply=final_state.get("reply", "I'm sorry, I encountered an error processing that."),
        recommendations=final_state.get("recommendations", []),
        end_of_conversation=final_state.get("end_of_conversation", False)
    )

# (You can leave the lightweight /search endpoint in routes.py if you want it for debugging, 
# but the evaluator will only hit /health and /chat)