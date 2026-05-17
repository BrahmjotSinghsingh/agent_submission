from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

# ─── Internal Data Models (Used by your code, not exposed to the API grader) ──

class AssessmentItem(BaseModel):
    """Rich internal representation of a retrieved assessment."""
    rank: int
    entity_id: str
    name: str
    url: str
    test_type: str
    keys: List[str]
    job_levels: List[str]
    languages: List[str]
    duration: str
    duration_minutes: Optional[int]
    adaptive: bool
    remote: bool
    skill_tags: List[str]
    retrieval_score: float


# ─── API Request Models (What the grader sends you) ───────────────────────────

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    messages: List[Message] = Field(
        ...,
        description="Full conversation history"
    )


# ─── API Response Models (What the grader STRICTLY expects back) ──────────────

class Recommendation(BaseModel):
    """Strict schema required by the SHL evaluator."""
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    """Strict response wrapper required by the SHL evaluator."""
    reply: str = Field(description="The agent's text response")
    recommendations: List[Recommendation] = Field(
        default_factory=list, 
        description="List of 1 to 10 recommended assessments, or empty if gathering context/refusing"
    )
    end_of_conversation: bool = Field(
        description="True only when the agent considers the task complete"
    )


# ─── Utility Models ───────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str

class ErrorResponse(BaseModel):
    detail: str
    error_type: str