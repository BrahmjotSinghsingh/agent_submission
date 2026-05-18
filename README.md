# agent_submission

This repository contains the FastAPI service for the SHL Assessment Recommender. It is a stateless, consultative conversational agent designed to guide users to appropriate SHL hiring assessments based on their requirements.

## Live API
**Base URL:** `https://agentsubmission-production.up.railway.app`

## Endpoints

### 1. Health Check
**`GET /health`**
Returns the readiness status of the API. 
> **Note:** The service may take up to 50 seconds to wake up from a cold start on the first request.

### 2. Chat
**`POST /chat`**
Takes a stateless conversation history and returns the agent's next reply along with any recommended assessments.

**Example Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "Sure. What is seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
