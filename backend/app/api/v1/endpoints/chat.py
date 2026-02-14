from fastapi import APIRouter, Depends, HTTPException
from backend.app.schemas.chat import ChatRequest, ChatResponse
from backend.app.services.orchestrator import ChatOrchestrator

router = APIRouter()

# Dependency Injection untuk Orchestrator
# Agar service bisa direuse atau di-mock saat testing
def get_orchestrator():
    return ChatOrchestrator()

@router.post("/completions", response_model=ChatResponse)
async def chat_completions(
    request: ChatRequest,
    orchestrator: ChatOrchestrator = Depends(get_orchestrator)
):
    """
    Endpoint utama untuk Chat.
    
    Flow:
    1. Validate request (FastAPI default)
    2. Orchestrator memproses query
    3. Return response JSON
    """
    try:
        response = await orchestrator.process_request(request)
        return response
    except Exception as e:
        # Basic Error Handling
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    return {"status": "ok", "component": "api_layer"}
