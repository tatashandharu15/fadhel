import requests
import json
from typing import Dict, Any, Optional

API_URL = "http://localhost:8000/v1/chat/completions"
TIMEOUT_SECONDS = 120

def get_chat_response(query: str) -> Optional[Dict[str, Any]]:
    """
    Sends a query to the automotive AI backend.
    
    Args:
        query: The user's question.
        
    Returns:
        The JSON response from the backend or None if failed.
        Raises exceptions for connection errors.
    """
    headers = {
        "Content-Type": "application/json"
    }
    
    # Backend expects ChatRequest: {"query": str, ...}
    payload = {
        "query": query,
        "use_rag": True  # Defaulting to True as per system context
    }
    
    try:
        response = requests.post(
            API_URL,
            json=payload,
            headers=headers,
            timeout=TIMEOUT_SECONDS
        )
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        # Re-raise to be handled by the UI
        raise e
