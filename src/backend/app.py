from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.customer_support_service import CustomerSupportService

app = FastAPI(
    title="Multilingual Customer Support API",
    description="API for processing customer support queries in multiple languages",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize customer support service
service = CustomerSupportService()

class QueryRequest(BaseModel):
    text: str
    preferred_language: Optional[str] = None

class QueryResponse(BaseModel):
    query: str
    detected_language: str
    language_confidence: float
    intent: str
    intent_confidence: float
    response: str

@app.post("/process_query", response_model=QueryResponse)
async def process_query(request: QueryRequest) -> Dict:
    """
    Process a customer support query and return the response.
    """
    try:
        result = service.process_query(
            query=request.text,
            preferred_language=request.preferred_language
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/supported_languages", response_model=List[str])
async def get_supported_languages() -> List[str]:
    """
    Get list of supported languages.
    """
    try:
        return service.get_supported_languages()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/supported_intents", response_model=List[str])
async def get_supported_intents() -> List[str]:
    """
    Get list of supported intents.
    """
    try:
        return service.get_supported_intents()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 