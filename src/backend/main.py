import os
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from src.ml.language_processor import LanguageProcessor
from src.ml.intent_classifier import IntentClassifier
from src.ml.response_generator import ResponseGenerator

app = FastAPI(title="Multilingual Customer Support API")

# Initialize ML components
language_processor = LanguageProcessor(device="cpu")  # or "cuda" if available
intent_classifier = IntentClassifier(device="cpu")
response_generator = ResponseGenerator()

class QueryRequest(BaseModel):
    text: str
    session_id: Optional[str] = None
    context: Optional[List[Dict]] = None

class QueryResponse(BaseModel):
    original_text: str
    detected_language: str
    translated_text: Optional[str]
    intent: str
    intent_confidence: float
    response: str
    translated_response: Optional[str]

@app.post("/process_query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        # Process language
        lang_result = language_processor.process_input(request.text)
        
        # Use translated text for intent classification if available
        text_for_intent = lang_result.get("translated_text") or request.text
        
        # Classify intent
        intent, confidence = intent_classifier.get_primary_intent(text_for_intent)
        
        # Generate response
        response = response_generator.generate_response(
            query=text_for_intent,
            intent=intent,
            context=request.context
        )
        
        # Translate response back if necessary
        translated_response = None
        if lang_result["detected_language"] != "en_XX":
            translated_response = response_generator.generate_multilingual_response(
                query=text_for_intent,
                intent=intent,
                target_language=lang_result["detected_language"],
                context=request.context
            )
        
        return QueryResponse(
            original_text=request.text,
            detected_language=lang_result["detected_language"],
            translated_text=lang_result.get("translated_text"),
            intent=intent,
            intent_confidence=confidence,
            response=response,
            translated_response=translated_response
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port) 