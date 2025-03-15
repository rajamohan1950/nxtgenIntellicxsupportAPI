from typing import Dict, Optional, Tuple
from .language_detector import LanguageDetector
from .intent_classifier import IntentClassifier
from .response_generator import ResponseGenerator

class CustomerSupportService:
    def __init__(self):
        """Initialize the multilingual customer support service."""
        self.language_detector = LanguageDetector()
        self.intent_classifier = IntentClassifier()
        self.response_generator = ResponseGenerator()
        
    def process_query(self, query: str, preferred_language: Optional[str] = None) -> Dict:
        """
        Process a customer query and generate an appropriate response.
        
        Args:
            query (str): The customer's query text
            preferred_language (Optional[str]): The customer's preferred language code, if known
            
        Returns:
            Dict: Response containing detected language, intent, confidence scores, and response
        """
        # Detect language if not provided
        if preferred_language:
            detected_language = preferred_language
            language_confidence = 1.0
        else:
            detected_language, language_confidence = self.language_detector.detect_language(query)
        
        # Classify intent
        intent, intent_confidence = self.intent_classifier.classify_intent(query)
        
        # Generate response in detected/preferred language
        response = self.response_generator.get_response(intent, detected_language)
        
        return {
            "query": query,
            "detected_language": detected_language,
            "language_confidence": language_confidence,
            "intent": intent,
            "intent_confidence": intent_confidence,
            "response": response
        }
        
    def get_supported_languages(self) -> list:
        """
        Get list of supported languages.
        
        Returns:
            list: List of supported language codes
        """
        return self.response_generator.get_supported_languages()
        
    def get_supported_intents(self) -> list:
        """
        Get list of supported intents.
        
        Returns:
            list: List of supported intent names
        """
        return self.intent_classifier.get_supported_intents() 