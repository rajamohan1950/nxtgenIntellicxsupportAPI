from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Dict, Tuple
import logging
import os

logger = logging.getLogger(__name__)

class LanguageDetector:
    def __init__(self):
        """Initialize the language detector model."""
        self.model_name = "papluca/xlm-roberta-base-language-detection"
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialized = False
        
        # Create a fallback language mapping in case model fails to load
        self.fallback_languages = {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "nl": "Dutch",
            "ru": "Russian",
            "zh": "Chinese",
            "ja": "Japanese"
        }
        
        try:
            # Set cache directory to ensure models are saved in a consistent location
            cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
            os.makedirs(cache_dir, exist_ok=True)
            
            logger.info(f"Loading language detection model from {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=cache_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, cache_dir=cache_dir)
            
            # Move model to GPU if available
            self.model = self.model.to(self.device)
            self.initialized = True
            logger.info("Language detection model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading language detection model: {str(e)}")
            logger.warning("Using fallback language detection")
        
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect the language of the input text.
        
        Args:
            text (str): Input text to detect language from
            
        Returns:
            Tuple[str, float]: Tuple containing (detected_language_code, confidence_score)
        """
        # If model failed to initialize, use a simple fallback
        if not self.initialized:
            # Simple fallback: default to English
            return "en", 0.5
            
        try:
            # Tokenize the text
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs.logits.softmax(dim=-1)
                
            # Get the predicted language and confidence score
            predicted_idx = predictions.argmax().item()
            confidence_score = predictions[0][predicted_idx].item()
            predicted_language = self.model.config.id2label[predicted_idx]
            
            return predicted_language, confidence_score
        except Exception as e:
            logger.error(f"Error detecting language: {str(e)}")
            # Fallback to English on error
            return "en", 0.5
        
    def get_supported_languages(self) -> Dict[str, str]:
        """
        Get a dictionary of supported languages.
        
        Returns:
            Dict[str, str]: Dictionary mapping language codes to language names
        """
        if self.initialized and hasattr(self.model, 'config') and hasattr(self.model.config, 'id2label'):
            return self.model.config.id2label
        else:
            # Return fallback languages if model not initialized
            return self.fallback_languages 