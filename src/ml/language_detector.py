from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Dict, Tuple

class LanguageDetector:
    def __init__(self):
        """Initialize the language detector model."""
        self.model_name = "papluca/xlm-roberta-base-language-detection"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect the language of the input text.
        
        Args:
            text (str): Input text to detect language from
            
        Returns:
            Tuple[str, float]: Tuple containing (detected_language_code, confidence_score)
        """
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
        
    def get_supported_languages(self) -> Dict[str, str]:
        """
        Get a dictionary of supported languages.
        
        Returns:
            Dict[str, str]: Dictionary mapping language codes to language names
        """
        return self.model.config.id2label 