from sentence_transformers import SentenceTransformer, util
import torch
from typing import Dict, List, Tuple
import json
import os
import logging

logger = logging.getLogger(__name__)

class IntentClassifier:
    def __init__(self):
        """Initialize the intent classifier."""
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialized = False
        
        # Load intents
        self.intents = self._load_intents()
        self.intent_embeddings = {}
        
        try:
            # Load the model
            logger.info("Loading intent classification model")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.model = self.model.to(self.device)
            
            # Compute intent embeddings
            self.intent_embeddings = self._compute_intent_embeddings()
            self.initialized = True
            logger.info("Intent classification model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading intent classification model: {str(e)}")
            logger.warning("Using fallback intent classification")
        
    def _load_intents(self) -> Dict[str, List[str]]:
        """
        Load predefined intents and their example utterances.
        
        Returns:
            Dict[str, List[str]]: Dictionary mapping intent names to lists of example utterances
        """
        # Get absolute path to data directory
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        intents_file = os.path.join(project_root, "data", "intents.json")
        intents_file = os.path.abspath(intents_file)
        
        if not os.path.exists(intents_file):
            # Create default intents if file doesn't exist
            default_intents = {
                "greeting": [
                    "hello", "hi", "hey", "good morning", "good evening",
                    "hola", "bonjour", "ciao", "hallo", "olá"
                ],
                "farewell": [
                    "goodbye", "bye", "see you", "have a nice day", "take care",
                    "adiós", "au revoir", "arrivederci", "auf wiedersehen", "adeus"
                ],
                "help": [
                    "help", "I need assistance", "can you help me", "support needed",
                    "ayuda", "aide", "aiuto", "hilfe", "ajuda"
                ],
                "product_info": [
                    "product information", "tell me about your products", "what do you sell",
                    "información del producto", "informations sur le produit",
                    "informazioni sul prodotto", "Produktinformation", "informações do produto"
                ]
            }
            os.makedirs(os.path.dirname(intents_file), exist_ok=True)
            with open(intents_file, 'w', encoding='utf-8') as f:
                json.dump(default_intents, f, indent=4, ensure_ascii=False)
            return default_intents
        
        try:
            with open(intents_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading intents file: {str(e)}")
            # Return default intents on error
            return {
                "greeting": ["hello", "hi"],
                "farewell": ["goodbye", "bye"],
                "help": ["help", "assistance"],
                "unknown": ["unknown"]
            }
    
    def _compute_intent_embeddings(self) -> Dict[str, torch.Tensor]:
        """
        Compute embeddings for all intent examples.
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping intent names to their embeddings
        """
        if not self.initialized or self.model is None:
            return {}
            
        intent_embeddings = {}
        try:
            for intent, examples in self.intents.items():
                embeddings = self.model.encode(examples, convert_to_tensor=True)
                intent_embeddings[intent] = embeddings
            return intent_embeddings
        except Exception as e:
            logger.error(f"Error computing intent embeddings: {str(e)}")
            return {}
    
    def classify_intent(self, text: str, threshold: float = 0.5) -> Tuple[str, float]:
        """
        Classify the intent of the input text.
        
        Args:
            text (str): Input text to classify
            threshold (float): Confidence threshold for classification
            
        Returns:
            Tuple[str, float]: Tuple containing (intent_name, confidence_score)
        """
        # If not initialized or no embeddings, return unknown
        if not self.initialized or not self.intent_embeddings:
            return "unknown", 0.0
            
        try:
            # Encode input text
            text_embedding = self.model.encode(text, convert_to_tensor=True)
            
            # Calculate similarities with all intent examples
            max_similarity = -1
            best_intent = "unknown"
            
            for intent, embeddings in self.intent_embeddings.items():
                # Calculate cosine similarities
                similarities = util.pytorch_cos_sim(text_embedding, embeddings)[0]
                max_intent_similarity = similarities.max().item()
                
                if max_intent_similarity > max_similarity:
                    max_similarity = max_intent_similarity
                    best_intent = intent
            
            # Return unknown if confidence is below threshold
            if max_similarity < threshold:
                return "unknown", max_similarity
                
            return best_intent, max_similarity
        except Exception as e:
            logger.error(f"Error classifying intent: {str(e)}")
            return "unknown", 0.0
    
    def get_supported_intents(self) -> List[str]:
        """
        Get a list of supported intents.
        
        Returns:
            List[str]: List of supported intent names
        """
        return list(self.intents.keys()) 