from transformers import MarianMTModel, MarianTokenizer
import torch
from typing import Dict, Optional
import json
import os

class ResponseGenerator:
    def __init__(self):
        """Initialize the multilingual response generator."""
        # Initialize translation models for different language pairs
        self.translation_models: Dict[str, tuple] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load response templates
        self.responses = self._load_responses()
        
    def _load_responses(self) -> Dict[str, Dict[str, str]]:
        """
        Load response templates for different intents.
        
        Returns:
            Dict[str, Dict[str, str]]: Dictionary mapping intents to their response templates
        """
        responses_file = os.path.join(os.path.dirname(__file__), "../../data/responses.json")
        if not os.path.exists(responses_file):
            # Create default responses if file doesn't exist
            default_responses = {
                "greeting": {
                    "en": "Hello! How can I assist you today?",
                    "es": "¡Hola! ¿Cómo puedo ayudarte hoy?",
                    "fr": "Bonjour! Comment puis-je vous aider aujourd'hui?",
                    "de": "Hallo! Wie kann ich Ihnen heute helfen?",
                    "it": "Ciao! Come posso aiutarti oggi?",
                    "pt": "Olá! Como posso ajudá-lo hoje?"
                },
                "farewell": {
                    "en": "Goodbye! Have a great day!",
                    "es": "¡Adiós! ¡Que tengas un excelente día!",
                    "fr": "Au revoir! Passez une excellente journée!",
                    "de": "Auf Wiedersehen! Haben Sie einen schönen Tag!",
                    "it": "Arrivederci! Buona giornata!",
                    "pt": "Adeus! Tenha um ótimo dia!"
                },
                "help": {
                    "en": "I'm here to help! Please let me know what you need assistance with.",
                    "es": "¡Estoy aquí para ayudar! Por favor, dime en qué necesitas ayuda.",
                    "fr": "Je suis là pour vous aider! Dites-moi ce dont vous avez besoin.",
                    "de": "Ich bin hier um zu helfen! Bitte lassen Sie mich wissen, wobei Sie Hilfe benötigen.",
                    "it": "Sono qui per aiutare! Per favore, fammi sapere con cosa hai bisogno di assistenza.",
                    "pt": "Estou aqui para ajudar! Por favor, me diga com o que você precisa de ajuda."
                },
                "product_info": {
                    "en": "I'd be happy to tell you about our products. What specific information are you looking for?",
                    "es": "Me complace informarte sobre nuestros productos. ¿Qué información específica buscas?",
                    "fr": "Je serai ravi de vous parler de nos produits. Quelle information spécifique recherchez-vous?",
                    "de": "Ich informiere Sie gerne über unsere Produkte. Welche spezifischen Informationen suchen Sie?",
                    "it": "Sarò felice di parlarti dei nostri prodotti. Quali informazioni specifiche stai cercando?",
                    "pt": "Ficarei feliz em falar sobre nossos produtos. Que informação específica você está procurando?"
                },
                "unknown": {
                    "en": "I'm not sure I understand. Could you please rephrase that?",
                    "es": "No estoy seguro de entender. ¿Podrías reformular eso?",
                    "fr": "Je ne suis pas sûr de comprendre. Pourriez-vous reformuler?",
                    "de": "Ich bin mir nicht sicher, ob ich das verstehe. Könnten Sie das bitte umformulieren?",
                    "it": "Non sono sicuro di capire. Potresti riformulare?",
                    "pt": "Não tenho certeza se entendi. Você poderia reformular isso?"
                }
            }
            os.makedirs(os.path.dirname(responses_file), exist_ok=True)
            with open(responses_file, 'w', encoding='utf-8') as f:
                json.dump(default_responses, f, indent=4, ensure_ascii=False)
            return default_responses
            
        with open(responses_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _get_translation_model(self, source_lang: str, target_lang: str) -> tuple:
        """
        Get or load the translation model for the specified language pair.
        
        Args:
            source_lang (str): Source language code
            target_lang (str): Target language code
            
        Returns:
            tuple: (tokenizer, model) for the translation
        """
        if source_lang == target_lang:
            return None, None
            
        model_key = f"{source_lang}-{target_lang}"
        if model_key not in self.translation_models:
            model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
            try:
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name).to(self.device)
                self.translation_models[model_key] = (tokenizer, model)
            except Exception as e:
                print(f"Warning: Could not load translation model for {model_key}: {e}")
                return None, None
                
        return self.translation_models[model_key]
    
    def translate(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """
        Translate text from source language to target language.
        
        Args:
            text (str): Text to translate
            source_lang (str): Source language code
            target_lang (str): Target language code
            
        Returns:
            Optional[str]: Translated text, or None if translation fails
        """
        if source_lang == target_lang:
            return text
            
        tokenizer, model = self._get_translation_model(source_lang, target_lang)
        if not tokenizer or not model:
            return None
            
        try:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(**inputs)
                
            translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translated
        except Exception as e:
            print(f"Translation error: {e}")
            return None
    
    def get_response(self, intent: str, language: str = "en") -> str:
        """
        Get response for the given intent in the specified language.
        
        Args:
            intent (str): Intent to generate response for
            language (str): Target language code
            
        Returns:
            str: Response in the specified language
        """
        # Get default response if intent not found
        if intent not in self.responses:
            intent = "unknown"
            
        # Get response template
        response_templates = self.responses[intent]
        
        # If we have a template in the target language, use it
        if language in response_templates:
            return response_templates[language]
            
        # Otherwise, translate from English
        translated = self.translate(response_templates["en"], "en", language)
        return translated if translated else response_templates["en"]
        
    def get_supported_languages(self) -> list:
        """
        Get list of supported languages.
        
        Returns:
            list: List of supported language codes
        """
        # Get unique language codes from all response templates
        languages = set()
        for templates in self.responses.values():
            languages.update(templates.keys())
        return list(languages) 