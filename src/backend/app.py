from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, List
import sys
import os
import logging
import traceback
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(src_dir)
logger.info(f"Added {src_dir} to Python path")

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

# Mock data for responses
SUPPORTED_LANGUAGES = ["en", "es", "fr", "de", "it", "pt"]

def detect_language_from_text(text: str) -> str:
    """
    Simple language detection based on common words and patterns.
    Returns language code: en, es, fr, de, it, pt
    """
    text_lower = text.lower()
    
    # Spanish indicators
    spanish_words = ['hola', 'gracias', 'por favor', 'adiós', 'sí', 'no', 'ayuda', 'necesito', 'información']
    spanish_patterns = ['ñ', '¿', '¡', 'qué', 'cómo', 'dónde', 'cuándo']
    
    # French indicators
    french_words = ['bonjour', 'merci', 's\'il vous plaît', 'au revoir', 'oui', 'non', 'aide', 'besoin']
    french_patterns = ['é', 'è', 'ê', 'à', 'ç', 'ù', 'ô']
    
    # German indicators
    german_words = ['hallo', 'danke', 'bitte', 'auf wiedersehen', 'ja', 'nein', 'hilfe', 'brauche']
    german_patterns = ['ä', 'ö', 'ü', 'ß']
    
    # Italian indicators
    italian_words = ['ciao', 'grazie', 'per favore', 'arrivederci', 'sì', 'no', 'aiuto', 'ho bisogno']
    italian_patterns = ['è', 'é', 'à', 'ò', 'ù']
    
    # Portuguese indicators
    portuguese_words = ['olá', 'obrigado', 'por favor', 'adeus', 'sim', 'não', 'ajuda', 'preciso']
    portuguese_patterns = ['ã', 'õ', 'ç', 'á', 'é', 'í', 'ó', 'ú']
    
    # Count matches
    spanish_score = sum(1 for word in spanish_words if word in text_lower) + sum(1 for pattern in spanish_patterns if pattern in text_lower)
    french_score = sum(1 for word in french_words if word in text_lower) + sum(1 for pattern in french_patterns if pattern in text_lower)
    german_score = sum(1 for word in german_words if word in text_lower) + sum(1 for pattern in german_patterns if pattern in text_lower)
    italian_score = sum(1 for word in italian_words if word in text_lower) + sum(1 for pattern in italian_patterns if pattern in text_lower)
    portuguese_score = sum(1 for word in portuguese_words if word in text_lower) + sum(1 for pattern in portuguese_patterns if pattern in text_lower)
    
    # Return language with highest score, default to English
    scores = {
        'es': spanish_score,
        'fr': french_score,
        'de': german_score,
        'it': italian_score,
        'pt': portuguese_score
    }
    
    max_score = max(scores.values())
    if max_score > 0:
        detected = [lang for lang, score in scores.items() if score == max_score][0]
        logger.info(f"Detected language: {detected} (score: {max_score})")
        return detected
    
    # Default to English
    return "en"
SUPPORTED_INTENTS = ["greeting", "farewell", "help", "product_info", "pricing", "contact", "technical_support", "unknown"]
MOCK_RESPONSES = {
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
    "pricing": {
        "en": "Our pricing plans start at $9.99/month for the basic package. We also offer premium plans at $19.99/month and enterprise solutions starting at $49.99/month. Would you like more details about any specific plan?",
        "es": "Nuestros planes de precios comienzan en $9.99/mes para el paquete básico. También ofrecemos planes premium a $19.99/mes y soluciones empresariales a partir de $49.99/mes. ¿Te gustaría más detalles sobre algún plan específico?",
        "fr": "Nos forfaits commencent à 9,99$/mois pour le forfait de base. Nous proposons également des forfaits premium à 19,99$/mois et des solutions d'entreprise à partir de 49,99$/mois. Souhaitez-vous plus de détails sur un forfait spécifique?",
        "de": "Unsere Preispläne beginnen bei 9,99$/Monat für das Basispaket. Wir bieten auch Premium-Pläne für 19,99$/Monat und Unternehmenslösungen ab 49,99$/Monat an. Möchten Sie weitere Details zu einem bestimmten Plan?",
        "it": "I nostri piani tariffari partono da $9,99/mese per il pacchetto base. Offriamo anche piani premium a $19,99/mese e soluzioni aziendali a partire da $49,99/mese. Desideri maggiori dettagli su un piano specifico?",
        "pt": "Nossos planos de preços começam em $9,99/mês para o pacote básico. Também oferecemos planos premium a $19,99/mês e soluções empresariais a partir de $49,99/mês. Gostaria de mais detalhes sobre algum plano específico?"
    },
    "contact": {
        "en": "You can reach our customer support team at support@example.com or call us at 1-800-123-4567. Our support hours are Monday to Friday, 9 AM to 6 PM EST.",
        "es": "Puedes contactar a nuestro equipo de atención al cliente en support@example.com o llamarnos al 1-800-123-4567. Nuestro horario de atención es de lunes a viernes, de 9 AM a 6 PM EST.",
        "fr": "Vous pouvez joindre notre équipe d'assistance à support@example.com ou nous appeler au 1-800-123-4567. Nos heures d'assistance sont du lundi au vendredi, de 9h à 18h EST.",
        "de": "Sie können unser Kundendienstteam unter support@example.com erreichen oder uns unter 1-800-123-4567 anrufen. Unsere Supportzeiten sind Montag bis Freitag, 9 bis 18 Uhr EST.",
        "it": "Puoi contattare il nostro team di supporto clienti all'indirizzo support@example.com o chiamarci al numero 1-800-123-4567. I nostri orari di supporto sono dal lunedì al venerdì, dalle 9 alle 18 EST.",
        "pt": "Você pode entrar em contato com nossa equipe de suporte ao cliente em support@example.com ou nos ligar em 1-800-123-4567. Nosso horário de suporte é de segunda a sexta-feira, das 9h às 18h EST."
    },
    "technical_support": {
        "en": "For technical issues, please try restarting the application first. If the problem persists, check our knowledge base at help.example.com or contact our technical support team at techsupport@example.com with details about your issue.",
        "es": "Para problemas técnicos, intenta reiniciar la aplicación primero. Si el problema persiste, consulta nuestra base de conocimientos en help.example.com o contacta a nuestro equipo de soporte técnico en techsupport@example.com con detalles sobre tu problema.",
        "fr": "Pour les problèmes techniques, veuillez d'abord essayer de redémarrer l'application. Si le problème persiste, consultez notre base de connaissances sur help.example.com ou contactez notre équipe de support technique à techsupport@example.com avec les détails de votre problème.",
        "de": "Bei technischen Problemen versuchen Sie bitte zunächst, die Anwendung neu zu starten. Wenn das Problem weiterhin besteht, schauen Sie in unsere Wissensdatenbank unter help.example.com oder kontaktieren Sie unser technisches Support-Team unter techsupport@example.com mit Details zu Ihrem Problem.",
        "it": "Per problemi tecnici, prova prima a riavviare l'applicazione. Se il problema persiste, consulta la nostra knowledge base su help.example.com o contatta il nostro team di supporto tecnico all'indirizzo techsupport@example.com con i dettagli del tuo problema.",
        "pt": "Para problemas técnicos, tente reiniciar o aplicativo primeiro. Se o problema persistir, verifique nossa base de conhecimento em help.example.com ou entre em contato com nossa equipe de suporte técnico em techsupport@example.com com detalhes sobre seu problema."
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
        logger.info(f"Processing query: {request.text}")
        
        # Detect language from input text
        detected_language = request.preferred_language if request.preferred_language else detect_language_from_text(request.text)
        language_confidence = 1.0
        
        # Improved intent detection based on keywords
        text_lower = request.text.lower()
        
        # Define keyword patterns for each intent
        intent_patterns = {
            "greeting": ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening", 
                         "hola", "bonjour", "ciao", "hallo", "howdy", "what's up", "sup"],
            "farewell": ["goodbye", "bye", "see you", "farewell", "adios", "au revoir", "cya", "take care", 
                         "have a nice day", "until next time", "talk to you later", "ttyl"],
            "help": ["help", "assist", "support", "guidance", "need help", "can you help", "how do i", 
                     "how to", "having trouble", "problem with", "issue with", "question about", "confused about"],
            "product_info": ["product", "service", "offer", "sell", "feature", "plan", 
                            "subscription", "package", "demo", "information about", "tell me about", 
                            "what is", "how does", "how do you", "what are your"],
            "pricing": ["price", "cost", "pricing", "fee", "subscription cost", "how much", "discount", 
                       "trial", "free trial", "payment", "pay", "affordable", "expensive", "cheap", "premium"],
            "contact": ["contact", "email", "phone", "call", "reach", "talk to", "speak with", "chat with", 
                       "customer service", "representative", "agent", "human", "person", "manager"],
            "technical_support": ["technical", "tech support", "bug", "error", "issue", "problem", "not working", 
                                 "broken", "fix", "repair", "troubleshoot", "crash", "glitch", "malfunction"]
        }
        
        # Check for intent matches
        matched_intent = "unknown"
        max_matches = 0
        intent_confidence = 0.5
        
        for intent, keywords in intent_patterns.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            # Also check for partial matches
            partial_matches = sum(1 for keyword in keywords if any(word.startswith(keyword) for word in text_lower.split()))
            
            total_matches = matches + (partial_matches * 0.5)  # Give partial matches half weight
            
            if total_matches > max_matches:
                max_matches = total_matches
                matched_intent = intent
                # Calculate confidence based on number of matches
                intent_confidence = min(0.5 + (total_matches * 0.1), 0.95)
        
        # If no matches found but text is short, assume it's a greeting
        if matched_intent == "unknown" and len(text_lower.split()) <= 3:
            matched_intent = "greeting"
            intent_confidence = 0.6
        
        # Get response
        response = MOCK_RESPONSES.get(matched_intent, MOCK_RESPONSES["unknown"])
        response_text = response.get(detected_language, response["en"])
        
        # Add variety to responses
        if matched_intent == "greeting":
            greeting_variations = [
                "Hello! How can I assist you today?",
                "Hi there! What can I help you with?",
                "Welcome! How may I be of service?",
                "Greetings! What brings you here today?",
                "Hello! I'm your virtual assistant. How can I help?"
            ]
            if detected_language == "en":
                import random
                response_text = random.choice(greeting_variations)
        elif matched_intent == "farewell":
            farewell_variations = [
                "Goodbye! Have a great day!",
                "Farewell! Feel free to come back if you have more questions.",
                "Take care! It was nice chatting with you.",
                "Goodbye! I hope I was able to help you today.",
                "See you next time! Have a wonderful day ahead."
            ]
            if detected_language == "en":
                import random
                response_text = random.choice(farewell_variations)
        elif matched_intent == "help":
            help_variations = [
                "I'm here to help! Please let me know what you need assistance with.",
                "I'd be happy to help you. What specific issue are you facing?",
                "How can I assist you today? Please provide more details about your question.",
                "I'm ready to help! Could you tell me more about what you need?",
                "I'm here to provide support. What can I help you with specifically?"
            ]
            if detected_language == "en":
                import random
                response_text = random.choice(help_variations)
        elif matched_intent == "product_info":
            product_variations = [
                "I'd be happy to tell you about our products. What specific information are you looking for?",
                "Our products offer a range of features. What would you like to know more about?",
                "We have several products that might interest you. Any specific aspect you'd like to explore?",
                "I can provide information about our products and services. What details are you interested in?",
                "Our product lineup is designed to meet various needs. What particular information would help you?"
            ]
            if detected_language == "en":
                import random
                response_text = random.choice(product_variations)
        elif matched_intent == "pricing":
            pricing_variations = [
                "Our pricing plans start at $9.99/month for the basic package. We also offer premium plans at $19.99/month and enterprise solutions starting at $49.99/month. Would you like more details about any specific plan?",
                "Our pricing plans are designed to fit your needs. What specific pricing plan are you interested in?",
                "We offer a range of pricing options. What would you like to know more about our pricing?",
                "Our pricing is competitive. What specific pricing details are you interested in?",
                "We have several pricing options. What particular pricing information would help you?"
            ]
            if detected_language == "en":
                import random
                response_text = random.choice(pricing_variations)
        elif matched_intent == "contact":
            contact_variations = [
                "You can reach our customer support team at support@example.com or call us at 1-800-123-4567. Our support hours are Monday to Friday, 9 AM to 6 PM EST.",
                "Our customer support team is available to assist you. How can we help you reach us?",
                "We're here to help you. What's the best way to contact you?",
                "Our contact information is always available. How would you like to reach us?",
                "We're always ready to assist you. How can we help you get in touch with us?"
            ]
            if detected_language == "en":
                import random
                response_text = random.choice(contact_variations)
        elif matched_intent == "technical_support":
            technical_support_variations = [
                "For technical issues, please try restarting the application first. If the problem persists, check our knowledge base at help.example.com or contact our technical support team at techsupport@example.com with details about your issue.",
                "We're here to help you with any technical issues you're facing. What's the best way to assist you?",
                "Our technical support team is available to assist you. How can we help you with your technical issue?",
                "We're always ready to assist you. How can we help you with your technical issue?",
                "We're here to help you. How can we assist you with your technical issue?"
            ]
            if detected_language == "en":
                import random
                response_text = random.choice(technical_support_variations)
        elif matched_intent == "unknown":
            unknown_variations = [
                "I'm not sure I understand. Could you please rephrase that?",
                "I didn't quite catch that. Can you explain in different words?",
                "I'm having trouble understanding your request. Could you provide more details?",
                "Could you clarify what you're looking for? I want to make sure I help you correctly.",
                "I'm not sure what you're asking for. Could you try asking in a different way?"
            ]
            if detected_language == "en":
                import random
                response_text = random.choice(unknown_variations)
        
        result = {
            "query": request.text,
            "detected_language": detected_language,
            "language_confidence": language_confidence,
            "intent": matched_intent,
            "intent_confidence": intent_confidence,
            "response": response_text
        }
        
        logger.info(f"Query processed successfully: {result}")
        return result
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/supported_languages", response_model=List[str])
async def get_supported_languages() -> List[str]:
    """
    Get list of supported languages.
    """
    logger.info("Getting supported languages")
    return SUPPORTED_LANGUAGES

@app.get("/supported_intents", response_model=List[str])
async def get_supported_intents() -> List[str]:
    """
    Get list of supported intents.
    """
    logger.info("Getting supported intents")
    return SUPPORTED_INTENTS

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "service_initialized": True
    }

# Serve frontend static files for cloud deployment
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "src", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
    
    @app.get("/")
    async def serve_frontend():
        """Serve the main frontend page"""
        index_path = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"message": "Frontend not found"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)