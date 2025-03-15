<<<<<<< HEAD
# nxtgenIntellicxsupportAPI
=======
# Multilingual Customer Support System

A powerful automated customer support system that can handle queries in 80+ languages using state-of-the-art machine learning models.

## Features

- Language detection and translation for 80+ languages
- Intent classification for customer queries
- Contextual response generation
- Automatic translation of responses back to the user's language
- RESTful API interface
- Scalable architecture

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)
- Redis
- PostgreSQL

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd multilingual-customer-support
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy the example environment file and update it with your settings:
```bash
cp config/.env.example config/.env
```

5. Update the `.env` file with your:
   - OpenAI API key
   - Database credentials
   - Redis configuration
   - Other settings as needed

## Usage

1. Start the API server:
```bash
python src/backend/main.py
```

2. The API will be available at `http://localhost:8000`

3. API Documentation will be available at `http://localhost:8000/docs`

## API Endpoints

### POST /process_query

Process a customer query in any supported language.

Request body:
```json
{
    "text": "Comment puis-je réinitialiser mon mot de passe?",
    "session_id": "optional-session-id",
    "context": [
        {
            "user": "Previous user message",
            "assistant": "Previous assistant response"
        }
    ]
}
```

Response:
```json
{
    "original_text": "Comment puis-je réinitialiser mon mot de passe?",
    "detected_language": "fr_XX",
    "translated_text": "How can I reset my password?",
    "intent": "technical_support",
    "intent_confidence": 0.95,
    "response": "I can help you reset your password. Please follow these steps...",
    "translated_response": "Je peux vous aider à réinitialiser votre mot de passe. Veuillez suivre ces étapes..."
}
```

### GET /health

Check the health status of the API.

## Architecture

The system consists of several components:

1. **Language Processor**: Handles language detection and translation
2. **Intent Classifier**: Determines the user's intent from their query
3. **Response Generator**: Generates contextual responses using GPT-4
4. **FastAPI Backend**: Coordinates all components and provides the API interface

## Model Information

- Language Detection: XLM-RoBERTa Base
- Translation: mBART-50 Many-to-Many
- Intent Classification: BART Large MNLI
- Response Generation: GPT-4

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
>>>>>>> c2b11cb (Initial commit)
