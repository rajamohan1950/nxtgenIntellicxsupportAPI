from typing import Dict, List, Optional, Tuple

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    MBartForConditionalGeneration,
    MBartTokenizer,
    pipeline,
)

class LanguageProcessor:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._init_language_detection()
        self._init_translation()
        
    def _init_language_detection(self):
        """Initialize the language detection model"""
        model_name = "facebook/xlm-roberta-base"
        self.lang_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.lang_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.lang_model.to(self.device)
        
    def _init_translation(self):
        """Initialize the translation model"""
        model_name = "facebook/mbart-large-50-many-to-many-mmt"
        self.trans_tokenizer = MBartTokenizer.from_pretrained(model_name)
        self.trans_model = MBartForConditionalGeneration.from_pretrained(model_name)
        self.trans_model.to(self.device)
        
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect the language of the input text
        Returns: Tuple of (language_code, confidence)
        """
        inputs = self.lang_tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        outputs = self.lang_model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        lang_id = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][lang_id].item()
        
        return self.lang_tokenizer.decode(lang_id), confidence
        
    def translate(
        self, text: str, source_lang: str, target_lang: str = "en_XX"
    ) -> str:
        """
        Translate text from source language to target language
        Default target is English (en_XX)
        """
        self.trans_tokenizer.src_lang = source_lang
        encoded = self.trans_tokenizer(text, return_tensors="pt").to(self.device)
        
        generated_tokens = self.trans_model.generate(
            **encoded,
            forced_bos_token_id=self.trans_tokenizer.lang_code_to_id[target_lang]
        )
        
        return self.trans_tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0]
        
    def process_input(self, text: str) -> Dict:
        """
        Process input text: detect language and translate if not English
        """
        detected_lang, confidence = self.detect_language(text)
        result = {
            "original_text": text,
            "detected_language": detected_lang,
            "confidence": confidence,
            "translated_text": None
        }
        
        if detected_lang != "en_XX":
            result["translated_text"] = self.translate(
                text, source_lang=detected_lang
            )
            
        return result 