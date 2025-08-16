from langdetect import detect
from deep_translator import GoogleTranslator

def translate_to_english(text: str):
    try:
        detected_lang = detect(text)
        if detected_lang == "en":
            return text, "en"
        translated_text = GoogleTranslator(source=detected_lang, target="en").translate(text)
        return translated_text, detected_lang
    except Exception:
        return text, "unknown"

def translate_back(text: str, target_lang: str):
    try:
        if target_lang == "en":
            return text
        return GoogleTranslator(source="en", target=target_lang).translate(text)
    except Exception:
        return text
