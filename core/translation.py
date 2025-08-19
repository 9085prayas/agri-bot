from langdetect import detect, DetectorFactory, LangDetectException
from deep_translator import GoogleTranslator

# Enforce consistent results from langdetect for reliability
DetectorFactory.seed = 0

def translate_to_english(text: str):
    """
    Detects the language of the input text and translates it to English if necessary.
    Includes improved checks to prevent misdetection of short English text.
    """
    try:
        # If the text is very short, it's often misclassified.
        # Assume it's English to prevent errors with words like "ok".
        if len(text.strip()) <= 3:
            return text, "en"

        detected_lang = detect(text)
        
        if detected_lang == "en":
            return text, "en"

        print(f"Language detected: {detected_lang}. Translating to English...")
        translated_text = GoogleTranslator(source=detected_lang, target="en").translate(text)
        return translated_text, detected_lang
        
    except LangDetectException:
        print("Language could not be detected. Defaulting to English.")
        return text, "en"
    except Exception as e:
        print(f"Language detection/translation error: {e}")
        return text, "en"

def translate_back(text: str, target_lang: str):
    """
    Translates text from English back to the target language.
    """
    try:
        if target_lang in ["en", "unknown"]:
            return text
        
        print(f"Translating response back to {target_lang}...")
        return GoogleTranslator(source="en", target=target_lang).translate(text)
        
    except Exception as e:
        print(f"Error translating back to {target_lang}: {e}")
        return text
