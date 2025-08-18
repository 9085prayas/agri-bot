from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator

# Enforce consistent results from langdetect for reliability
DetectorFactory.seed = 0

def translate_to_english(text: str):
    """
    Detects the language of the input text and translates it to English if necessary.
    Includes checks to prevent common misdetections of English.
    """
    try:
        detected_lang = detect(text)
        
        # If the detected language is English, return immediately.
        if detected_lang == "en":
            return text, "en"
        
        # The 'langdetect' library can misclassify short English text.
        # This is a pragmatic fix to catch common false positives.
        common_misdetections = ['pl', 'so', 'ca', 'ro', 'nl', 'de']
        if detected_lang in common_misdetections:
             print(f"Warning: Language detected as '{detected_lang}', but likely English due to common misclassification. Bypassing translation.")
             return text, "en"

        print(f"Language detected: {detected_lang}. Translating to English...")
        translated_text = GoogleTranslator(source=detected_lang, target="en").translate(text)
        return translated_text, detected_lang
        
    except Exception as e:
        print(f"Language detection/translation error: {e}")
        # Default to treating the text as English if detection fails
        return text, "en"

def translate_back(text: str, target_lang: str):
    """
    Translates text from English back to the target language.
    """
    try:
        # No translation is needed if the original language was English or unknown.
        if target_lang in ["en", "unknown"]:
            return text
        
        print(f"Translating response back to {target_lang}...")
        return GoogleTranslator(source="en", target=target_lang).translate(text)
        
    except Exception as e:
        print(f"Error translating back to {target_lang}: {e}")
        return text
