from langdetect import detect

def detect_language(text):
    try:
        lang = detect(text)
        return lang
    except Exception as e:
        print(f"Error detecting language: {e}")
        return "unknown"