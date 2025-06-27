import requests

def translate_back(text: str, target_language: str) -> str:
    """
    Translates text from English back to the specified target language
    using the MyMemory public API.

    Args:
        text (str): The English text to be translated back.
        target_language (str): The target language code (e.g., 'es', 'fr').

    Returns:
        str: The translated text in the target language, or the original text if translation fails.
    """
    try:
        if target_language == 'en':
            # If target language is English, return the text as is
            return text
            
        api_url = "https://api.mymemory.translated.net/get"
        lang_pair = f"en|{target_language}"
        params = {
            "q": text,
            "langpair": lang_pair
        }

        response = requests.get(api_url, params=params)
        response.raise_for_status()

        data = response.json()
        translated_text = data.get("responseData", {}).get("translatedText")

        if translated_text:
            print(f"Translated back to {target_language}: {translated_text}")
            return translated_text
        else:
            print("Translation back failed: No translated text found in response.")
            return text

    except requests.exceptions.RequestException as e:
        print(f"API request failed during translation back: {e}")
        return text
    except Exception as e:
        print(f"An unexpected error occurred during translation back: {e}")
        return text
