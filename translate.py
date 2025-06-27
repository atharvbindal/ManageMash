import requests
from langdetect import detect, LangDetectException

def translate_text(text: str) -> tuple[str, str] | tuple[None, None]:
    """
    Detects the language of the input text and translates it to English
    using the MyMemory public API.

    Args:
        text (str): The text to be translated.

    Returns:
        tuple[str, str] | tuple[None, None]: A tuple containing the detected
        original language code (e.g., 'es', 'fr') and the translated English text.
        Returns (None, None) if detection or translation fails.
    """
    try:
        original_lang = detect(text)
        print(f"Detected Original Language: {original_lang}")

        api_url = "https://api.mymemory.translated.net/get"
        lang_pair = f"{original_lang}|en"
        params = {
            "q": text,
            "langpair": lang_pair
        }

        response = requests.get(api_url, params=params)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        data = response.json()
        translated_text = data.get("responseData", {}).get("translatedText")

        if translated_text:
            print(f"Translated Text (English): {translated_text}")
            return original_lang, translated_text
        else:
            print("Translation failed: No translated text found in response.")
            return None, None

    except LangDetectException as e:
        print(f"Language detection failed: {e}")
        return None, None
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None

# --- Test Cases ---
if __name__ == "__main__":
    # Test Case 1: Spanish to English (Active)
    print("\n--- Test Case 1 ---")
    text_to_translate = "Hola, ¿cómo estás?"
    original_language, translated_english_text = translate_text(text_to_translate)
    if original_language and translated_english_text:
        print(f"Return Value: Original Language: {original_language}, Translated Text: {translated_english_text}")

    # Test Case 2: French to English (Commented out)
    # print("\n--- Test Case 2 ---")
    # text_to_translate = "Bonjour le monde, comment ça va?"
    # original_language, translated_english_text = translate_text(text_to_translate)
    # if original_language and translated_english_text:
    #     print(f"Return Value: Original Language: {original_language}, Translated Text: {translated_english_text}")

    # Test Case 3: German to English (Commented out)
    # print("\n--- Test Case 3 ---")
    # text_to_translate = "Guten Tag, wie geht es Ihnen?"
    # original_language, translated_english_text = translate_text(text_to_translate)
    # if original_language and translated_english_text:
    #     print(f"Return Value: Original Language: {original_language}, Translated Text: {translated_english_text}")

    # Test Case 4: Japanese to English (Commented out)
    # print("\n--- Test Case 4 ---")
    # text_to_translate = "こんにちは、元気ですか？" # Konnichiwa, genki desu ka?
    # original_language, translated_english_text = translate_text(text_to_translate)
    # if original_language and translated_english_text:
    #     print(f"Return Value: Original Language: {original_language}, Translated Text: {translated_english_text}")

    # Test Case 5: English (should detect as 'en' and effectively return the same text) (Commented out)
    # print("\n--- Test Case 5 ---")
    # text_to_translate = "Hello, how are you today?"
    # original_language, translated_english_text = translate_text(text_to_translate)
    # if original_language and translated_english_text:
    #     print(f"Return Value: Original Language: {original_language}, Translated Text: {translated_english_text}")

    # Test Case 6: Empty string (should cause language detection to fail) (Commented out)
    # print("\n--- Test Case 6 ---")
    # text_to_translate = ""
    # original_language, translated_english_text = translate_text(text_to_translate)
    # if original_language is None:
    #     print("Return Value: Language detection failed as expected for empty string.")

    # Test Case 7: Text with mixed languages or insufficient data for detection (may vary) (Commented out)
    # print("\n--- Test Case 7 ---")
    # text_to_translate = "This is some text and some more text in English."
    # original_language, translated_english_text = translate_text(text_to_translate)
    # if original_language and translated_english_text:
    #     print(f"Return Value: Original Language: {original_language}, Translated Text: {translated_english_text}")

