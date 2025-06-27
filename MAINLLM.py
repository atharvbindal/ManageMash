import requests
import json
import os

def generate_text_with_groq(prompt_text: str) -> str:
    groq_api_key = ""

    if not groq_api_key or groq_api_key == "YOUR_GROQ_API_KEY_HERE":
        return "Error: Groq API key not provided or placeholder still present. Please replace 'YOUR_GROQ_API_KEY_HERE' with your actual key or set it as an environment variable."

    groq_api_url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "messages": [
            {
                "role": "user",
                "content": prompt_text
            }
        ],
        "model": "llama3-8b-8192",
        "temperature": 0.7,
        "max_tokens": 500,
        "top_p": 1,
        "stream": False,
        "stop": None,
    }

    try:
        response = requests.post(groq_api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()

        response_data = response.json()

        if response_data and response_data.get("choices"):
            generated_content = response_data["choices"][0]["message"]["content"]
            return generated_content
        else:
            return "Error: No content generated or unexpected response structure from Groq API."

    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err} - Response: {response.text}"
    except requests.exceptions.ConnectionError as conn_err:
        return f"Connection error occurred: {conn_err}. Check your internet connection."
    except requests.exceptions.Timeout as timeout_err:
        return f"Timeout error occurred: {timeout_err}. The request took too long."
    except requests.exceptions.RequestException as req_err:
        return f"An unhandled requests error occurred: {req_err}"
    except json.JSONDecodeError:
        return f"Error: Could not decode JSON response from Groq API. Raw response: {response.text}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

if __name__ == "__main__":
    print("--- Testing with a simple prompt ---")
    test_prompt_1 = "What is the capital of France?"
    result_1 = generate_text_with_groq(test_prompt_1)
    print(f"Prompt: '{test_prompt_1}'")
    print(f"Response: {result_1}\n")

    print("--- Testing with a creative prompt ---")
    test_prompt_2 = "Write a short, whimsical poem about a coding bug."
    result_2 = generate_text_with_groq(test_prompt_2)
    print(f"Prompt: '{test_prompt_2}'")
    print(f"Response: {result_2}\n")

    print("--- Testing with an empty prompt (Groq might handle this differently) ---")
    test_prompt_3 = ""
    result_3 = generate_text_with_groq(test_prompt_3)
    print(f"Prompt: '{test_prompt_3}'")
    print(f"Response: {result_3}\n")
