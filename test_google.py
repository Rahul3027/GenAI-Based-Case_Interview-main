import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print(f"API Key loaded: {bool(GOOGLE_API_KEY)}")

genai.configure(api_key=GOOGLE_API_KEY)

try:
    models = genai.list_models()
    print("Available models:")
    for model in models:
        print(f"- {model.name}")
except Exception as e:
    print(f"Error listing models: {e}")

try:
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("Hello")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error with gemini-pro: {e}")

try:
    model = genai.GenerativeModel('gemini-1.0-pro')
    response = model.generate_content("Hello")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error with gemini-1.0-pro: {e}")