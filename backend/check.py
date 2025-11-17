import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env file
load_dotenv(dotenv_path="D:/algotrade/backend/.env")

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("API key not found in .env file")

genai.configure(api_key=api_key)

# List available models
models = genai.list_models()
for model in models:
    print("Model:", model.name)
    # Safely inspect all fields
    print("Raw model info:", model)
    print("-" * 40)