# check_models.py
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ No API Key found in .env")
else:
    try:
        # --- NEW LIBRARY SYNTAX ---
        # 1. Initialize the Client (instead of genai.configure)
        client = genai.Client(api_key=api_key)
        
        print(f"✅ Key found. Listing available models...")
        
        # 2. List models using the client
        # Note: The new SDK simplifies listing. We iterate through the result.
        for m in client.models.list():
            # Filter for models that generate content (optional, but good for clarity)
            # The new SDK model objects typically have a name like "models/gemini-1.5-flash"
            print(f"- {m.name}")
            
    except Exception as e:
        print(f"Error: {e}")