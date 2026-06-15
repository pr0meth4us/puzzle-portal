from google import genai
import os
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
for m in client.models.list():
    if "flash" in m.name or "pro" in m.name:
        print(m.name)
