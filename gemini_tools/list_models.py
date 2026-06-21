from google import genai
import os
from utils.bifrost_config import get_config
client = genai.Client(api_key=get_config('GEMINI_API_KEY'))
for m in client.models.list():
    if "flash" in m.name or "pro" in m.name:
        print(m.name)
