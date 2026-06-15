import os
import sys
from dotenv import load_dotenv
from ocr_tools.lens import get_google_lens_context, upload_to_temp_host
from serpapi import GoogleSearch

load_dotenv()

with open(sys.argv[1], 'rb') as f:
    image_bytes = f.read()

image_url = upload_to_temp_host(image_bytes)
print("Uploaded to:", image_url)

params = {
    "engine": "google_lens",
    "url": image_url,
    "api_key": os.getenv("SERPAPI_KEY")
}

search = GoogleSearch(params)
results = search.get_dict()

for idx, match in enumerate(results.get("visual_matches", [])[:5]):
    print(f"Match {idx}: {match.get('title')}")
    print(f"Thumbnail: {match.get('thumbnail')}")
