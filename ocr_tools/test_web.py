import sys
import os
from pathlib import Path
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_DIR))

from google.cloud import vision
from utils.bifrost_config import get_config

load_dotenv()
client = vision.ImageAnnotatorClient(client_options={"api_key": get_config("GOOGLE_API_KEY")})
with open(sys.argv[1], 'rb') as f:
    image = vision.Image(content=f.read())
response = client.web_detection(image=image)
web = response.web_detection

print("Best guess labels:")
for label in web.best_guess_labels:
    print(f" - {label.label}")

print("\nPages with matching images:")
for page in web.pages_with_matching_images[:5]:
    print(f" - {page.page_title}")

print("\nEntities:")
for entity in web.web_entities[:5]:
    print(f" - {entity.description}")
