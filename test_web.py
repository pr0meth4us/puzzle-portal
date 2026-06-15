from google.cloud import vision
import sys
import os
from dotenv import load_dotenv

load_dotenv()
client = vision.ImageAnnotatorClient(client_options={"api_key": os.getenv("GOOGLE_API_KEY")})
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
