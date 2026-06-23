import os
import sys
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.bifrost_config import get_config

load_dotenv(Path(__file__).resolve().parents[2] / '.env')
GEMINI_API_KEY = get_config('GEMINI_API_KEY')
client = genai.Client(api_key=GEMINI_API_KEY)

def describe_crop(img_path):
    with open(img_path, 'rb') as f:
        data = f.read()
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            "This is a crop from a spot-the-difference puzzle. "
            "The left half is from the first image, and the right half is from the second image. "
            "Please describe what is shown (e.g. text, cloud, bird, spider, etc.) and state clearly "
            "if there is a difference between the left and right halves.",
            types.Part.from_bytes(data=data, mime_type="image/png")
        ]
    )
    return response.text

def main():
    crops_dir = Path("scratch/crops")
    files = sorted([f for f in os.listdir(crops_dir) if f.startswith("crop_")])
    
    out_path = Path("scratch/crop_descriptions.txt")
    with open(out_path, "w") as out:
        for f in files:
            path = crops_dir / f
            print(f"Describing {f}...")
            desc = describe_crop(path)
            out.write(f"=== {f} ===\n{desc.strip()}\n\n")
            out.flush()
    print("Done! Descriptions saved to scratch/crop_descriptions.txt")

if __name__ == '__main__':
    main()
