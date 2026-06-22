import os
import json
import re
from google import genai
from PIL import Image, ImageDraw, ImageFont
from utils.bifrost_config import get_config

def count_and_label_rabbits_gemini(image_path, output_path):
    api_key = get_config('GEMINI_API_KEY')
    print("Initializing Gemini API...")
    client = genai.Client(api_key=api_key)
    
    print(f"Opening image: {image_path}")
    try:
        img = Image.open(image_path)
        img_width, img_height = img.size
    except FileNotFoundError:
        print(f"Error: Could not find '{image_path}'.")
        return
        
    # We ask Gemini to locate all the rabbit faces and return a single 
    # center point (x, y) for each, normalized to a 1000x1000 grid
    prompt = """
    This is an AI-generated image containing many white rabbits overlapping each other. 
    Please carefully locate EVERY single rabbit head/face in the image (including the tiny or partially hidden ones).
    For each rabbit, output a single point (y, x) that marks the center of its face (e.g., its nose).
    Output the result STRICTLY as a JSON list of points, with coordinates normalized from 0 to 1000.
    
    Example format:
    ```json
    [
      {"y": 150, "x": 500},
      {"y": 300, "x": 450}
    ]
    ```
    """
    
    print("Sending image to Gemini 1.5 Pro for advanced detection (this may take a few seconds)...")
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[img, prompt],
    )
    
    # Extract JSON from the response
    text = response.text
    json_match = re.search(r'```(?:json)?\n(.*?)```', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Fallback if there are no markdown blocks
        json_str = text
        
    try:
        boxes = json.loads(json_str)
    except json.JSONDecodeError:
        print("Failed to parse Gemini's response as JSON. Raw response:")
        print(text)
        return
        
    print(f"\nGemini found {len(boxes)} rabbits! Drawing labels...")
    
    # Sort points top to bottom, left to right for logical numbering
    boxes.sort(key=lambda b: (b.get("y", 0), b.get("x", 0)))
    
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 36)
    except IOError:
        font = ImageFont.load_default()
        
    for i, point in enumerate(boxes):
        # Convert from 0-1000 scale back to actual image pixels
        y = int(point["y"] * img_height / 1000)
        x = int(point["x"] * img_width / 1000)
        
        # Draw a small circle at the center point
        r = 8
        draw.ellipse((x - r, y - r, x + r, y + r), fill="yellow", outline="red")
        
        # Draw label number nearby
        label = str(i + 1)
        text_bbox = draw.textbbox((x + 10, y - 20), label, font=font)
        draw.rectangle(text_bbox, fill="red")
        draw.text((x + 10, y - 20), label, fill="white", font=font)
        
    print(f"Saving labeled image to: {output_path}")
    img.save(output_path)
    img.show()

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    count_and_label_rabbits_gemini(SCRIPT_DIR / "image.png", SCRIPT_DIR / "labeled_rabbits_gemini.jpg")
