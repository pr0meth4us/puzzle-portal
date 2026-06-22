import torch
from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont
import sys

def count_and_label_rabbits(image_path, output_path):
    print("Loading the zero-shot object detection model (OwlViT)...")
    # Using OwlViT, a zero-shot object detection model from Google
    detector = pipeline(task="zero-shot-object-detection", model="google/owlvit-base-patch32")
    
    print(f"Opening image: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Could not find '{image_path}'. Please ensure the image is in the correct directory.")
        sys.exit(1)
    
    print("Detecting rabbits...")
    # We ask the model to look specifically for "rabbit"
    predictions = detector(image, candidate_labels=["rabbit"])
    
    # Filter predictions by a confidence threshold to avoid false positives
    threshold = 0.1
    filtered_predictions = [p for p in predictions if p["score"] > threshold]
    
    # Sort predictions from top to bottom, left to right for sensible numbering
    filtered_predictions.sort(key=lambda p: (p["box"]["ymin"], p["box"]["xmin"]))
    
    print(f"Found {len(filtered_predictions)} rabbits! Drawing labels...")
    
    draw = ImageDraw.Draw(image)
    
    # Try to load a larger font, fallback to default if not available
    try:
        # For macOS, standard fonts are in /Library/Fonts/
        font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 36)
    except IOError:
        font = ImageFont.load_default()
    
    for i, pred in enumerate(filtered_predictions):
        box = pred["box"]
        xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
        
        # Draw bounding box
        draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=4)
        
        # Draw label number
        label = str(i + 1)
        
        # Draw text background for better visibility
        text_bbox = draw.textbbox((xmin, ymin), label, font=font)
        draw.rectangle(text_bbox, fill="red")
        draw.text((xmin, ymin), label, fill="white", font=font)
        
    print(f"Saving labeled image to: {output_path}")
    image.save(output_path)
    image.show()

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    # Change 'rabbits.jpg' to the actual filename of the image you downloaded
    input_image = SCRIPT_DIR / "image.png" 
    output_image = SCRIPT_DIR / "labeled_rabbits.jpg"
    
    count_and_label_rabbits(input_image, output_image)
