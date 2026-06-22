import torch
from transformers import pipeline
from PIL import Image

def test_owlvit_labels(image_path):
    print("Loading OwlViT...")
    detector = pipeline(task="zero-shot-object-detection", model="google/owlvit-base-patch32")
    image = Image.open(image_path).convert("RGB")
    
    labels_to_test = [
        ["rabbit head"],
        ["rabbit face"],
        ["rabbit"],
        ["animal"],
        ["white rabbit"],
        ["bunny head"]
    ]
    
    for labels in labels_to_test:
        predictions = detector(image, candidate_labels=labels)
        filtered = [p for p in predictions if p["score"] > 0.05] # Lower threshold
        print(f"Labels: {labels} -> Found {len(filtered)} at threshold 0.05")
        
        # Count at 0.1
        filtered_10 = [p for p in predictions if p["score"] > 0.1]
        print(f"Labels: {labels} -> Found {len(filtered_10)} at threshold 0.10")

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    test_owlvit_labels(SCRIPT_DIR / "image.png")
