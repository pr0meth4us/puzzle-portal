from PIL import Image
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

def verify():
    img = Image.open(SCRIPT_DIR / "birds.png")
    
    exact_coords = [
        (120, 260), # 1: Top-Left Main
        (100, 240), # 2: Top-Left Back
        (400, 260), # 3: Top-Mid Main
        (700, 260), # 4: Top-Right Main
        
        (110, 485), # 5: Mid-Left Main
        (95, 465),  # 6: Mid-Left First Back
        (75, 445),  # 7: Mid-Left Second Back (new)
        (400, 485), # 8: Mid-Mid Main
        (705, 485), # 9: Mid-Right Main
        (680, 460), # 10: Mid-Right Back (new)
        
        (120, 735), # 11: Bottom-Left Main
        (130, 770), # 12: Bottom-Left Tiny
        (425, 735), # 13: Bottom-Mid Main
        (405, 715), # 14: Bottom-Mid Back (new)
        (735, 735), # 15: Bottom-Right Main
        (715, 715), # 16: Bottom-Right Back (new)
    ]
    
    canvas = Image.new("RGB", (800, 800))
    
    for i, (x, y) in enumerate(exact_coords):
        patch = img.crop((x - 20, y - 20, x + 20, y + 20))
        patch.putpixel((20, 20), (255, 0, 0))
        patch.putpixel((20, 21), (255, 0, 0))
        patch.putpixel((21, 20), (255, 0, 0))
        patch.putpixel((21, 21), (255, 0, 0))
        
        row = i // 4
        col = i % 4
        canvas.paste(patch, (col * 50, row * 50))
        
    canvas.save(SCRIPT_DIR / "patches3.png")

if __name__ == "__main__":
    verify()
