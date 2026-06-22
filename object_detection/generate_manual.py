from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

def generate():
    img = Image.open(SCRIPT_DIR / "image.png")
    
    coords = [
        (240, 145), # 1
        (142, 125), # 2
        (325, 140), # 3
        (155, 255), # 4
        (315, 250), # 5
        (75, 355),  # 6
        (165, 370), # 7
        (245, 300), # 8
        (290, 355), # 9
        (380, 350), # 10
        (195, 485), # 11
        (265, 455), # 12
        (320, 490), # 13
        (100, 555), # 14
        (145, 575), # 15
        (175, 575), # 16 (tiny one next to 15)
        (255, 535), # 17
        (410, 545), # 18
    ]
    
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()
        
    for i, (x, y) in enumerate(coords):
        r = 6
        draw.ellipse((x - r, y - r, x + r, y + r), fill="yellow", outline="red")
        
        label = str(i + 1)
        text_bbox = draw.textbbox((x + 8, y - 12), label, font=font)
        draw.rectangle(text_bbox, fill="red")
        draw.text((x + 8, y - 12), label, fill="white", font=font)
        
    img.save(SCRIPT_DIR / "temp.png")

if __name__ == "__main__":
    generate()
