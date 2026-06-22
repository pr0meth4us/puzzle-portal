from PIL import Image, ImageDraw, ImageFont

def draw_grid(image_path, output_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    try:
        font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
        
    for x in range(0, width, 50):
        draw.line([(x, 0), (x, height)], fill="white", width=1)
        draw.text((x + 2, 5), str(x), fill="red", font=font)
        
    for y in range(0, height, 50):
        draw.line([(0, y), (width, y)], fill="white", width=1)
        draw.text((5, y + 2), str(y), fill="red", font=font)
        
    img.save(output_path)
    
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    draw_grid(SCRIPT_DIR / "birds.png", SCRIPT_DIR / "grid_birds.png")
