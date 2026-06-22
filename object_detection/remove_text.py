from PIL import Image

def remove_black_text(image_path, output_path):
    print("Loading original image...")
    try:
        img = Image.open(image_path).convert("RGBA")
    except FileNotFoundError:
        print(f"Error: Could not find '{image_path}'.")
        return
        
    pixels = img.load()
    width, height = img.size
    
    print("Removing dark text pixels from the top of the image...")
    # The text is only in the top 120 pixels or so
    for y in range(min(120, height)):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            
            # If the pixel is dark (text), make it white
            # The text is black, so all RGB values will be low.
            # Rabbits and background are white/light gray (RGB values high).
            if r < 100 and g < 100 and b < 100:
                pixels[x, y] = (255, 255, 255, 255)
                
    # Also optionally clean up any stray anti-aliasing gray pixels around the text
    for y in range(min(120, height)):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if r < 200 and g < 200 and b < 200:
                pixels[x, y] = (255, 255, 255, 255)
                
    img = img.convert("RGB")
    print(f"Saving clean image without text to: {output_path}")
    img.save(output_path)

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    remove_black_text(SCRIPT_DIR / "image.png", SCRIPT_DIR / "clean_image.png")
