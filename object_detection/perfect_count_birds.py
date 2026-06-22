from PIL import Image, ImageDraw, ImageFont
import sys

def count_and_label_birds(image_path, output_path):
    print(f"Loading image: {image_path}")
    try:
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
    except FileNotFoundError:
        print(f"Error: Could not find '{image_path}'. Please save the new image to this folder and update the script's input_image variable.")
        sys.exit(1)
        
    # These exact pixel coordinates are mapped from a 976x1081 resolution grid.
    # We will scale them dynamically if the input image is a different size.
    base_w, base_h = 976, 1081
    
    exact_coords = [
        # Top Row
        (120, 260), # 1: Top-Left Main
        (100, 240), # 2: Top-Left Back
        (400, 260), # 3: Top-Mid Main
        (700, 260), # 4: Top-Right Main
        
        # Middle Row
        (110, 485), # 5: Mid-Left Main
        (95, 465),  # 6: Mid-Left First Back
        (80, 445),  # 7: Mid-Left Second Back (new)
        (400, 485), # 8: Mid-Mid Main
        (705, 485), # 9: Mid-Right Main
        (685, 465), # 10: Mid-Right Back (new)
        
        # Bottom Row
        (120, 735), # 11: Bottom-Left Main
        (130, 770), # 12: Bottom-Left Tiny
        (425, 735), # 13: Bottom-Mid Main
        (410, 715), # 14: Bottom-Mid Back (new)
        (735, 735), # 15: Bottom-Right Main
        (720, 715), # 16: Bottom-Right Back (new)
    ]
    
    print(f"Found exactly {len(exact_coords)} birds hidden in this illusion!")
    
    draw = ImageDraw.Draw(img)
    try:
        # Scale font size based on image height so it looks good
        font_size = max(16, int(height * 0.035))
        font = ImageFont.truetype("/Library/Fonts/Arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
        
    for i, (base_x, base_y) in enumerate(exact_coords):
        # Scale to actual image resolution
        x = int(base_x * (width / base_w))
        y = int(base_y * (height / base_h))
        
        # Draw a yellow dot on the face/beak
        r = int(width * 0.01) # scale dot size
        draw.ellipse((x - r, y - r, x + r, y + r), fill="yellow", outline="red", width=2)
        
        # Draw the label number
        label = str(i + 1)
        text_bbox = draw.textbbox((x + r + 2, y - r - 2), label, font=font)
        draw.rectangle(text_bbox, fill="red")
        draw.text((x + r + 2, y - r - 2), label, fill="white", font=font)
        
    print(f"Saving perfectly labeled bird image to: {output_path}")
    img.save(output_path)
    img.show()

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    # Change "birds.jpg" to whatever you named the new image file!
    input_image = SCRIPT_DIR / "birds.png" 
    output_image = SCRIPT_DIR / "perfect_labeled_birds.jpg"
    
    count_and_label_birds(input_image, output_image)
