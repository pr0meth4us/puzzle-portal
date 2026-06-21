import cv2
import spot_the_differences

img = cv2.imread("validation_dataset/puzzle_04.jpg")
h, w = img.shape[:2]

# base swan is centered. The whole image is 480 wide.
# So the base swan is roughly from w//4 to w*3//4.
base = img[:240, w//4:w*3//4]

swans = [
    img[280:480, :w//2],
    img[280:480, w//2:],
    img[530:730, :w//2],
    img[530:730, w//2:]
]

total = 0
for i, swan in enumerate(swans):
    aligned_base, _, _ = spot_the_differences.align(swan, base, skip_ecc=False)
    circles, _ = spot_the_differences.detect_line(swan, aligned_base)
    print(f"Swan {i} differences: {len(circles)}")
    total += len(circles)

print("Total:", total)
