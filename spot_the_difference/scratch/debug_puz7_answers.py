import cv2
import numpy as np

def main():
    img = cv2.imread("correct_answers/answer_07.jpg")
    h, w = img.shape[:2]
    print(f"answer_07.jpg shape: {h}x{w}")
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 150, 50])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 150, 50])
    upper2 = np.array([180, 255, 255])
    
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print("Red answer circles detected:")
    for idx, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > 10:
            (cx, cy), r = cv2.minEnclosingCircle(c)
            print(f"Circle {idx}: center=({cx:.1f}, {cy:.1f}), r={r:.1f}, area={area:.1f}")

if __name__ == "__main__":
    main()
