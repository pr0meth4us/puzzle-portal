import cv2
import numpy as np

ans = cv2.imread("correct_answers/answer_07.jpg")
hsv = cv2.cvtColor(ans, cv2.COLOR_BGR2HSV)
mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
mask2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
red_mask = mask1 | mask2

cnts, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
ans_drawn = ans.copy()
count = 0
for c in cnts:
    area = cv2.contourArea(c)
    if area > 10:
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if 4 < r < 40:
            count += 1
            cv2.circle(ans_drawn, (int(cx), int(cy)), int(r) + 2, (0, 255, 0), 2)
            cv2.putText(ans_drawn, str(count), (int(cx) - 10, int(cy) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

cv2.imwrite("/Users/nicksng/.gemini/antigravity-ide/brain/a9d592a8-47fc-46e0-97b5-25fed2b74e19/artifacts/inspect_ans07.png", ans_drawn)
print(f"Saved {count} circles to artifacts/inspect_ans07.png")
