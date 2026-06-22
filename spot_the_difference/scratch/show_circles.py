import cv2
import numpy as np

ans = cv2.imread("correct_answers/answer_07.jpg")
hsv = cv2.cvtColor(ans, cv2.COLOR_BGR2HSV)

lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 100, 100])
upper_red2 = np.array([180, 255, 255])

mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

vis = ans.copy()
for i, c in enumerate(cnts):
    area = cv2.contourArea(c)
    if area > 5:
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if r > 2:
            # Draw contour index and circle in green
            cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
            cv2.putText(vis, str(i), (int(cx) - 10, int(cy) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

cv2.imwrite("/Users/nicksng/.gemini/antigravity-ide/brain/a9d592a8-47fc-46e0-97b5-25fed2b74e19/artifacts/inspect_ans07.png", vis)
print("Saved visualization to artifacts/inspect_ans07.png")
