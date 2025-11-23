# air_draw.py
import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
draw_points = []

cap = cv2.VideoCapture(0)  # if webcam fails, try 1

print("[INFO] Press 's' to save drawing, 'c' to clear, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read from webcam.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            h, w, _ = frame.shape
            cx = int(hand.landmark[8].x * w)
            cy = int(hand.landmark[8].y * h)
            draw_points.append((cx, cy))
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    # Draw stored points as white dots
    for pt in draw_points:
        cv2.circle(frame, pt, 2, (255, 255, 255), -1)

    cv2.imshow("Air Drawing", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        draw_points = []
    elif key == ord('s'):
        break
    elif key == ord('q'):
        draw_points = []
        break

cap.release()
cv2.destroyAllWindows()

if draw_points:
    np.save("draw_points.npy", np.array(draw_points, dtype=np.int32))
    print("✅ Saved draw_points.npy")
else:
    print("No drawing captured.")
