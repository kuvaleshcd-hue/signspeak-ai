import numpy as np
import mediapipe as mp
import cv2

def landmarks_from_rgb(image_rgb):
    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.55,
    ) as hands:
        result = hands.process(image_rgb)
    if not result.multi_hand_landmarks:
        return None
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in result.multi_hand_landmarks[0].landmark],
        dtype=float,
    )

def landmarks_from_bgr(frame):
    return landmarks_from_rgb(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
