import numpy as np
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def detect_hand(image_rgb):
    """
    Process an RGB image and detect hand landmarks.
    
    Returns:
        landmarks: np.ndarray of shape (21, 3) or None if no hand detected.
        handedness: "Right" or "Left" (or None).
        annotated_image: RGB image with drawn skeleton landmarks and bounding box.
    """
    if image_rgb is None or image_rgb.size == 0:
        return None, None, image_rgb

    h, w = image_rgb.shape[:2]
    annotated = image_rgb.copy()

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.50,
        min_tracking_confidence=0.50,
    ) as hands:
        result = hands.process(image_rgb)

    if not result.multi_hand_landmarks:
        return None, None, annotated

    hand_landmarks = result.multi_hand_landmarks[0]
    
    # Handedness label
    handedness = "Right"
    if result.multi_handedness:
        handedness = result.multi_handedness[0].classification[0].label

    # Draw skeleton connections and joint landmarks
    mp_draw.draw_landmarks(
        annotated,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_styles.get_default_hand_landmarks_style(),
        mp_styles.get_default_hand_connections_style(),
    )

    # Convert to numpy array of normalized (x, y, z)
    landmarks = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=float,
    )

    # Draw bounding box and handedness badge
    try:
        xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
        ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
        x_min, x_max = max(0, min(xs) - 15), min(w - 1, max(xs) + 15)
        y_min, y_max = max(0, min(ys) - 15), min(h - 1, max(ys) + 15)

        # Subtle bounding box in emerald green
        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (34, 197, 94), 2)
        
        # Handedness badge header
        badge_text = f"{handedness} Hand"
        cv2.rectangle(annotated, (x_min, max(0, y_min - 24)), (x_min + 110, y_min), (34, 197, 94), -1)
        cv2.putText(
            annotated,
            badge_text,
            (x_min + 6, max(12, y_min - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    except Exception:
        pass

    return landmarks, handedness, annotated

def landmarks_from_rgb(image_rgb):
    """Backwards-compatible landmark extractor."""
    landmarks, _, _ = detect_hand(image_rgb)
    return landmarks

def landmarks_from_bgr(frame):
    """Backwards-compatible BGR frame landmark extractor."""
    return landmarks_from_rgb(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
