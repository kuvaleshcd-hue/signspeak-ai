"""Heuristic ASL static-letter recognizer using MediaPipe's 21 hand landmarks.

This is a prototype, not a substitute for a trained ASL model. It supports a
limited set of easy-to-describe poses and returns UNKNOWN for uncertain cases.
Landmarks are normalized to image coordinates. Fingers are indexed:
thumb 1-4, index 5-8, middle 9-12, ring 13-16, pinky 17-20.
"""
import math

def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def _finger_extended(p, tip, pip, mcp):
    # For a roughly upright hand in the image, fingertips sit above joints.
    # This image-axis heuristic is imperfect for rotated/sideways hands.
    return p[tip][1] < p[pip][1] < p[mcp][1]

def classify_landmarks(points):
    """Return (label, confidence, explanation) from 21 [x,y,z] points."""
    if points is None or len(points) != 21:
        return "UNKNOWN", 0.0, "No hand detected"

    p = points
    index = _finger_extended(p, 8, 6, 5)
    middle = _finger_extended(p, 12, 10, 9)
    ring = _finger_extended(p, 16, 14, 13)
    pinky = _finger_extended(p, 20, 18, 17)

    # Thumb extension estimated by distance from thumb tip to index MCP.
    thumb_open = _dist(p[4], p[5]) > _dist(p[3], p[5]) * 1.15
    extended = [index, middle, ring, pinky]
    n = sum(extended)

    # Basic shape-based starter rules; exact ASL distinctions need trained data.
    if n == 4 and thumb_open:
        return "B", 0.62, "Four fingers extended"
    if n == 1 and index and not middle and not ring and not pinky and not thumb_open:
        return "D", 0.48, "Index finger extended"
    if n == 2 and index and middle and not ring and not pinky:
        return "V", 0.55, "Index and middle extended"
    if n == 3 and index and middle and ring and not pinky:
        return "W", 0.50, "Three fingers extended"
    if n == 0 and not thumb_open:
        # A and E/S/T can look similar in coarse landmarks; don't guess.
        return "UNKNOWN", 0.25, "Closed hand: A/E/S/T need a trained classifier"
    if n == 0 and thumb_open:
        return "A_OR_S", 0.30, "Closed fingers; thumb position ambiguous"
    if n == 4 and not thumb_open:
        return "UNKNOWN", 0.30, "Open fingers detected; thumb pose ambiguous"
    return "UNKNOWN", 0.20, "Pose not confidently matched"
