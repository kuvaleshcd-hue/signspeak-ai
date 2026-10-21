"""Robust heuristic ASL and hand sign recognizer using MediaPipe's 21 hand landmarks.

Supports standard ASL letters: A, B, C, D, E, F, I, L, O, U, V, W, Y
and common expressive gestures: ILY (🤟), THUMBS_UP (👍), PEACE (✌️), OPEN_PALM (🖐️), FIST (✊).
"""
import math
import numpy as np

# Metadata dictionary for all supported signs
SUPPORTED_SIGNS = {
    "A": {
        "name": "Letter A",
        "emoji": "✊",
        "char": "A",
        "category": "letter",
        "description": "Closed fist with fingers curled and thumb resting upright alongside index finger.",
    },
    "B": {
        "name": "Letter B",
        "emoji": "✋",
        "char": "B",
        "category": "letter",
        "description": "Four fingers straight up together, thumb folded across palm.",
    },
    "C": {
        "name": "Letter C",
        "emoji": "🤏",
        "char": "C",
        "category": "letter",
        "description": "Fingers and thumb curved in an open 'C' shape.",
    },
    "D": {
        "name": "Letter D",
        "emoji": "☝️",
        "char": "D",
        "category": "letter",
        "description": "Index finger extended upright, thumb touching curled middle/ring fingers.",
    },
    "E": {
        "name": "Letter E",
        "emoji": "✊",
        "char": "E",
        "category": "letter",
        "description": "All four fingers curled tightly down over the folded thumb.",
    },
    "F": {
        "name": "Letter F (OK)",
        "emoji": "👌",
        "char": "F",
        "category": "letter",
        "description": "Thumb and index tips touching in a circle; middle, ring, pinky extended upright.",
    },
    "I": {
        "name": "Letter I",
        "emoji": "🤙",
        "char": "I",
        "category": "letter",
        "description": "Pinky finger extended upright; thumb folded over curled fingers.",
    },
    "L": {
        "name": "Letter L",
        "emoji": "👆",
        "char": "L",
        "category": "letter",
        "description": "Index finger extended upright, thumb extended horizontally at ~90 degrees.",
    },
    "O": {
        "name": "Letter O",
        "emoji": "⭕",
        "char": "O",
        "category": "letter",
        "description": "All fingertips curved down to meet thumb tip, forming an 'O' circle.",
    },
    "U": {
        "name": "Letter U",
        "emoji": "✌️",
        "char": "U",
        "category": "letter",
        "description": "Index and middle fingers extended straight up together (touching).",
    },
    "V": {
        "name": "Letter V (Peace)",
        "emoji": "✌️",
        "char": "V",
        "category": "letter",
        "description": "Index and middle fingers extended straight up and spread apart in a 'V'.",
    },
    "W": {
        "name": "Letter W",
        "emoji": "🖖",
        "char": "W",
        "category": "letter",
        "description": "Three fingers extended: index, middle, and ring fingers spread upright.",
    },
    "Y": {
        "name": "Letter Y",
        "emoji": "🤙",
        "char": "Y",
        "category": "letter",
        "description": "Thumb and pinky extended outward; index, middle, ring curled into palm.",
    },
    "ILY": {
        "name": "I Love You",
        "emoji": "🤟",
        "char": "🤟",
        "category": "gesture",
        "description": "Thumb, index finger, and pinky finger extended simultaneously.",
    },
    "THUMBS_UP": {
        "name": "Thumbs Up",
        "emoji": "👍",
        "char": "👍",
        "category": "gesture",
        "description": "Fist with thumb pointing straight up.",
    },
    "OPEN_PALM": {
        "name": "Open Palm (Stop)",
        "emoji": "🖐️",
        "char": " ",
        "category": "gesture",
        "description": "All five digits (fingers and thumb) fully extended and spread out.",
    },
    "FIST": {
        "name": "Fist",
        "emoji": "✊",
        "char": "✊",
        "category": "gesture",
        "description": "All fingers and thumb curled tightly into palm.",
    },
}

def _dist(a, b):
    """Euclidean distance in 2D image coordinates."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def _dist3d(a, b):
    """Euclidean distance in 3D coordinates."""
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def analyze_landmarks(points, handedness="Right"):
    """Extract normalized geometric features from 21 MediaPipe hand points."""
    if points is None or len(points) != 21:
        return None

    p = points
    # Reference palm size: distance from wrist (0) to middle MCP (9)
    palm_size = _dist(p[0], p[9])
    if palm_size < 1e-4:
        return None

    def d_norm(i, j):
        return _dist(p[i], p[j]) / palm_size

    # Check finger extension for index (5-8), middle (9-12), ring (13-16), pinky (17-20)
    def finger_extension(tip, dip, pip, mcp):
        tip_wrist = d_norm(tip, 0)
        pip_wrist = d_norm(pip, 0)
        tip_mcp = d_norm(tip, mcp)

        # For upright hand: fingertip y < pip y < mcp y (y=0 is top)
        upright_extended = (p[tip][1] < p[pip][1] < p[mcp][1])
        # Radial distance check (works even when tilted)
        radial_extended = (tip_wrist > pip_wrist * 1.12 and tip_mcp > 0.65)
        # Curled check
        is_curled = (tip_mcp < 0.52) or (p[tip][1] > p[pip][1] and tip_wrist < pip_wrist * 1.05)

        is_ext = (upright_extended or radial_extended) and not (is_curled and not upright_extended)
        return is_ext, is_curled, tip_mcp, tip_wrist

    idx_ext, idx_curl, idx_mcp, idx_wrist = finger_extension(8, 7, 6, 5)
    mid_ext, mid_curl, mid_mcp, mid_wrist = finger_extension(12, 11, 10, 9)
    rng_ext, rng_curl, rng_mcp, rng_wrist = finger_extension(16, 15, 14, 13)
    pnk_ext, pnk_curl, pnk_mcp, pnk_wrist = finger_extension(20, 19, 18, 17)

    # Thumb analysis
    thumb_tip_wrist = d_norm(4, 0)
    thumb_tip_index_mcp = d_norm(4, 5)
    thumb_tip_pinky_mcp = d_norm(4, 17)
    thumb_tip_index_tip = d_norm(4, 8)
    thumb_tip_middle_tip = d_norm(4, 12)
    thumb_tip_ring_tip = d_norm(4, 16)
    thumb_tip_pinky_tip = d_norm(4, 20)

    # Thumb pointing up (tip y above IP and MCP, and vertically oriented)
    thumb_dy = p[2][1] - p[4][1]
    thumb_dx = abs(p[4][0] - p[2][0])
    thumb_pointing_up = (p[4][1] < p[3][1] < p[2][1]) and (thumb_dy > thumb_dx * 0.75)
    # Thumb extended outward laterally away from index MCP
    thumb_extended_out = (thumb_tip_index_mcp > 0.65 and d_norm(4, 2) > d_norm(3, 2))
    # Thumb tucked across palm
    thumb_tucked = (thumb_tip_index_mcp <= 0.60 and not thumb_pointing_up)

    # Tip-to-tip distances
    index_middle_dist = d_norm(8, 12)
    middle_ring_dist = d_norm(12, 16)
    ring_pinky_dist = d_norm(16, 20)

    # Touch checks
    thumb_index_touch = (thumb_tip_index_tip < 0.35)
    thumb_middle_touch = (thumb_tip_middle_tip < 0.45)

    return {
        "palm_size": palm_size,
        "index_ext": idx_ext,
        "middle_ext": mid_ext,
        "ring_ext": rng_ext,
        "pinky_ext": pnk_ext,
        "index_curl": idx_curl,
        "middle_curl": mid_curl,
        "ring_curl": rng_curl,
        "pinky_curl": pnk_curl,
        "thumb_pointing_up": thumb_pointing_up,
        "thumb_extended_out": thumb_extended_out,
        "thumb_tucked": thumb_tucked,
        "thumb_index_touch": thumb_index_touch,
        "thumb_middle_touch": thumb_middle_touch,
        "thumb_tip_index_tip": thumb_tip_index_tip,
        "thumb_tip_middle_tip": thumb_tip_middle_tip,
        "thumb_tip_index_mcp": thumb_tip_index_mcp,
        "thumb_tip_pinky_tip": thumb_tip_pinky_tip,
        "index_middle_dist": index_middle_dist,
        "middle_ring_dist": middle_ring_dist,
        "ring_pinky_dist": ring_pinky_dist,
        "d_norm": d_norm,
        "handedness": handedness,
    }

def classify_landmarks(points, handedness="Right"):
    """
    Return (label, confidence, explanation) from 21 [x,y,z] landmark points.
    
    Supports: A, B, C, D, E, F, I, L, O, U, V, W, Y, ILY, THUMBS_UP, OPEN_PALM, FIST
    """
    if points is None or len(points) != 21:
        return "UNKNOWN", 0.0, "No hand detected"

    p = points
    h = analyze_landmarks(points, handedness=handedness)
    if h is None:
        return "UNKNOWN", 0.0, "Invalid hand landmarks"

    idx = h["index_ext"]
    mid = h["middle_ext"]
    rng = h["ring_ext"]
    pnk = h["pinky_ext"]
    n_fingers = sum([idx, mid, rng, pnk])

    # 1. ILY (I Love You): Thumb out, Index up, Pinky up, Middle & Ring curled
    if idx and pnk and not mid and not rng and h["thumb_extended_out"]:
        return "ILY", 0.92, "Thumb, index, and pinky extended ('I Love You' sign 🤟)"

    # 2. Y: Thumb out, Pinky up, Index, Middle, Ring curled
    if pnk and not idx and not mid and not rng and h["thumb_extended_out"]:
        return "Y", 0.90, "Thumb and pinky extended ('Y' / Hang loose 🤙)"

    # 3. L: Index up, Thumb extended sideways (~90°), Middle, Ring, Pinky curled
    if idx and not mid and not rng and not pnk and h["thumb_extended_out"]:
        return "L", 0.90, "Index finger upright and thumb extended at right angle ('L')"

    # 4. I: Pinky up only, Index, Middle, Ring curled, Thumb folded
    if pnk and not idx and not mid and not rng and not h["thumb_extended_out"]:
        return "I", 0.88, "Pinky finger extended upright ('I')"

    # 5. F / OK: Thumb and Index tips touching in circle, Middle, Ring, Pinky extended
    if mid and rng and pnk and h["thumb_index_touch"]:
        return "F", 0.88, "Thumb and index tips touching in circle with 3 fingers extended ('F' / OK 👌)"

    # 6. W: Index, Middle, Ring extended, Pinky curled
    if idx and mid and rng and not pnk:
        return "W", 0.85, "Three fingers extended: index, middle, and ring ('W')"

    # 7. U vs V (Index & Middle extended, Ring & Pinky curled)
    if idx and mid and not rng and not pnk:
        if h["index_middle_dist"] < 0.38:
            return "U", 0.84, "Index and middle extended together ('U')"
        else:
            return "V", 0.88, "Index and middle extended in V-shape ('V' / Peace ✌️)"

    # 8. D: Index extended upright, others curled, thumb touching curled fingers
    if idx and not mid and not rng and not pnk and not h["thumb_extended_out"]:
        return "D", 0.82, "Index finger extended upright, thumb curled ('D')"

    # 9. 4 fingers extended: B vs OPEN_PALM (Stop)
    if n_fingers == 4:
        if h["thumb_extended_out"]:
            return "OPEN_PALM", 0.90, "Open hand with all fingers and thumb spread out (Stop 🖐️)"
        else:
            return "B", 0.85, "Four fingers extended upright together, thumb tucked ('B')"

    # 10. 0 fingers extended (Fist variants: THUMBS_UP, A, E, O, C, FIST)
    if n_fingers == 0:
        # Check O: all fingertips meet thumb tip in an open circular loop (thumb arched, not tucked flat)
        if h["thumb_index_touch"] and h["thumb_middle_touch"] and not h["thumb_tucked"]:
            return "O", 0.82, "Fingers curved down to meet thumb forming an 'O' circle"

        # Check C: fingers and thumb curved forming an open 'C' arc
        if 0.30 <= h["thumb_tip_index_tip"] <= 0.90 and p[8][1] < p[5][1] and not h["thumb_tucked"]:
            return "C", 0.78, "Fingers and thumb curved forming an open 'C' shape"

        # Check Thumbs Up: thumb tip clearly pointing upright high into the air
        thumb_fully_extended_up = (
            h["thumb_pointing_up"]
            and (p[4][1] < p[5][1] - 0.25 * h["palm_size"])
            and (h["d_norm"](4, 2) > 0.60)
        )
        if thumb_fully_extended_up:
            return "THUMBS_UP", 0.90, "Fist with thumb pointing upright (Thumbs Up 👍)"

        # Check A: fist with thumb upright alongside index MCP/PIP
        if h["thumb_pointing_up"]:
            return "A", 0.82, "Closed fist with thumb resting upright alongside index finger ('A')"

        # Check E: fingertips curled down resting right on top of folded thumb
        is_e_perched = (
            h["thumb_tucked"]
            and (h["d_norm"](4, 8) < 0.25)
            and (p[8][1] <= p[4][1])
        )
        if is_e_perched:
            return "E", 0.78, "Fingertips curled down resting on top of folded thumb ('E')"

        # General Fist / Letter S
        return "FIST", 0.82, "Closed fist / Letter S (✊)"

    return "UNKNOWN", 0.25, f"{n_fingers} fingers extended; pose not confidently matched"


def get_sample_landmarks(sign_key):
    """Generate synthetic 21-point landmarks for a given supported sign."""
    pts = np.zeros((21, 3))
    pts[0] = [0.5, 0.8, 0.0]   # Wrist
    pts[1] = [0.44, 0.72, 0.0] # Thumb CMC
    pts[2] = [0.38, 0.65, 0.0] # Thumb MCP
    pts[5] = [0.42, 0.52, 0.0] # Index MCP
    pts[9] = [0.50, 0.50, 0.0] # Middle MCP
    pts[13] = [0.58, 0.52, 0.0] # Ring MCP
    pts[17] = [0.65, 0.55, 0.0] # Pinky MCP

    def set_finger(pip_idx, dip_idx, tip_idx, x_base, state):
        if state == "extended":
            pts[pip_idx] = [x_base, 0.38, 0.0]
            pts[dip_idx] = [x_base, 0.28, 0.0]
            pts[tip_idx] = [x_base, 0.18, 0.0]
        elif state == "curled":
            pts[pip_idx] = [x_base, 0.44, 0.0]
            pts[dip_idx] = [x_base, 0.52, 0.0]
            pts[tip_idx] = [x_base, 0.58, 0.0]
        elif state == "c_curve":
            pts[pip_idx] = [x_base, 0.40, 0.0]
            pts[dip_idx] = [x_base, 0.38, -0.05]
            pts[tip_idx] = [x_base, 0.45, -0.08]
        elif state == "o_curve":
            pts[pip_idx] = [x_base, 0.42, 0.0]
            pts[dip_idx] = [x_base, 0.48, -0.05]
            pts[tip_idx] = [0.38, 0.50, -0.05]

    config = {
        "ILY": {"thumb": "out", "idx": "extended", "mid": "curled", "rng": "curled", "pnk": "extended", "spread": False},
        "Y": {"thumb": "out", "idx": "curled", "mid": "curled", "rng": "curled", "pnk": "extended", "spread": False},
        "L": {"thumb": "out", "idx": "extended", "mid": "curled", "rng": "curled", "pnk": "curled", "spread": False},
        "I": {"thumb": "tucked", "idx": "curled", "mid": "curled", "rng": "curled", "pnk": "extended", "spread": False},
        "W": {"thumb": "tucked", "idx": "extended", "mid": "extended", "rng": "extended", "pnk": "curled", "spread": True},
        "V": {"thumb": "tucked", "idx": "extended", "mid": "extended", "rng": "curled", "pnk": "curled", "spread": True},
        "U": {"thumb": "tucked", "idx": "extended", "mid": "extended", "rng": "curled", "pnk": "curled", "spread": False},
        "D": {"thumb": "touch_mid", "idx": "extended", "mid": "curled", "rng": "curled", "pnk": "curled", "spread": False},
        "F": {"thumb": "touch_idx", "idx": "curled", "mid": "extended", "rng": "extended", "pnk": "extended", "spread": False},
        "B": {"thumb": "tucked", "idx": "extended", "mid": "extended", "rng": "extended", "pnk": "extended", "spread": False},
        "OPEN_PALM": {"thumb": "out", "idx": "extended", "mid": "extended", "rng": "extended", "pnk": "extended", "spread": True},
        "THUMBS_UP": {"thumb": "up", "idx": "curled", "mid": "curled", "rng": "curled", "pnk": "curled", "spread": False},
        "A": {"thumb": "up_side", "idx": "curled", "mid": "curled", "rng": "curled", "pnk": "curled", "spread": False},
        "E": {"thumb": "tucked", "idx": "curled", "mid": "curled", "rng": "curled", "pnk": "curled", "spread": False},
        "FIST": {"thumb": "fist", "idx": "curled", "mid": "curled", "rng": "curled", "pnk": "curled", "spread": False},
        "C": {"thumb": "c_curve", "idx": "c_curve", "mid": "c_curve", "rng": "c_curve", "pnk": "c_curve", "spread": False},
        "O": {"thumb": "o_curve", "idx": "o_curve", "mid": "o_curve", "rng": "o_curve", "pnk": "o_curve", "spread": False},
    }

    c = config.get(sign_key, config["B"])
    idx_x = 0.38 if c["spread"] else 0.44
    mid_x = 0.54 if c["spread"] else 0.48
    set_finger(6, 7, 8, idx_x, c["idx"])
    set_finger(10, 11, 12, mid_x, c["mid"])
    set_finger(14, 15, 16, 0.58, c["rng"])
    set_finger(18, 19, 20, 0.65, c["pnk"])

    thumb_mode = c["thumb"]
    if thumb_mode == "out":
        pts[3] = [0.30, 0.62, 0.0]
        pts[4] = [0.18, 0.58, 0.0]
    elif thumb_mode == "up":
        pts[3] = [0.36, 0.50, 0.0]
        pts[4] = [0.36, 0.35, 0.0]
    elif thumb_mode == "up_side":
        pts[3] = [0.37, 0.54, 0.0]
        pts[4] = [0.38, 0.45, 0.0]
    elif thumb_mode == "tucked":
        pts[3] = [0.42, 0.62, 0.0]
        pts[4] = [0.46, 0.60, 0.0]
    elif thumb_mode == "touch_idx":
        pts[3] = [0.36, 0.30, 0.0]
        pts[4] = pts[8].copy()
    elif thumb_mode == "touch_mid":
        pts[3] = [0.42, 0.54, 0.0]
        pts[4] = pts[12].copy()
    elif thumb_mode == "fist":
        pts[3] = [0.46, 0.65, 0.0]
        pts[4] = [0.54, 0.62, 0.0]
    elif thumb_mode == "c_curve":
        pts[3] = [0.32, 0.58, 0.0]
        pts[4] = [0.32, 0.48, -0.05]
    elif thumb_mode == "o_curve":
        pts[3] = [0.36, 0.54, 0.0]
        pts[4] = [0.38, 0.50, -0.05]

    return pts


def render_landmarks_image(landmarks, width=420, height=420):
    """Render 21 hand landmarks on a modern dark canvas with connections and styled joints."""
    import cv2
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (20, 24, 34) # Dark background

    if landmarks is None or len(landmarks) != 21:
        cv2.putText(img, "No Landmarks Available", (60, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (148, 163, 184), 1, cv2.LINE_AA)
        return img

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
    ]

    # Draw connection lines
    for start, end in connections:
        pt1 = (int(landmarks[start][0] * width), int(landmarks[start][1] * height))
        pt2 = (int(landmarks[end][0] * width), int(landmarks[end][1] * height))
        cv2.line(img, pt1, pt2, (168, 85, 247), 3, cv2.LINE_AA)

    # Draw joints and fingertips
    fingertips = {4: "Thumb", 8: "Index", 12: "Middle", 16: "Ring", 20: "Pinky"}
    for idx, pt in enumerate(landmarks):
        center = (int(pt[0] * width), int(pt[1] * height))
        if idx in fingertips:
            cv2.circle(img, center, 7, (6, 182, 212), -1, cv2.LINE_AA)
            cv2.circle(img, center, 9, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.circle(img, center, 4, (241, 245, 249), -1, cv2.LINE_AA)

    return img

