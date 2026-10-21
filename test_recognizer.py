"""Unit and template tests for recognizer.py."""
import unittest
import numpy as np
from recognizer import classify_landmarks, SUPPORTED_SIGNS, analyze_landmarks

def build_template_hand(
    thumb="tucked",
    index="extended",
    middle="curled",
    ring="curled",
    pinky="curled",
    spread_uv=False,
):
    """Generate 21 3D landmarks for a simulated hand pose."""
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
            pts[dip_idx] = [x_base, 0.36, -0.05]
            pts[tip_idx] = [x_base, 0.42, -0.10]
        elif state == "o_curve":
            pts[pip_idx] = [x_base, 0.42, 0.0]
            pts[dip_idx] = [x_base, 0.48, -0.08]
            pts[tip_idx] = [0.38, 0.50, -0.10]

    idx_x = 0.42
    mid_x = 0.50
    if spread_uv and index == "extended" and middle == "extended":
        idx_x = 0.38
        mid_x = 0.54
    elif not spread_uv and index == "extended" and middle == "extended":
        idx_x = 0.45
        mid_x = 0.48

    set_finger(6, 7, 8, idx_x, index)
    set_finger(10, 11, 12, mid_x, middle)
    set_finger(14, 15, 16, 0.58, ring)
    set_finger(18, 19, 20, 0.65, pinky)

    if thumb == "out":
        pts[3] = [0.30, 0.62, 0.0]
        pts[4] = [0.18, 0.58, 0.0]
    elif thumb == "up":
        pts[3] = [0.36, 0.50, 0.0]
        pts[4] = [0.36, 0.35, 0.0]
    elif thumb == "tucked":
        pts[3] = [0.42, 0.62, 0.0]
        pts[4] = [0.46, 0.60, 0.0]
    elif thumb == "touch_index":
        pts[3] = [0.36, 0.30, 0.0]
        pts[4] = pts[8].copy()
    elif thumb == "touch_middle":
        pts[3] = [0.42, 0.54, 0.0]
        pts[4] = pts[12].copy()
    elif thumb == "c_curve":
        pts[3] = [0.30, 0.60, 0.0]
        pts[4] = [0.30, 0.46, -0.05]
    elif thumb == "o_curve":
        pts[3] = [0.36, 0.54, 0.0]
        pts[4] = [0.38, 0.50, -0.10]
    elif thumb == "fist":
        pts[3] = [0.42, 0.65, 0.0]
        pts[4] = [0.44, 0.65, 0.0]

    return pts


class TestSignRecognizer(unittest.TestCase):
    def test_null_and_invalid(self):
        label, conf, desc = classify_landmarks(None)
        self.assertEqual(label, "UNKNOWN")
        self.assertEqual(conf, 0.0)

        label, conf, desc = classify_landmarks(np.zeros((10, 3)))
        self.assertEqual(label, "UNKNOWN")

    def test_sign_ily(self):
        pts = build_template_hand(thumb="out", index="extended", middle="curled", ring="curled", pinky="extended")
        label, conf, _ = classify_landmarks(pts)
        self.assertEqual(label, "ILY")
        self.assertGreaterEqual(conf, 0.8)

    def test_sign_y(self):
        pts = build_template_hand(thumb="out", index="curled", middle="curled", ring="curled", pinky="extended")
        label, conf, _ = classify_landmarks(pts)
        self.assertEqual(label, "Y")

    def test_sign_l(self):
        pts = build_template_hand(thumb="out", index="extended", middle="curled", ring="curled", pinky="curled")
        label, conf, _ = classify_landmarks(pts)
        self.assertEqual(label, "L")

    def test_sign_i(self):
        pts = build_template_hand(thumb="tucked", index="curled", middle="curled", ring="curled", pinky="extended")
        label, conf, _ = classify_landmarks(pts)
        self.assertEqual(label, "I")

    def test_sign_w(self):
        pts = build_template_hand(thumb="tucked", index="extended", middle="extended", ring="extended", pinky="curled")
        label, conf, _ = classify_landmarks(pts)
        self.assertEqual(label, "W")

    def test_sign_v(self):
        pts = build_template_hand(thumb="tucked", index="extended", middle="extended", ring="curled", pinky="curled", spread_uv=True)
        label, conf, _ = classify_landmarks(pts)
        self.assertEqual(label, "V")

    def test_sign_u(self):
        pts = build_template_hand(thumb="tucked", index="extended", middle="extended", ring="curled", pinky="curled", spread_uv=False)
        label, conf, _ = classify_landmarks(pts)
        self.assertEqual(label, "U")

    def test_sign_d(self):
        pts = build_template_hand(thumb="tucked", index="extended", middle="curled", ring="curled", pinky="curled")
        label, conf, _ = classify_landmarks(pts)
        self.assertEqual(label, "D")

    def test_sign_f(self):
        pts = build_template_hand(thumb="touch_index", index="curled", middle="extended", ring="extended", pinky="extended")
        label, conf, _ = classify_landmarks(pts)
        self.assertEqual(label, "F")

    def test_sign_b(self):
        pts = build_template_hand(thumb="tucked", index="extended", middle="extended", ring="extended", pinky="extended")
        label, conf, _ = classify_landmarks(pts)
        self.assertEqual(label, "B")

    def test_sign_open_palm(self):
        pts = build_template_hand(thumb="out", index="extended", middle="extended", ring="extended", pinky="extended")
        label, conf, _ = classify_landmarks(pts)
        self.assertEqual(label, "OPEN_PALM")

    def test_sign_thumbs_up(self):
        pts = build_template_hand(thumb="up", index="curled", middle="curled", ring="curled", pinky="curled")
        label, conf, _ = classify_landmarks(pts)
        self.assertEqual(label, "THUMBS_UP")

    def test_all_sample_signs(self):
        from recognizer import get_sample_landmarks
        expected_signs = [
            'ILY', 'Y', 'L', 'I', 'W', 'V', 'U', 'D', 'F', 'B',
            'OPEN_PALM', 'THUMBS_UP', 'A', 'E', 'FIST', 'C', 'O'
        ]
        for sign in expected_signs:
            with self.subTest(sign=sign):
                pts = get_sample_landmarks(sign)
                label, conf, desc = classify_landmarks(pts)
                self.assertEqual(label, sign, f"Expected {sign}, got {label}")
                self.assertGreaterEqual(conf, 0.70)

    def test_metadata_keys(self):
        for sign_id, info in SUPPORTED_SIGNS.items():
            self.assertIn("name", info)
            self.assertIn("emoji", info)
            self.assertIn("category", info)
            self.assertIn("description", info)

if __name__ == "__main__":
    unittest.main()

