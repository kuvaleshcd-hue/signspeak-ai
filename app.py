import json
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from hand_utils import landmarks_from_rgb
from recognizer import classify_landmarks

st.set_page_config(page_title="SignSpeak AI", page_icon="🤟", layout="wide")
st.title("🤟 SignSpeak AI")
st.caption("Webcam hand signs → text → speech")
st.warning(
    "Prototype mode: this app uses hand-landmark rules, not a fully trained A–Z ASL model. "
    "Predictions are limited and may be wrong. J and Z require motion recognition."
)

if "message" not in st.session_state:
    st.session_state.message = ""
if "prediction" not in st.session_state:
    st.session_state.prediction = "UNKNOWN"

left, right = st.columns([1, 1])
with left:
    st.subheader("1. Capture a hand sign")
    photo = st.camera_input("Show one hand sign clearly")
    if photo is not None:
        rgb = np.array(Image.open(photo).convert("RGB"))
        landmarks = landmarks_from_rgb(rgb)
        label, confidence, reason = classify_landmarks(landmarks)
        st.session_state.prediction = label
        if label == "UNKNOWN" or confidence < 0.5:
            st.warning(f"Not confident: {label} — {reason}. Try a clear, upright pose.")
        else:
            st.success(f"Predicted letter: **{label}** (heuristic confidence {confidence:.0%})")
        st.image(rgb, caption="Captured frame", use_container_width=True)

with right:
    st.subheader("2. Build text")
    st.text_area("Editable message", key="message", height=180)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Add predicted letter", type="primary", use_container_width=True):
            label = st.session_state.prediction
            if label in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
                st.session_state.message += label
                st.rerun()
            else:
                st.info("No supported confident letter. Capture a gesture first.")
    with c2:
        if st.button("Add space", use_container_width=True):
            st.session_state.message += " "
            st.rerun()
    c3, c4 = st.columns(2)
    with c3:
        if st.button("Backspace", use_container_width=True):
            st.session_state.message = st.session_state.message[:-1]
            st.rerun()
    with c4:
        if st.button("Clear", use_container_width=True):
            st.session_state.message = ""
            st.rerun()

    st.subheader("3. Text-to-speech")
    speech_text = json.dumps(st.session_state.message.strip())
    components.html(f"""
    <div style="font-family:system-ui">
      <button id="speak" style="padding:10px 14px;border-radius:8px">🔊 Speak sentence</button>
      <button id="stop" style="padding:10px 14px;border-radius:8px;margin-left:8px">Stop</button>
      <span id="status" style="margin-left:8px"></span>
    </div>
    <script>
      const text = {speech_text};
      const status = document.getElementById("status");
      document.getElementById("speak").onclick = () => {{
        if (!text.trim()) {{ status.textContent = "Add text first."; return; }}
        if (!window.speechSynthesis) {{ status.textContent = "Speech unavailable."; return; }}
        speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance(text);
        u.lang = "en-US";
        u.onstart = () => status.textContent = "Speaking…";
        u.onend = () => status.textContent = "Finished.";
        u.onerror = () => status.textContent = "Speech failed.";
        speechSynthesis.speak(u);
      }};
      document.getElementById("stop").onclick = () => {{
        if (window.speechSynthesis) speechSynthesis.cancel();
        status.textContent = "Stopped.";
      }};
    </script>
    """, height=60)

st.divider()
st.subheader("Supported prototype poses")
st.markdown(
    "- **B**: four fingers extended, thumb position heuristic\\n"
    "- **V**: index and middle extended\\n"
    "- **W**: index, middle, and ring extended\\n"
    "- **D**: index extended (low confidence)\\n"
    "- Other letters: return UNKNOWN until a trained model/rules are added"
)
st.caption("For accurate A–Z, use a labeled dataset or collect real examples across users, angles, and lighting.")
