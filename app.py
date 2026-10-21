import json
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from hand_utils import detect_hand
from recognizer import (
    classify_landmarks,
    SUPPORTED_SIGNS,
    get_sample_landmarks,
    render_landmarks_image,
)

st.set_page_config(
    page_title="SignSpeak AI — Hand Sign Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished look
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .hero-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 20px;
        backdrop-filter: blur(8px);
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-success {
        background-color: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .badge-warning {
        background-color: rgba(234, 179, 8, 0.15);
        color: #facc15;
        border: 1px solid rgba(234, 179, 8, 0.3);
    }
    .sign-pill {
        display: inline-block;
        padding: 6px 12px;
        margin: 4px;
        background: rgba(51, 65, 85, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 8px;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.title("🤟 SignSpeak AI")
st.caption("AI-Powered Hand Sign Recognition • ASL Letters & Expressive Gestures • Text & Speech Studio")

# Session state initialization
if "message" not in st.session_state:
    st.session_state.message = ""
if "prediction" not in st.session_state:
    st.session_state.prediction = "UNKNOWN"
if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0
if "explanation" not in st.session_state:
    st.session_state.explanation = "No hand analyzed yet."
if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = "ILY"

# Sidebar: Quick Navigation & Cheat Sheet
with st.sidebar:
    st.header("📖 Hand Sign Reference")
    st.caption(f"{len(SUPPORTED_SIGNS)} Supported Hand Signs")

    ref_category = st.radio("Browse category:", ["All", "ASL Letters", "Gestures"], horizontal=True)
    
    for key, info in SUPPORTED_SIGNS.items():
        if ref_category == "ASL Letters" and info["category"] != "letter":
            continue
        if ref_category == "Gestures" and info["category"] != "gesture":
            continue

        with st.expander(f"{info['emoji']} {info['name']}", expanded=False):
            st.markdown(f"**Pose instructions:**\n\n{info['description']}")
            if st.button(f"Try '{key}' in Simulator", key=f"sim_btn_{key}", use_container_width=True):
                st.session_state.selected_sample = key
                st.session_state.input_tab_index = 2
                st.rerun()

    st.divider()
    st.caption("SignSpeak AI prototype powered by Google MediaPipe & geometric hand landmark kinematics.")

# Main 2-column layout
left_col, right_col = st.columns([1.1, 0.9], gap="large")

with left_col:
    st.subheader("1. Input & Sign Recognition")

    tab1, tab2, tab3 = st.tabs(["📸 Webcam Snap", "📁 Upload Image", "🧪 Try Sample Signs"])

    active_landmarks = None
    annotated_display = None
    source_label = ""

    # Tab 1: Webcam Snap
    with tab1:
        st.write("Hold your hand clearly in front of the camera, palm facing forward.")
        photo = st.camera_input("Take a photo of your hand sign", key="webcam_camera")
        if photo is not None:
            rgb = np.array(Image.open(photo).convert("RGB"))
            landmarks, handedness, annotated = detect_hand(rgb)
            active_landmarks = landmarks
            annotated_display = annotated
            source_label = f"Webcam ({handedness or 'Right'} hand)"

    # Tab 2: Upload Image
    with tab2:
        uploaded_file = st.file_uploader(
            "Upload a photo of a hand sign (JPG, PNG, WebP)",
            type=["jpg", "jpeg", "png", "webp"],
            key="image_uploader",
        )
        if uploaded_file is not None:
            rgb = np.array(Image.open(uploaded_file).convert("RGB"))
            landmarks, handedness, annotated = detect_hand(rgb)
            active_landmarks = landmarks
            annotated_display = annotated
            source_label = f"Uploaded photo ({handedness or 'Right'} hand)"

    # Tab 3: Try Sample Signs Simulator
    with tab3:
        st.write("Test any sign immediately using geometric 3D landmark kinematics:")
        sample_options = list(SUPPORTED_SIGNS.keys())
        default_idx = (
            sample_options.index(st.session_state.selected_sample)
            if st.session_state.selected_sample in sample_options
            else 0
        )
        picked_sample = st.selectbox(
            "Select a sign to simulate:",
            sample_options,
            index=default_idx,
            format_func=lambda k: f"{SUPPORTED_SIGNS[k]['emoji']} {SUPPORTED_SIGNS[k]['name']} ({k})",
        )
        st.session_state.selected_sample = picked_sample

        sim_landmarks = get_sample_landmarks(picked_sample)
        active_landmarks = sim_landmarks
        annotated_display = render_landmarks_image(sim_landmarks)
        source_label = f"Simulation: {SUPPORTED_SIGNS[picked_sample]['name']}"

    # Prediction Analysis
    if active_landmarks is not None:
        label, conf, reason = classify_landmarks(active_landmarks)
        st.session_state.prediction = label
        st.session_state.confidence = conf
        st.session_state.explanation = reason

        # Prediction Display Box
        sign_info = SUPPORTED_SIGNS.get(label, None)
        emoji_display = sign_info["emoji"] if sign_info else "❓"
        name_display = sign_info["name"] if sign_info else label

        if label != "UNKNOWN" and conf >= 0.50:
            st.success(
                f"### {emoji_display} **{name_display}** ({label})\n"
                f"**Confidence:** `{conf:.0%}` • *{reason}*"
            )
        else:
            st.warning(
                f"### ❓ **{label}**\n"
                f"**Confidence:** `{conf:.0%}` • *{reason}*\n\n"
                f"👉 *Tip: Hold hand upright, palm facing camera, with fingers distinct.*"
            )

        if annotated_display is not None:
            st.image(
                annotated_display,
                caption=f"Hand tracking overlay: {source_label}",
                use_container_width=True,
            )
    else:
        st.info("Capture a webcam snap, upload a picture, or switch to the 'Try Sample Signs' tab.")

with right_col:
    st.subheader("2. Build Sentence & Speak")

    # Editable Text Area
    st.text_area(
        "Constructed Message",
        key="message",
        height=160,
        placeholder="Your signed words will appear here...",
        help="Edit freely or use the buttons below to append recognized signs.",
    )

    # Action Buttons
    c_add, c_space = st.columns([1.2, 0.8])
    with c_add:
        pred_label = st.session_state.prediction
        pred_info = SUPPORTED_SIGNS.get(pred_label, None)
        add_btn_label = f"Add '{pred_label}'" if pred_label != "UNKNOWN" else "Add Sign"

        if st.button(f"➕ {add_btn_label}", type="primary", use_container_width=True):
            if pred_info is not None:
                char_to_add = pred_info.get("char", pred_label)
                st.session_state.message += char_to_add
                st.rerun()
            elif pred_label in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
                st.session_state.message += pred_label
                st.rerun()
            else:
                st.info("No confident sign detected to add. Please capture or select a pose.")

    with c_space:
        if st.button("␣ Space", use_container_width=True):
            st.session_state.message += " "
            st.rerun()

    c_bs, c_clr, c_dot = st.columns([1, 1, 0.8])
    with c_bs:
        if st.button("⌫ Backspace", use_container_width=True):
            st.session_state.message = st.session_state.message[:-1]
            st.rerun()
    with c_clr:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.message = ""
            st.rerun()
    with c_dot:
        if st.button("➕ '.'", use_container_width=True):
            st.session_state.message += "."
            st.rerun()

    st.write("")
    st.subheader("3. Audio Playback (Text-to-Speech)")
    
    clean_speech_text = json.dumps(st.session_state.message.strip())
    components.html(
        f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; padding: 12px; background: rgba(30, 41, 59, 0.5); border-radius: 10px; border: 1px solid rgba(148, 163, 184, 0.2);">
          <div style="display: flex; gap: 8px; align-items: center;">
            <button id="speak-btn" style="background: #2563eb; color: white; border: none; padding: 10px 18px; border-radius: 8px; font-weight: 600; cursor: pointer; transition: all 0.2s;">
              🔊 Speak Message
            </button>
            <button id="stop-btn" style="background: #475569; color: white; border: none; padding: 10px 14px; border-radius: 8px; font-weight: 500; cursor: pointer;">
              ⏹ Stop
            </button>
            <span id="tts-status" style="margin-left: 10px; font-size: 0.9rem; color: #94a3b8;"></span>
          </div>
        </div>
        <script>
          const rawText = {clean_speech_text};
          const btnSpeak = document.getElementById("speak-btn");
          const btnStop = document.getElementById("stop-btn");
          const statusText = document.getElementById("tts-status");

          btnSpeak.onclick = () => {{
            if (!rawText.trim()) {{
              statusText.textContent = "Please add some text first.";
              return;
            }}
            if (!window.speechSynthesis) {{
              statusText.textContent = "Speech synthesis is not supported in this browser.";
              return;
            }}
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(rawText);
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            utterance.lang = "en-US";
            utterance.onstart = () => {{ statusText.textContent = "🔊 Speaking..."; }};
            utterance.onend = () => {{ statusText.textContent = "✓ Finished."; }};
            utterance.onerror = (e) => {{ statusText.textContent = "Error during speech: " + e.error; }};
            window.speechSynthesis.speak(utterance);
          }};

          btnStop.onclick = () => {{
            if (window.speechSynthesis) {{
              window.speechSynthesis.cancel();
              statusText.textContent = "⏹ Stopped.";
            }}
          }};
        </script>
        """,
        height=75,
    )

    st.write("")
    st.subheader("💡 Tips for Best Hand Detection")
    st.markdown(
        "- **Lighting:** Keep your hand evenly lit with minimal background clutter.\n"
        "- **Positioning:** Center your hand at comfortable arm's length from the camera.\n"
        "- **Upright Orientation:** Keep your wrist towards the bottom and fingertips pointing upwards.\n"
        "- **Simulator:** Switch to the **'Try Sample Signs'** tab anytime to explore all 17 supported poses."
    )

st.divider()
st.caption(
    "SignSpeak AI • Built with Python, Streamlit, Google MediaPipe & Web Speech API"
)
