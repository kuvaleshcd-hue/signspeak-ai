# 🤟 SignSpeak AI — Real-time Hand Sign Recognition

**Breaking communication barriers through Artificial Intelligence and Computer Vision.**

SignSpeak AI translates American Sign Language (ASL) static letters and expressive hand gestures into text and spoken audio using Google MediaPipe and geometric 3D landmark kinematics.

## 📌 Project Overview

SignSpeak AI uses computer vision to detect and analyze hand gestures through a camera. By identifying hand landmarks and processing gesture information, the system recognizes signs and displays their corresponding meaning — then lets you build sentences and hear them spoken aloud.

## 🚀 Quick Start

Ensure you have Python 3.11 installed.

### 1. Clone the repository

```bash
git clone https://github.com/kuvaleshcd-hue/signspeak-ai.git
cd signspeak-ai
```

### 2. Create and activate a virtual environment

**macOS / Linux**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run unit tests

```bash
python -m unittest test_recognizer.py
```

### 5. Launch the Streamlit Web App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ✨ Features

- **3 Ways to Try Hand Signs**:
  1. 📸 **Webcam Snap**: Real-time camera snapshot.
  2. 📁 **Image Upload**: Upload your own hand sign photos (JPG, PNG, WebP).
  3. 🧪 **Try Sample Signs Simulator**: Test any supported sign instantly using interactive 3D kinematic landmarks without needing a camera.
- **MediaPipe Hand Landmark Tracking**:
  - Live 21-point skeleton tracking.
  - Emerald green bounding box with Handedness badge (`Right Hand` / `Left Hand`).
  - High-visibility fingertip and joint markers.
- **17 Supported Hand Signs**:
  - **ASL Letters**: `A`, `B`, `C`, `D`, `E`, `F (OK)`, `I`, `L`, `O`, `U`, `V (Peace)`, `W`, `Y (Hang Loose)`.
  - **Expressive Gestures**: `🤟 ILY (I Love You)`, `👍 Thumbs Up`, `🖐️ Open Palm (Stop)`, `✊ Closed Fist / S`.
- **Text Studio & Text-to-Speech**:
  - Append recognized signs to an editable message.
  - Space, backspace, punctuation, and clear controls.
  - Browser text-to-speech audio synthesis via Web Speech API.
- **Visual Reference Guide / Cheat Sheet**:
  - Sidebar directory of all supported signs with finger-by-finger instructions.
  - One-click "Try in Simulator" buttons.

---

## 🛠️ Technologies Used

- Python
- Streamlit
- MediaPipe
- OpenCV
- NumPy
- Web Speech API

---

## 🔄 How It Works

1. **Input:** Capture hand gestures using a camera, upload an image, or use the sample simulator.
2. **Detection:** Detect hand landmarks using MediaPipe (21 3D keypoints per hand).
3. **Processing:** Extract geometric features normalized by palm size for scale invariance.
4. **Recognition:** Identify the sign using heuristic landmark-kinematic rules.
5. **Output:** Display the recognized sign with confidence, build text, and speak it aloud.

---

## 🖐️ Hand Sign Reference Guide

| Sign | Category | Emoji | Key Finger Pose |
|---|---|---|---|
| **A** | ASL Letter | ✊ | Fist with thumb upright resting alongside index finger |
| **B** | ASL Letter | ✋ | 4 fingers straight up together, thumb folded across palm |
| **C** | ASL Letter | 🤏 | Curved hand forming open 'C' arc shape |
| **D** | ASL Letter | ☝️ | Index finger upright, other fingers curled touching thumb |
| **E** | ASL Letter | ✊ | Fingertips curled down resting on top of folded thumb |
| **F** | ASL Letter | 👌 | Thumb and index tips touching (OK sign), other 3 fingers extended |
| **I** | ASL Letter | 🤙 | Pinky extended upright, other fingers curled |
| **L** | ASL Letter | 👆 | Index upright, thumb extended horizontally at ~90° |
| **O** | ASL Letter | ⭕ | All fingertips and thumb curved together forming an 'O' circle |
| **U** | ASL Letter | ✌️ | Index and middle fingers extended upright touching together |
| **V** | ASL Letter | ✌️ | Index and middle extended upright in a spread 'V' (Peace) |
| **W** | ASL Letter | 🖖 | Index, middle, and ring extended and spread upright |
| **Y** | ASL Letter | 🤙 | Thumb and pinky extended outward (Hang Loose / Shaka) |
| **ILY** | Gesture | 🤟 | Thumb, index, and pinky extended simultaneously |
| **THUMBS_UP** | Gesture | 👍 | Fist with thumb pointing upright into the air |
| **OPEN_PALM** | Gesture | 🖐️ | Open hand with all 5 digits spread out |
| **FIST** | Gesture | ✊ | All fingers and thumb curled into palm |

---

## 🧪 Testing

Run the automated test suite verifying all 17 hand signs:

```bash
python -m unittest test_recognizer.py
```

---

## 🎯 Project Objectives

- Explore real-time sign language recognition.
- Apply computer vision to a real-world accessibility problem.
- Develop practical AI and machine learning skills.
- Build a foundation for future sign-to-speech communication systems.

## 🚀 Future Improvements

- Support more sign language gestures.
- Improve recognition accuracy through model training and testing.
- Support dynamic gestures (J, Z) and sentence formation.
- Explore Indian Sign Language (ISL) support.

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome.

1. Fork this repository.
2. Create a feature branch.
3. Commit your changes.
4. Push the branch to your fork.
5. Submit a pull request.

## 👨‍💻 Author

**Kuvalesh CD**

GitHub: [@kuvaleshcd-hue](https://github.com/kuvaleshcd-hue)

Project Repository: [SignSpeak AI](https://github.com/kuvaleshcd-hue/signspeak-ai)

---

⭐ If you find this project interesting, consider starring the repository.

**Built with Python, Computer Vision, and a vision for accessible communication.**
