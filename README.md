# SignSpeak AI — reference-based starter

This version removes the requirement to train a classifier on your own A/B/C CSV before opening the web app. It uses MediaPipe hand landmarks and a simple configurable landmark-rule recognizer for a useful subset of static ASL letters, then builds text and browser speech.

**Honest limitation:** an alphabet poster is not a labeled landmark dataset, and this rule-based starter is not a fully accurate A–Z ASL model. Finger-pose rules can confuse similar letters; J and Z require motion tracking and are not handled as static signs. Use it as a prototype, and verify/adjust the rules against your own webcam.

## Recommended environment
Python 3.11 on macOS.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If you already have an old project `.venv`, the safest approach is to create a fresh environment. Avoid mixing `opencv-python` with `opencv-python-headless`.

## Features
- Browser webcam capture (Streamlit camera input)
- MediaPipe hand landmark extraction
- Reference/rule-based prediction for a small set of static poses
- Editable message text, add letter, space, backspace, clear
- Browser text-to-speech

## Important
The starter's predictions are heuristic, not a pretrained ASL model. For robust A–Z recognition, replace `recognizer.py` with a trained model based on a real labeled ASL dataset or collect varied webcam examples. The app explicitly shows low-confidence/unsupported cases rather than claiming full accuracy.



# 🤟 SignSpeak AI — Sign Language Recognition System

**Breaking communication barriers through Artificial Intelligence and Computer Vision.**

SignSpeak AI is an AI-powered project designed to recognize hand gestures and help convert sign language into understandable text. The goal is to make communication more accessible for people who use sign language.

## 📌 Project Overview

SignSpeak AI uses computer vision to detect and analyze hand gestures through a camera. By identifying hand landmarks and processing gesture information, the system aims to recognize signs and display their corresponding meaning.

## ✨ Features

- ✋ **Hand Gesture Detection** — Detects hands using computer vision.
- 👁️ **Real-Time Recognition** — Designed to process gestures through a live camera feed.
- 🤖 **AI Integration** — Supports gesture recognition using machine learning techniques.
- 📝 **Text Conversion** — Displays the recognized sign as text.
- ♿ **Accessibility Focused** — Aims to reduce communication barriers.
- 💻 **Python-Based Development** — Uses Python and relevant computer vision libraries.

## 🛠️ Technologies Used

- Python
- MediaPipe
- OpenCV
- Machine Learning
- Computer Vision

*Note: Technologies and features may vary depending on the current implementation.*

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/kuvaleshcd-hue/signspeak-ai.git
```

### 2. Navigate to the project directory

```bash
cd signspeak-ai
```

### 3. Create a virtual environment

```bash
python3 -m venv .venv
```

### 4. Activate the virtual environment

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

If your repository contains a `requirements.txt` file, run:

```bash
python -m pip install -r requirements.txt
```

Otherwise, install the required libraries listed in your project dependencies.

## ▶️ Running the Project

Run the Python entry-point file used by your application. For example, if your main file is `app.py`:

```bash
python app.py
```

Replace `app.py` with your actual main Python filename.

If the application uses a webcam, grant camera permission when prompted.

## 🔄 How It Works

1. **Input:** Capture hand gestures using a camera.
2. **Detection:** Detect hand landmarks using MediaPipe.
3. **Processing:** Extract gesture information from the detected landmarks.
4. **Recognition:** Identify the gesture using the implemented recognition logic or trained model.
5. **Output:** Display the corresponding text when a supported sign is recognized.

## 🎯 Project Objectives

- Explore real-time sign language recognition.
- Apply computer vision to a real-world accessibility problem.
- Develop practical AI and machine learning skills.
- Build a foundation for future sign-to-speech communication systems.

## 🚀 Future Improvements

- Support more sign language gestures.
- Add text-to-speech conversion.
- Improve recognition accuracy through model training and testing.
- Support dynamic gestures and sentence formation.
- Develop a user-friendly web or mobile interface.
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

just i add the read me file

