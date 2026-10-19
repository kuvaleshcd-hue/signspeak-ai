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
