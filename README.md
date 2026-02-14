# 🎯 AI Emotion + Iris Liveness Detection

A real-time computer vision system that combines:

- 😊 Facial Emotion Recognition
- 👁 Iris Liveness Detection (Anti-Spoofing)
- 📷 Live Camera Processing using OpenCV
- 🧠 Deep Learning powered by DeepFace & TensorFlow

---

## 📌 Project Overview

This project performs **real-time facial emotion detection** and **iris-based liveness verification** using a webcam.

It analyzes:

1. Face emotions (Happy, Sad, Angry, Surprise, Neutral)
2. Iris texture sharpness
3. Eye edge density
4. Reflection variance
5. Blink detection

The system helps detect:
- Fake printed iris images
- Glossy spoof attempts
- No-blink attacks
- Real live human iris

---

## 🖥️ Live Output Features

### 🎭 Emotion Detection
- Uses DeepFace
- Displays:
  - Top emotion with confidence %
  - Emotion score bars
  - Real-time emotion updates

Example output:
HAPPY (90%)
NEUTRAL (97%)


---

### 👁 Iris Liveness Detection

The system checks:

| Parameter | Purpose |
|-----------|----------|
| Laplacian Variance | Measures sharpness |
| Edge Density | Measures texture complexity |
| Reflection Variance | Detects glossy surface |
| Blink Detection | Prevents photo spoofing |

Possible outputs:

Real Iris Detected!
Fake Iris Detected! (Smooth Texture)
Fake Iris Detected! (Glossy Printed Surface)
Fake Iris Detected! (No Blinking)


---

## 🛠️ Technologies Used

- Python 3.10
- OpenCV
- DeepFace
- TensorFlow
- NumPy
- Haar Cascade (Eye Detection)

---

## 📂 Project Structure

    AI-Emotion-Iris-Liveness/
    │
    ├── emotion_iris1.py      # Main application script
    └── README.md             # Project documentation

---

## ⚙️ Installation Guide

### 1️⃣ Install Python (3.8 – 3.10 recommended)

Download from:
https://www.python.org/downloads/

Make sure to check:
✅ Add Python to PATH

---

### 2️⃣ Install Required Libraries

python -m pip install opencv-python
python -m pip install deepface
python -m pip install tf-keras


If TensorFlow version causes issues:

python -m pip uninstall tensorflow -y
python -m pip install tensorflow==2.15.0


---

## ▶️ How to Run

python emotion_iris1.py


Press:

q → Quit the application


---

## 🎛 Adjustable Parameters

### Laplacian Threshold (Trackbar)

You can adjust sharpness sensitivity using the trackbar:

Laplacian Threshold (0 – 300)


This helps tune real vs fake iris detection sensitivity.

---

## 🔍 How Liveness Detection Works

### Real Iris Indicators:
- Natural texture variation
- Balanced edge density
- Natural reflections
- Regular blinking

### Fake Iris Indicators:
- Overly smooth texture
- Printed glossy reflection
- No blinking for 40+ seconds
- Abnormal sharpness patterns

---

## 🚀 Applications

- Biometric authentication systems
- Anti-spoofing security systems
- Smart attendance systems
- Emotion-based analytics
- AI surveillance research

---

## ⚠️ Limitations

- Haar cascade eye detection is basic (not deep-learning based)
- Lighting conditions affect accuracy
- Blink detection is brightness-based (can be improved)
- Not production-grade anti-spoofing

---

## 🔮 Future Improvements

- Use MediaPipe / RetinaFace for better eye tracking
- Use CNN-based iris spoof detection
- Improve blink detection using Eye Aspect Ratio (EAR)
- Add face recognition
- Add database logging
- Deploy as a web application

---

## 👨‍💻 Author

Ayush Gawali  

AI + Computer Vision Project  
Emotion Recognition + Iris Liveness Detection

---

## 🧠 Concept Summary

This project demonstrates how:

Deep Learning (Emotion AI) + Classical Computer Vision (Laplacian, Edge Detection, Blink Logic)

can be combined to create a real-time biometric liveness detection system.
