import cv2
import numpy as np
import time
from deepface import DeepFace

# =======================
# EMOTION DETECTION SETUP
# =======================
EMOTIONS = ["angry", "happy", "sad", "surprise", "neutral"]

COLORS = {
    "background": (30, 30, 30),
    "text": (255, 255, 255),
    "highlight": (0, 200, 255),
    "bar_active": (0, 255, 128),
    "bar_inactive": (100, 100, 100),
    "alert": (255, 80, 80)
}

ANALYZE_EVERY = 1.0
last_analysis_time = 0

emotion_scores = {e: 0 for e in EMOTIONS}
top_emotion = "neutral"
top_conf = 0


def analyze_emotion(frame_bgr):
    global emotion_scores, top_emotion, top_conf
    try:
        result = DeepFace.analyze(
            frame_bgr,
            actions=["emotion"],
            enforce_detection=False
        )
        emo_dict = result[0]["emotion"]

        for e in EMOTIONS:
            emotion_scores[e] = float(emo_dict.get(e, 0.0))

        top_emotion = max(EMOTIONS, key=lambda e: emotion_scores[e])
        top_conf = int(emotion_scores[top_emotion])
    except Exception:
        top_emotion, top_conf = "neutral", 0


# =======================
# IRIS LIVENESS SETUP
# =======================
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

no_blink_frames = 0
last_blink_time = time.time()


def edge_density(eye_region):
    edges = cv2.Canny(eye_region, 50, 150)
    return (np.sum(edges > 0) / (eye_region.shape[0] * eye_region.shape[1])) * 100


def detect_blink(eye_region):
    global no_blink_frames, last_blink_time
    avg_brightness = np.mean(eye_region)
    is_blinking = avg_brightness < 50

    if is_blinking:
        last_blink_time = time.time()
        no_blink_frames = 0
    else:
        no_blink_frames += 1

    time_since_last_blink = time.time() - last_blink_time
    return is_blinking, time_since_last_blink


def detect_fake_iris(eye_region, threshold):
    # Compute Laplacian (sharpness)
    laplacian = cv2.Laplacian(eye_region, cv2.CV_64F).var()

    # Compute Edge Density (Texture Complexity)
    edges = edge_density(eye_region)

    # Compute Reflection (Brightness Variance)
    reflection = np.std(eye_region)

    # Check blinking
    _, time_since_last_blink = detect_blink(eye_region)

    # Fake iris conditions
    if laplacian > threshold and edges > 10 and reflection < 15:
        return "Fake Iris Detected! (Smooth Texture)", COLORS["alert"]
    elif laplacian < threshold and edges < 10 and reflection > 15:
        return "Fake Iris Detected! (Glossy Printed Surface)", COLORS["alert"]
    elif time_since_last_blink > 40:  # 40s without blink
        return "Fake Iris Detected! (No Blinking)", COLORS["alert"]
    else:
        return "Real Iris Detected!", COLORS["bar_active"]


def update_threshold(val):
    global threshold
    threshold = val


threshold = 100

cv2.namedWindow("AI Emotion + Iris Liveness")
cv2.createTrackbar(
    "Laplacian Threshold",
    "AI Emotion + Iris Liveness",
    threshold,
    300,
    update_threshold
)

# =======================
# MAIN CAMERA LOOP
# =======================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -----------------------
    # EMOTION ANALYSIS
    # -----------------------
    if time.time() - last_analysis_time > ANALYZE_EVERY:
        analyze_emotion(frame)
        last_analysis_time = time.time()

    # HUD BAR
    cv2.rectangle(frame, (0, 0), (w, 50), COLORS["background"], -1)
    cv2.putText(frame, "AI Emotion + Iris Liveness Detection",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                COLORS["highlight"],
                2)

    # FACE BOX
    box_w, box_h = 260, 300
    x = w // 2 - box_w // 2
    y = h // 2 - box_h // 2
    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), COLORS["highlight"], 2)

    # Emotion label
    label = f"{top_emotion.upper()} ({top_conf}%)"
    cv2.rectangle(frame, (x, y - 40), (x + box_w, y), COLORS["background"], -1)
    cv2.putText(frame, label,
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                COLORS["text"],
                2)

    # Emotion bars
    panel_x, panel_y = 20, 80
    max_bar_w = 200

    for i, emo in enumerate(EMOTIONS):
        y_off = panel_y + i * 35
        score = emotion_scores[emo]
        bar_w = int((score / 100) * max_bar_w)

        cv2.putText(frame, emo.capitalize(),
                    (panel_x, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    COLORS["text"],
                    1)

        cv2.rectangle(frame,
                      (panel_x + 100, y_off - 15),
                      (panel_x + 100 + bar_w, y_off + 5),
                      COLORS["bar_active"] if emo == top_emotion else COLORS["bar_inactive"],
                      -1)

    # -----------------------
    # IRIS LIVENESS
    # -----------------------
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

    iris_result = "Analyzing..."
    iris_color = COLORS["highlight"]

    for (ex, ey, ew, eh) in eyes[:2]:
        eye_region = gray[ey:ey + eh, ex:ex + ew]
        if eye_region.size == 0:
            continue

        iris_result, iris_color = detect_fake_iris(eye_region, threshold)
        eye_center = (ex + ew // 2, ey + eh // 2)
        cv2.circle(frame, eye_center, 5, iris_color, -1)

    cv2.putText(frame,
                iris_result,
                (w - 420, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                iris_color,
                2)

    cv2.imshow("AI Emotion + Iris Liveness", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
