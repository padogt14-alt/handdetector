import cv2
import mediapipe as mp
import math
from collections import Counter

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Indeks landmark MediaPipe
TIP_IDS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky

def finger_states(hand_landmarks, hand_label):
    """
    Mengembalikan list boolean [thumb, index, middle, ring, pinky]
    True = finger terbuka / terentang
    Menggunakan aturan sederhana:
    - Untuk index..pinky: compare tip.y dan pip.y (landmark tip vs tip-2)
    - Untuk thumb: bandingkan x antara tip dan ip tergantung label tangan
    """
    lm = hand_landmarks.landmark
    states = []

    # Index..Pinky: tip.y < pip.y => terbuka (coordinate y lebih kecil = lebih atas di image)
    for tip_id in TIP_IDS[1:]:
        tip_y = lm[tip_id].y
        pip_y = lm[tip_id - 2].y
        states.append(tip_y < pip_y)

    # Thumb: arah x tergantung left/right
    # Jika tangan kanan (hand_label == "Right"), ibu jari umumnya ke kiri di image (x kecil)
    # Rule: thumb_open = tip.x < ip.x for Right, sebaliknya untuk Left
    thumb_tip_x = lm[TIP_IDS[0]].x
    thumb_ip_x = lm[TIP_IDS[0] - 1].x  # 3
    if hand_label == "Right":
        thumb_open = thumb_tip_x < thumb_ip_x
    else:
        thumb_open = thumb_tip_x > thumb_ip_x
    # Insert thumb di depan
    return [thumb_open] + states  # [thumb, index, middle, ring, pinky]


def classify_pose(states):
    """
    states: [thumb, index, middle, ring, pinky] (booleans)
    Kembalikan string label pose.
    Rule sederhana:
    - Open Palm: semua True
    - Fist: semua False
    - Peace: index & middle True, others False
    - Thumbs Up: thumb True, others False
    - Kalau tidak cocok: "Unknown"
    """
    thumb, index, middle, ring, pinky = states

    if all(states):
        return "Open Palm"
    if not any(states):
        return "Fist"
    if index and middle and (not ring) and (not pinky):
        # allow thumb either way for V sign; but prefer thumb False
        if not thumb:
            return "Peace (V)"
        else:
            return "Peace (V)"
    if thumb and (not index) and (not middle) and (not ring) and (not pinky):
        return "Thumbs Up"
    # small heuristics for pointing (index only)
    if index and (not middle) and (not ring) and (not pinky):
        return "Pointing"
    return "Unknown"


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Tidak dapat membuka kamera")
        return

    with mp_hands.Hands(min_detection_confidence=0.6,
                        min_tracking_confidence=0.5,
                        max_num_hands=2) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # mirror supaya lebih natural
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                      results.multi_handedness):
                    label = handedness.classification[0].label  # "Left" or "Right"
                    # hitung states
                    states = finger_states(hand_landmarks, label)
                    pose_label = classify_pose(states)

                    # gambar landmarks & koneksi
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))

                    # dapatkan bounding box sederhana untuk menempatkan teks
                    xs = [lm.x for lm in hand_landmarks.landmark]
                    ys = [lm.y for lm in hand_landmarks.landmark]
                    x_min, x_max = int(min(xs) * w), int(max(xs) * w)
                    y_min, y_max = int(min(ys) * h), int(max(ys) * h)

                    # teks: label tangan + pose
                    text = f"{label} hand: {pose_label}"
                    # Draw rectangle behind text
                    cv2.rectangle(frame, (x_min - 10, y_min - 30), (x_min + 220, y_min), (0, 0, 0), -1)
                    cv2.putText(frame, text, (x_min - 5, y_min - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # instruksi kecil
            cv2.putText(frame, "Tekan 'q' untuk keluar", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

            cv2.imshow("Hand Pose Detector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    import streamlit as st
import cv2
import mediapipe as mp

st.title("Hand Pose Detector")
run = st.checkbox("Start Camera")

if run:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    while True:
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        stframe.image(frame, channels="BGR")


