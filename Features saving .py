import cv2
import math
import time
import os
import numpy as np
import mediapipe as mp

# LSTM DATASET COLLECTION FILE
# This file extracts pose-based features from one labeled video
# and saves them in a separate .npz dataset file.
#
# Run this file multiple times:
#   1. squat video   -> label 0
#   2. curl video    -> label 1
#   3. handsup video -> label 2
#   4. idle video    -> label 3
#
# Output examples:
#   squat_sequences.npz
#   curl_sequences.npz
#   handsup_sequences.npz
#   idle_sequences.npz


# video path:

video_path = r"C:\Users\zaina\PycharmProjects\Objectdetection-yolo\Video gesture\videos2\V1.mp4"

# Change these for each action
action_label_name = "idle"   # squat / curl / handsup / idle
action_label_id = 3        # squat=0, curl=1, handsup=2, idle=3

# Output save folder
save_dir = r"C:\Users\zaina\PycharmProjects\Objectdetection-yolo\lstm_dataset"

# Sequence settings
SEQUENCE_LENGTH = 30
SKIP = 2
TARGET_W, TARGET_H = 640, 480

# MediaPipe setup
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose

pose = mpPose.Pose(
    static_image_mode=False,
    model_complexity=0,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# color palette
COLORS = {
    "landmark": (80, 220, 255),
    "connection": (0, 165, 255),
    "text": (245, 245, 245),
    "text_shadow": (30, 30, 30),
    "box": (40, 40, 40),
    "fps": (0, 200, 255),
    "title": (255, 220, 120),
    "info": (0, 255, 255),
    "angle_arm": (80, 220, 255),
    "angle_leg": (120, 255, 120)
}

# Drawing styles

landmark_style = mpDraw.DrawingSpec(
    color=COLORS["landmark"],
    thickness=2,
    circle_radius=3
)

connection_style = mpDraw.DrawingSpec(
    color=COLORS["connection"],
    thickness=2,
    circle_radius=2
)

# Helper functions

def draw_text(img, text, pos, scale=0.6, thickness=2, color=(245, 245, 245)):
    x, y = pos
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        COLORS["text_shadow"],
        thickness + 2,
        cv2.LINE_AA
    )
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )

def draw_text_bg(img, text, pos, scale=0.6, thickness=2, color=(245, 245, 245)):
    x, y = pos
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(img, (x - 6, y - h - 10), (x + w + 6, y + 6), COLORS["box"], -1)
    draw_text(img, text, (x, y), scale, thickness, color)

def calculate_angle(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c

    abx, aby = ax - bx, ay - by
    cbx, cby = cx - bx, cy - by

    dot = abx * cbx + aby * cby
    mag_ab = math.sqrt(abx ** 2 + aby ** 2)
    mag_cb = math.sqrt(cbx ** 2 + cby ** 2)

    if mag_ab == 0 or mag_cb == 0:
        return 0.0

    cos_angle = max(-1.0, min(1.0, dot / (mag_ab * mag_cb)))
    angle = math.degrees(math.acos(cos_angle))
    return angle

def normalize_y(y_pixel, frame_height):
    return y_pixel / frame_height

# Create output folder

os.makedirs(save_dir, exist_ok=True)
save_file = os.path.join(save_dir, f"{action_label_name}_sequences.npz")


# Video load

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video:", video_path)
    raise SystemExit

fps_video = cap.get(cv2.CAP_PROP_FPS)
if fps_video <= 0:
    fps_video = 30

print("LSTM DATASET COLLECTION STARTED")
print("Video Path:", video_path)
print("Action Label Name:", action_label_name)
print("Action Label ID:", action_label_id)
print("Video FPS:", fps_video)
print("Frame Skip:", SKIP)
print("Sequence Length:", SEQUENCE_LENGTH)
print("Output File:", save_file)

# Variables
ptime = time.time()
frame_id = 0
processed_frame_count = 0
last_results = None

frame_features_list = []
X_sequences = []
y_labels = []

# Main loop
while True:
    success, img = cap.read()
    if not success:
        print("Video ended or cannot read frame.")
        break

    img = cv2.resize(img, (TARGET_W, TARGET_H))
    frame_id += 1

    # Process only every SKIP frames
    if frame_id % SKIP != 0:
        continue

    processed_frame_count += 1

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    last_results = pose.process(imgRGB)

    if last_results and last_results.pose_landmarks:
        mpDraw.draw_landmarks(
            img,
            last_results.pose_landmarks,
            mpPose.POSE_CONNECTIONS,
            landmark_drawing_spec=landmark_style,
            connection_drawing_spec=connection_style
        )

        h, w, _ = img.shape
        pts = {}

        # Store all landmarks as pixel coordinates
        for idx, lm in enumerate(last_results.pose_landmarks.landmark):
            cx = int(lm.x * w)
            cy = int(lm.y * h)
            pts[idx] = (cx, cy)


        # Feature Extraction
        left_elbow_angle = 0.0
        right_elbow_angle = 0.0
        left_knee_angle = 0.0
        right_knee_angle = 0.0
        avg_knee_angle = 0.0

        left_wrist_y = 0.0
        right_wrist_y = 0.0
        left_shoulder_y = 0.0
        right_shoulder_y = 0.0

        # Left elbow angle: 11-13-15
        if 11 in pts and 13 in pts and 15 in pts:
            left_elbow_angle = calculate_angle(pts[11], pts[13], pts[15])

        # Right elbow angle: 12-14-16
        if 12 in pts and 14 in pts and 16 in pts:
            right_elbow_angle = calculate_angle(pts[12], pts[14], pts[16])

        # Left knee angle: 23-25-27
        if 23 in pts and 25 in pts and 27 in pts:
            left_knee_angle = calculate_angle(pts[23], pts[25], pts[27])

        # Right knee angle: 24-26-28
        if 24 in pts and 26 in pts and 28 in pts:
            right_knee_angle = calculate_angle(pts[24], pts[26], pts[28])

        # Average knee angle
        if left_knee_angle > 0 and right_knee_angle > 0:
            avg_knee_angle = (left_knee_angle + right_knee_angle) / 2.0

        # Normalized Y positions
        if 15 in pts:
            left_wrist_y = normalize_y(pts[15][1], h)
        if 16 in pts:
            right_wrist_y = normalize_y(pts[16][1], h)
        if 11 in pts:
            left_shoulder_y = normalize_y(pts[11][1], h)
        if 12 in pts:
            right_shoulder_y = normalize_y(pts[12][1], h)


        # Final feature vector (9 features)
        # Angles are normalized to 0-1 by dividing by 180

        frame_features = [
            left_elbow_angle / 180.0,
            right_elbow_angle / 180.0,
            left_knee_angle / 180.0,
            right_knee_angle / 180.0,
            avg_knee_angle / 180.0,
            left_wrist_y,
            right_wrist_y,
            left_shoulder_y,
            right_shoulder_y
        ]

        frame_features_list.append(frame_features)

        # Build LSTM sequences
        if len(frame_features_list) >= SEQUENCE_LENGTH:
            sequence = frame_features_list[-SEQUENCE_LENGTH:]
            X_sequences.append(sequence)
            y_labels.append(action_label_id)

        # -----------------------------
        # Display information
        # -----------------------------
        draw_text_bg(img, "LSTM Feature Collection", (10, 30), scale=0.7, thickness=2, color=COLORS["title"])
        draw_text_bg(img, f"Label: {action_label_name} ({action_label_id})", (10, 65), scale=0.6, thickness=2, color=COLORS["info"])
        draw_text_bg(img, f"Processed Frames: {processed_frame_count}", (10, 100), scale=0.55, thickness=1, color=COLORS["text"])
        draw_text_bg(img, f"Stored Sequences: {len(X_sequences)}", (10, 130), scale=0.55, thickness=1, color=COLORS["text"])

        draw_text_bg(img, f"L Elbow: {int(left_elbow_angle)}", (10, 170), scale=0.5, thickness=1, color=COLORS["angle_arm"])
        draw_text_bg(img, f"R Elbow: {int(right_elbow_angle)}", (10, 195), scale=0.5, thickness=1, color=COLORS["angle_arm"])
        draw_text_bg(img, f"L Knee: {int(left_knee_angle)}", (10, 220), scale=0.5, thickness=1, color=COLORS["angle_leg"])
        draw_text_bg(img, f"R Knee: {int(right_knee_angle)}", (10, 245), scale=0.5, thickness=1, color=COLORS["angle_leg"])
        draw_text_bg(img, f"Avg Knee: {int(avg_knee_angle)}", (10, 270), scale=0.5, thickness=1, color=COLORS["title"])

    # -----------------------------
    # FPS display
    # -----------------------------
    ctime = time.time()
    fps = 1 / max(ctime - ptime, 1e-6)
    ptime = ctime

    draw_text_bg(img, f"FPS: {int(fps)}", (10, TARGET_H - 15), scale=0.55, thickness=1, color=COLORS["fps"])

    cv2.imshow("LSTM Dataset Collection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# Save output
# -----------------------------
cap.release()
cv2.destroyAllWindows()

X_sequences = np.array(X_sequences, dtype=np.float32)
y_labels = np.array(y_labels, dtype=np.int32)

print("========================================")
print("DATA COLLECTION FINISHED")
print("X shape:", X_sequences.shape)
print("y shape:", y_labels.shape)

np.savez(
    save_file,
    X=X_sequences,
    y=y_labels,
    label_name=action_label_name,
    label_id=action_label_id,
    sequence_length=SEQUENCE_LENGTH,
    feature_count=9
)

print("Saved successfully:")
print(save_file)
print("========================================")