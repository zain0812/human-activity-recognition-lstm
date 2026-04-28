import cv2
import time
import math
import mediapipe as mp
from collections import deque

# -----------------------------
# MediaPipe setup
# -----------------------------
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

# -----------------------------
# Video path
# -----------------------------
video_path = r"C:\Users\zaina\PycharmProjects\Objectdetection-yolo\Video gesture\videos2\V6.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video:", video_path)
    raise SystemExit

# -----------------------------
# Settings
# -----------------------------
TARGET_W, TARGET_H = 400, 300
SKIP = 2
movement_threshold = 5
switch_delay = 0.8
WINDOW_SIZE = 20   # save last 20 frames

# -----------------------------
# Drawing styles
# -----------------------------
landmark_style = mpDraw.DrawingSpec(
    color=(0, 255, 255),   # Yellow
    thickness=3,
    circle_radius=4
)

connection_style = mpDraw.DrawingSpec(
    color=(0, 255, 0),     # Green
    thickness=3,
    circle_radius=2
)

# -----------------------------
# Body region groups
# -----------------------------
BODY_REGIONS = {
    "Head": [0, 2, 5, 7, 8],
    "Left Arm": [11, 13, 15],
    "Right Arm": [12, 14, 16],
    "Torso": [11, 12, 23, 24],
    "Left Leg": [23, 25, 27],
    "Right Leg": [24, 26, 28]
}

# -----------------------------
# Helper functions
# -----------------------------
def draw_text(img, text, pos, scale=0.7, thickness=2):
    x, y = pos
    cv2.putText(
        img, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, scale,
        (0, 0, 0), thickness + 2, cv2.LINE_AA
    )
    cv2.putText(
        img, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, scale,
        (255, 255, 255), thickness, cv2.LINE_AA
    )

def draw_text_bg(img, text, pos, scale=0.7, thickness=2):
    x, y = pos
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(img, (x - 5, y - h - 8), (x + w + 5, y + 5), (50, 50, 50), -1)
    draw_text(img, text, (x, y), scale, thickness)

def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def get_region_center(current_landmarks, region_indices):
    points = []
    for idx in region_indices:
        if idx in current_landmarks:
            points.append(current_landmarks[idx])

    if not points:
        return None

    avg_x = int(sum(p[0] for p in points) / len(points))
    avg_y = int(sum(p[1] for p in points) / len(points))
    return (avg_x, avg_y)

def compute_region_motion(current_landmarks, prev_landmarks, region_indices):
    motions = []

    for idx in region_indices:
        if idx in current_landmarks and idx in prev_landmarks:
            cx, cy = current_landmarks[idx]
            px, py = prev_landmarks[idx]
            move_value = distance(px, py, cx, cy)
            motions.append(move_value)

    if not motions:
        return 0.0

    return sum(motions) / len(motions)

def average_region_motion(history, region_names):
    avg_motion = {}

    for region in region_names:
        values = [frame_data[region] for frame_data in history]
        avg_motion[region] = sum(values) / len(values) if values else 0.0

    return avg_motion

# -----------------------------
# Variables
# -----------------------------
ptime = time.time()
frame_id = 0
last_results = None
prev_landmarks = {}

# Stores the last 20 frame motions
motion_history = deque(maxlen=WINDOW_SIZE)

display_region = "None"
display_motion = 0.0
display_center = None
last_switch_time = time.time()

# -----------------------------
# Main loop
# -----------------------------
while True:
    success, img = cap.read()
    if not success:
        print("Video ended or cannot read frame.")
        break

    img = cv2.resize(img, (TARGET_W, TARGET_H))
    frame_id += 1

    # Process every SKIP frames
    if frame_id % SKIP == 0:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        last_results = pose.process(imgRGB)

    current_landmarks = {}

    if last_results and last_results.pose_landmarks:
        # Draw pose landmarks
        mpDraw.draw_landmarks(
            img,
            last_results.pose_landmarks,
            mpPose.POSE_CONNECTIONS,
            landmark_drawing_spec=landmark_style,
            connection_drawing_spec=connection_style
        )

        h, w, _ = img.shape

        # Save current landmark coordinates
        for idx, lm in enumerate(last_results.pose_landmarks.landmark):
            cx = int(lm.x * w)
            cy = int(lm.y * h)
            current_landmarks[idx] = (cx, cy)

        # Compute motion of each region for CURRENT frame
        region_motions = {}
        for region_name, region_indices in BODY_REGIONS.items():
            region_motions[region_name] = compute_region_motion(
                current_landmarks,
                prev_landmarks,
                region_indices
            )

        # Save this frame's region-motions in history
        motion_history.append(region_motions)

        # Use average of last 20 frames
        if len(motion_history) > 0:
            avg_region_motions = average_region_motion(motion_history, BODY_REGIONS.keys())

            moving_region = None
            max_region_motion = 0.0

            for region_name, motion_value in avg_region_motions.items():
                if motion_value > max_region_motion:
                    max_region_motion = motion_value
                    moving_region = region_name

            current_time = time.time()

            if (
                moving_region is not None
                and max_region_motion > movement_threshold
                and (current_time - last_switch_time) >= switch_delay
            ):
                display_region = moving_region
                display_motion = max_region_motion
                display_center = get_region_center(
                    current_landmarks,
                    BODY_REGIONS[moving_region]
                )
                last_switch_time = current_time

                print(f"Detected: {display_region} | Avg Motion (last {WINDOW_SIZE} frames): {display_motion:.2f}")

        # Draw selected region on frame
        if display_region != "None":
            draw_text_bg(img, f"Moving: {display_region}", (6, 40), scale=0.55, thickness=1)
            draw_text_bg(img, f"Avg Motion: {display_motion:.1f}", (6, 80), scale=0.55, thickness=1)
            draw_text_bg(img, f"Saved Frames: {len(motion_history)}/{WINDOW_SIZE}", (6, 100), scale=0.55, thickness=1)

            if display_center is not None:
                rx, ry = display_center
                cv2.circle(img, (rx, ry), 12, (0, 0, 255), cv2.FILLED)
                draw_text_bg(img, display_region, (rx + 10, ry - 10), scale=0.5, thickness=1)
        else:
            draw_text_bg(img, "Moving: None", (10, 65), scale=0.55, thickness=2)
            draw_text_bg(img, f"Saved Frames: {len(motion_history)}/{WINDOW_SIZE}", (10, 100), scale=0.55, thickness=2)

        # Update previous landmarks for next frame comparison
        prev_landmarks = current_landmarks.copy()

    # FPS
    ctime = time.time()
    fps = 1 / max(ctime - ptime, 1e-6)
    ptime = ctime

    draw_text_bg(img, f"FPS: {int(fps)}", (10, 30), scale=0.7, thickness=2)
    draw_text_bg(img, "Press Q to Quit", (10, TARGET_H - 10), scale=0.5, thickness=1)

    cv2.imshow("Body Region Motion - 20 Frame Window", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()