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
# Professional color palette
# -----------------------------
COLORS = {
    "landmark": (80, 220, 255),     # soft cyan
    "connection": (0, 165, 255),    # orange
    "text": (245, 245, 245),        # near white
    "text_shadow": (30, 30, 30),    # dark shadow
    "box": (40, 40, 40),            # dark gray
    "highlight": (0, 90, 255),      # red-orange
    "fps": (0, 200, 255),           # amber/cyan
    "angle_arm": (80, 220, 255),    # cyan
    "angle_leg": (120, 255, 120),   # soft green
    "angle_title": (255, 220, 120)  # warm light orange
}

# -----------------------------
# Drawing styles
# -----------------------------
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

# -----------------------------
# Helper functions
# -----------------------------
def draw_text(img, text, pos, scale=0.6, thickness=2, color=(245, 245, 245)):
    x, y = pos
    cv2.putText(
        img, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, scale,
        COLORS["text_shadow"], thickness + 2, cv2.LINE_AA
    )
    cv2.putText(
        img, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, scale,
        color, thickness, cv2.LINE_AA
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
TARGET_W, TARGET_H = 640, 480
SKIP = 2

# -----------------------------
# Variables
# -----------------------------
ptime = time.time()
frame_id = 0
last_results = None

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
        pts = {}
        for idx, lm in enumerate(last_results.pose_landmarks.landmark):
            cx = int(lm.x * w)
            cy = int(lm.y * h)
            pts[idx] = (cx, cy)

        # -----------------------------
        # Step 1: Calculate Angles
        # -----------------------------
        draw_text_bg(img, "Joint Angles", (10, 65), scale=0.65, thickness=2, color=COLORS["angle_title"])

        # LEFT ELBOW → 11-13-15
        if 11 in pts and 13 in pts and 15 in pts:
            left_elbow_angle = calculate_angle(pts[11], pts[13], pts[15])
            draw_text_bg(
                img,
                f"L Elbow: {int(left_elbow_angle)} deg",
                (10, 100),
                scale=0.55,
                thickness=1,
                color=COLORS["angle_arm"]
            )

        # RIGHT ELBOW → 12-14-16
        if 12 in pts and 14 in pts and 16 in pts:
            right_elbow_angle = calculate_angle(pts[12], pts[14], pts[16])
            draw_text_bg(
                img,
                f"R Elbow: {int(right_elbow_angle)} deg",
                (10, 130),
                scale=0.55,
                thickness=1,
                color=COLORS["angle_arm"]
            )

        # LEFT KNEE → 23-25-27
        if 23 in pts and 25 in pts and 27 in pts:
            left_knee_angle = calculate_angle(pts[23], pts[25], pts[27])
            draw_text_bg(
                img,
                f"L Knee: {int(left_knee_angle)} deg",
                (10, 160),
                scale=0.55,
                thickness=1,
                color=COLORS["angle_leg"]
            )

        # RIGHT KNEE → 24-26-28
        if 24 in pts and 26 in pts and 28 in pts:
            right_knee_angle = calculate_angle(pts[24], pts[26], pts[28])
            draw_text_bg(
                img,
                f"R Knee: {int(right_knee_angle)} deg",
                (10, 190),
                scale=0.55,
                thickness=1,
                color=COLORS["angle_leg"]
            )

        # Optional highlight points for important joints
        important_joints = [13, 14, 25, 26]
        for joint_id in important_joints:
            if joint_id in pts:
                jx, jy = pts[joint_id]
                cv2.circle(img, (jx, jy), 6, COLORS["highlight"], cv2.FILLED)

    # -----------------------------
    # FPS
    # -----------------------------
    ctime = time.time()
    fps = 1 / max(ctime - ptime, 1e-6)
    ptime = ctime

    draw_text_bg(img, f"FPS: {int(fps)}", (10, 30), scale=0.65, thickness=2, color=COLORS["fps"])
    draw_text_bg(img, "Press Q to Quit", (10, TARGET_H - 15), scale=0.5, thickness=1, color=COLORS["text"])

    cv2.imshow("Pose + Angle Features", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()