import cv2
import time
import mediapipe as mp

# MediaPipe setup
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose

# Faster pose model settings
pose = mpPose.Pose(
    static_image_mode=False,
    model_complexity=0,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Video path
video_path = r"C:\Users\zaina\PycharmProjects\Objectdetection-yolo\Video gesture\videos2\V6.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video opened
if not cap.isOpened():
    print("Error: Could not open video:", video_path)
    raise SystemExit

# Smaller frame for faster processing, output frame size:
TARGET_W, TARGET_H = 450, 350

# Skip frames for speed : For fast processing because checking each frame takes times
SKIP = 2   # Process every 2nd frame

# High-visibility drawing styles
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
#Time and counters
ptime = time.time()
frame_id = 0
last_results = None


# Function for readable outlined text
def draw_text(img, text, pos, scale=0.7, thickness=2):
    x, y = pos
    # Black outline
    cv2.putText(
        img, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, scale,
        (0, 0, 0), thickness + 2, cv2.LINE_AA
    )
    # White main text
    cv2.putText(
        img, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, scale,
        (255, 255, 255), thickness, cv2.LINE_AA
    )


# Function for text with background box
def draw_text_bg(img, text, pos, scale=0.7, thickness=2):
    x, y = pos
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)

    # Background rectangle
    cv2.rectangle(
        img,
        (x - 5, y - h - 8),
        (x + w + 5, y + 5),
        (50, 50, 50),
        -1
    )

    # Text with outline
    draw_text(img, text, (x, y), scale, thickness)


while True:
    success, img = cap.read()
    if not success:
        print("Video ended or cannot read frame.")
        break

    # Resize for faster processing
    img = cv2.resize(img, (TARGET_W, TARGET_H))
    frame_id += 1

    # Only process every SKIP frames
    if frame_id % SKIP == 0:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        last_results = pose.process(imgRGB)

    lmlist = []

    # Draw last detected pose result
    if last_results and last_results.pose_landmarks:
        mpDraw.draw_landmarks(
            img,
            last_results.pose_landmarks,
            mpPose.POSE_CONNECTIONS,
            landmark_drawing_spec=landmark_style,
            connection_drawing_spec=connection_style
        )

        h, w, _ = img.shape

        for idx, lm in enumerate(last_results.pose_landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)  #Convert normalized coordinates to pixel coordinates
            lmlist.append([idx, cx, cy])

        # Example: highlight left shoulder (id 11)
        if len(lmlist) > 11:
            shoulder_x, shoulder_y = lmlist[11][1], lmlist[11][2]

            # Red highlight circle
            cv2.circle(img, (shoulder_x, shoulder_y), 8, (0, 0, 255), cv2.FILLED)

            # Label near shoulder
            draw_text_bg(img, "Left Shoulder", (shoulder_x + 10, shoulder_y - 10), scale=0.5, thickness=1)

    # FPS calculation
    ctime = time.time()
    fps = 1 / max(ctime - ptime, 1e-6)
    ptime = ctime

    # Show FPS
    draw_text_bg(img, f"FPS: {int(fps)}", (10, 30), scale=0.7, thickness=2)

    # Show instruction
    draw_text_bg(img, "Press Q to Quit", (10, TARGET_H - 10), scale=0.6, thickness=1)

    cv2.imshow("Pose Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()