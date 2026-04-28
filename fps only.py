import cv2
import time
import mediapipe as mp

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose

# Faster settings
pose = mpPose.Pose(
    model_complexity=0,     # 0 = fastest
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(r"C:\Users\zaina\PycharmProjects\Objectdetection-yolo\Video gesture\videos2\V1.mp4")

# Match playback speed to the video's real FPS
video_fps = cap.get(cv2.CAP_PROP_FPS)
if video_fps <= 0:
    video_fps = 30
delay = int(1000 / video_fps)

# Smaller size = faster
TARGET_W, TARGET_H = 500, 450  # try 640x360 for even faster

landmark_style = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=4)
connection_style = mpDraw.DrawingSpec(color=(255, 255, 0), thickness=3)

ptime = time.time()

while True:
    success, img = cap.read()
    if not success:
        break

    # Resize BEFORE mediapipe (major speed boost)
    img = cv2.resize(img, (TARGET_W, TARGET_H))

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(
            img, results.pose_landmarks, mpPose.POSE_CONNECTIONS,
            landmark_drawing_spec=landmark_style,
            connection_drawing_spec=connection_style
        )

    # FPS text
    ctime = time.time()
    fps = 1 / max(ctime - ptime, 1e-6)
    ptime = ctime
    cv2.putText(img, f"FPS: {int(fps)}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()