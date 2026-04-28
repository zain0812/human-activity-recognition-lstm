import cv2
import time
import mediapipe as mp

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose

pose = mpPose.Pose(model_complexity=0, smooth_landmarks=True)

video_path = r"C:\Users\zaina\PycharmProjects\Objectdetection-yolo\Video gesture\videos2\V6.mp4"
cap = cv2.VideoCapture(video_path)

# Check video opened
if not cap.isOpened():
    print(" Error: Could not open video:", video_path)
    raise SystemExit

# Match display speed to video FPS
fps_video = cap.get(cv2.CAP_PROP_FPS)
if fps_video <= 0:
    fps_video = 30
delay = int(1000 / fps_video)

# Control output size (smaller = faster)
TARGET_W, TARGET_H = 540, 460

# More visible drawing styles
landmark_style = mpDraw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2)
connection_style = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=2)

ptime = time.time()
frame_id = 0
SKIP = 1  # 1 = process every frame, 2 = every 2nd frame, etc.

while True:
    success, img = cap.read()
    if not success:
        print(" Video ended or cannot read frame.")
        break
   # Resizing the video:
    img = cv2.resize(img, (TARGET_W, TARGET_H))

    frame_id += 1
    if frame_id % SKIP == 0:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)

        # Create list ONCE per frame (stores all 33 landmarks)
        lmlist = []

        if results.pose_landmarks:
            mpDraw.draw_landmarks(
                img,
                results.pose_landmarks,
                mpPose.POSE_CONNECTIONS,
                landmark_drawing_spec=landmark_style,
                connection_drawing_spec=connection_style
            )

            h, w, _ = img.shape

            for idx, lm in enumerate(results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)

                #  Store id + pixel coordinates
                lmlist.append([idx, cx, cy])

                # Optional: show id on frame
                cv2.putText(img, str(idx), (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)

        #  Print the full list once per frame (comment if too much output)
        print("Landmarks list:", lmlist)

        # Example: access a specific landmark (e.g., left shoulder = 11)
        if len(lmlist) > 11:
            print("Left shoulder (id 11):", lmlist[11])

    # FPS on screen
    ctime = time.time()
    fps = 1 / max(ctime - ptime, 1e-6)
    ptime = ctime

    cv2.putText(img, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Image", img)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()