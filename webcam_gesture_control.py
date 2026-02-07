import cv2
import mediapipe as mp
from pathlib import Path

# Correct imports from MediaPipe Tasks
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
ImageFormat = mp.ImageFormat
Image = mp.Image

# Load the model
model_path = Path("hand_landmarker.task").resolve()
if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}")

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(model_path)),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)
landmarker = HandLandmarker.create_from_options(options)


# Gesture Recognition

def recognize_gesture(landmarks):
    fingers = []

    # Thumb
    if landmarks[4].x < landmarks[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        if landmarks[tip].y < landmarks[pip].y:            #If finger is above middle joint then considers it as the used finger otherwise discards it
            fingers.append(1)
        else:
            fingers.append(0)

    # Finger pattern mapping [Thumb,index,middle,ring,small]
    if fingers == [0,0,0,0,0]: return "Stop"
    if fingers == [1,1,1,1,1]: return "Start"
    if fingers == [0,1,0,0,0]: return "Forward"
    if fingers == [0,1,1,0,0]: return "Backward"
    if fingers == [1,1,0,0,0]: return "Flip"
    if fingers == [1,0,0,0,0]: return "Up"
    if fingers == [0,1,1,1,0]: return "Down"
    if fingers == [0,1,1,1,1]: return "Hover"
    return "Unknown"

# To Start webcam

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # mirror
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to MediaPipe Image
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)

    # Detect hands
    result = landmarker.detect_for_video(mp_image, timestamp_ms=int(cap.get(cv2.CAP_PROP_POS_MSEC)))

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            landmarks = hand  # hand is already a list of 21 landmarks for tips, wrist and joints(1*5 tips+3*5 finger joints+1 wrist joint)
            h, w, _ = frame.shape

            # Draw landmarks
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0,255,0), -1)

            # Recognises gesture
            gesture = recognize_gesture(landmarks)
            cv2.putText(frame, f"Gesture: {gesture}", (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Hand Gesture Control", frame)
    if cv2.waitKey(1) == ord(" "): #Press SPACEBAR to exit the capture
        break

cap.release()
cv2.destroyAllWindows()
