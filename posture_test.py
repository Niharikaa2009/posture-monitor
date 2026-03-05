import cv2
import mediapipe as mp
import numpy as np
import time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def angle(p1, p2, p3):
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosang = np.clip(cosang, -1, 1)
    return np.degrees(np.arccos(cosang))

cap = cv2.VideoCapture(0)

# Smoothing
smooth_head = None
smooth_shoulder = None
alpha = 0.2

# Bad posture alert
bad_start = None
ALERT_TIME = 15  # seconds

# Flexible thresholds
HEAD_IDEAL = 168
HEAD_RANGE = 7
SHOULDER_IDEAL = 83
SHOULDER_RANGE = 6

# Session tracking
session_start = time.time()
good_time = 0
bad_time = 0
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_h, frame_w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark
        L = lambda x: [lm[x].x, lm[x].y, lm[x].z]

        # Landmarks
        shoulder_L = L(mp_pose.PoseLandmark.LEFT_SHOULDER)
        shoulder_R = L(mp_pose.PoseLandmark.RIGHT_SHOULDER)
        hip_L = L(mp_pose.PoseLandmark.LEFT_HIP)
        hip_R = L(mp_pose.PoseLandmark.RIGHT_HIP)
        ear_L = L(mp_pose.PoseLandmark.LEFT_EAR)
        ear_R = L(mp_pose.PoseLandmark.RIGHT_EAR)

        # Midpoints
        shoulder_mid = (np.array(shoulder_L) + np.array(shoulder_R)) / 2
        hip_mid = (np.array(hip_L) + np.array(hip_R)) / 2
        ear_mid = (np.array(ear_L) + np.array(ear_R)) / 2

        # Angles
        head_angle = angle(hip_mid, shoulder_mid, ear_mid)
        shoulder_angle_L = angle(hip_L, shoulder_L, shoulder_R)
        shoulder_angle_R = angle(hip_R, shoulder_R, shoulder_L)
        shoulder_angle = min(shoulder_angle_L, shoulder_angle_R)

        # Initialize smoothing
        if smooth_head is None:
            smooth_head = head_angle
            smooth_shoulder = shoulder_angle

        # Smooth values
        smooth_head = smooth_head * (1 - alpha) + head_angle * alpha
        smooth_shoulder = smooth_shoulder * (1 - alpha) + shoulder_angle * alpha

        # Flexible classification
        head_good = HEAD_IDEAL - HEAD_RANGE <= smooth_head <= HEAD_IDEAL + HEAD_RANGE
        shoulder_good = SHOULDER_IDEAL - SHOULDER_RANGE <= smooth_shoulder <= SHOULDER_IDEAL + SHOULDER_RANGE

        # Time tracking
        current_time = time.time()
        delta = current_time - prev_time
        prev_time = current_time

        if head_good and shoulder_good:
            status = "GOOD POSTURE"
            color = (0, 255, 0)
            bad_start = None
            good_time += delta
        else:
            status = "BAD POSTURE"
            color = (0, 0, 255)
            bad_time += delta

            if bad_start is None:
                bad_start = time.time()

            elapsed = time.time() - bad_start

            cv2.putText(frame, f"Bad posture: {elapsed:.1f}s", (30, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            if elapsed > ALERT_TIME:
                cv2.putText(frame, "ALERT: Sit up straight!", (30, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        # Posture score
        head_score = max(0, 100 - abs(smooth_head - HEAD_IDEAL) * 4)
        shoulder_score = max(0, 100 - abs(smooth_shoulder - SHOULDER_IDEAL) * 4)
        posture_score = int((head_score + shoulder_score) / 2)

        # UI text
        cv2.putText(frame, f"Posture Score: {posture_score}/100", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        cv2.putText(frame, status, (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        cv2.putText(frame, f"Head Angle: {smooth_head:.1f}°", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.putText(frame, f"Shoulder Tilt: {smooth_shoulder:.1f}°", (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        # Indicator circle
        cv2.circle(frame, (frame_w - 50, 50), 20, color, -1)

        # Progress bars
        head_bar = int(((smooth_head - (HEAD_IDEAL - HEAD_RANGE)) / (2 * HEAD_RANGE)) * 200)
        head_bar = np.clip(head_bar, 0, 200)

        shoulder_bar = int(((smooth_shoulder - (SHOULDER_IDEAL - SHOULDER_RANGE)) / (2 * SHOULDER_RANGE)) * 200)
        shoulder_bar = np.clip(shoulder_bar, 0, 200)

        cv2.rectangle(frame, (30, frame_h - 60),
                      (30 + shoulder_bar, frame_h - 40), (255,0,0), -1)

        cv2.putText(frame, "Shoulder tilt", (30, frame_h - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.rectangle(frame, (30, frame_h - 120),
                      (30 + head_bar, frame_h - 100), (0,255,255), -1)

        cv2.putText(frame, "Head angle", (30, frame_h - 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Posture Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Final session report
session_total = time.time() - session_start

print("\n------ POSTURE SESSION REPORT ------")
print(f"Total session time: {session_total:.1f} seconds")
print(f"Good posture time: {good_time:.1f} seconds")
print(f"Bad posture time: {bad_time:.1f} seconds")

if session_total > 0:
    good_percent = (good_time / session_total) * 100
    print(f"Good posture percentage: {good_percent:.1f}%")

print("-----------------------------------")
