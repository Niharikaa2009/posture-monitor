import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def angle(p1, p2, p3):
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc))
    cosang = np.clip(cosang, -1, 1)
    return np.degrees(np.arccos(cosang))

cap = cv2.VideoCapture(0)

smooth_head = None
smooth_shoulder = None
alpha = 0.2   # smoothing amount

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark
        L = lambda x: [lm[x].x, lm[x].y, lm[x].z]

        shoulder_L = L(mp_pose.PoseLandmark.LEFT_SHOULDER)
        shoulder_R = L(mp_pose.PoseLandmark.RIGHT_SHOULDER)
        hip_L = L(mp_pose.PoseLandmark.LEFT_HIP)
        hip_R = L(mp_pose.PoseLandmark.RIGHT_HIP)
        ear_L = L(mp_pose.PoseLandmark.LEFT_EAR)
        ear_R = L(mp_pose.PoseLandmark.RIGHT_EAR)

        shoulder_mid = (np.array(shoulder_L) + np.array(shoulder_R)) / 2
        hip_mid = (np.array(hip_L) + np.array(hip_R)) / 2
        ear_mid = (np.array(ear_L) + np.array(ear_R)) / 2

        head_angle = angle(hip_mid, shoulder_mid, ear_mid)
        shoulder_angle = angle(hip_L, shoulder_L, shoulder_R)

        if smooth_head is None:
            smooth_head = head_angle
            smooth_shoulder = shoulder_angle

        smooth_head = smooth_head * (1 - alpha) + head_angle * alpha
        smooth_shoulder = smooth_shoulder * (1 - alpha) + shoulder_angle * alpha

        # YOUR PERSONALIZED THRESHOLDS
        HEAD_THRESHOLD = 165     # < 165 = head forward
        SHOULDER_THRESHOLD = 82  # < 82 = slouched shoulders

        if smooth_head < HEAD_THRESHOLD or smooth_shoulder < SHOULDER_THRESHOLD:
            status = "BAD POSTURE"
            color = (0, 0, 255)
        else:
            status = "GOOD POSTURE"
            color = (0, 255, 0)

        cv2.putText(frame, status, (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Posture Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
