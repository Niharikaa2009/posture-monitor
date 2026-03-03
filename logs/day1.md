# Day 1 – Posture Monitor Project Log

## What I did today
- Decided the project idea and the main goal: a posture monitoring tool using computer vision.
- Set up the initial project folder structure.
- Explored resources (OpenCV, Mediapipe, Pose Estimation).
- Understood that the project will use pose landmarks to detect back/neck posture.
- Watched demos of Mediapipe Pose to learn how posture angles can be calculated.

## What I learned
- Mediapipe Pose can track major body joints like shoulders, hips, and spine alignment.
- Posture detection is basically about calculating angles between these points.
- A basic prototype will require webcam input + angle calculation + alert system.

## Next Steps
- Write the first prototype that reads webcam frames and identifies pose landmarks.
- Calculate the shoulder–hip line angle.
- Display “Good posture” / “Bad posture” on screen.
- Later: integrate audio alerts or statistics.