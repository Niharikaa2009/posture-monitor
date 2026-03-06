# Day 4 — Posture Stability and Noise Reduction

# Goal:

Reduce flickering in posture classification caused by small fluctuations in pose detection.

# Tasks Completed:

Observed that the posture label was rapidly switching between GOOD and BAD even when the user was sitting relatively still.
Identified that small variations in pose landmarks from MediaPipe were causing minor angle changes every frame.
Implemented a frame-consistency mechanism where posture only changes after multiple consecutive frames agree.
Added counters to track consecutive good and bad frames.
Introduced a frame threshold so that posture state changes only after the threshold is exceeded.
Adjusted the threshold value to balance responsiveness and stability.
Tested the system by intentionally making small movements to verify that minor posture shifts no longer immediately change the classification.

# Result:

The posture detection system is now significantly more stable.
Minor fluctuations in head or shoulder angles no longer cause the posture status to flicker between GOOD and BAD.
The detector now responds only to consistent posture changes, making the real-time feedback smoother and more reliable.
This improves the usability of the posture monitoring system during longer sessions.
