# AI Posture Detection System

## What this project does
This project detects sitting posture in real time and gives feedback if the posture is incorrect. It also sends an alert if poor posture is maintained continuously for a certain period of time.

The idea came from a simple problem I faced regularly—while studying, I would correct my posture but end up slouching again within minutes. I wanted to see if I could use AI to solve something this common.

---

## How it works
- Uses webcam input for real-time video  
- Detects body landmarks using MediaPipe  
- Calculates relative positions and joint angles  
- Compares these values with predefined thresholds  
- Classifies posture as correct or incorrect  
- Triggers an alert if bad posture is sustained  

---

## Tech used
- Python  
- OpenCV  
- MediaPipe  
- NumPy  

---

## Key challenges
One of the main challenges was defining what “correct posture” actually means. Small variations in movement often led to inconsistent results.

To handle this, I focused on:
- Selecting key body landmarks  
- Using relative positions instead of absolute ones  
- Adjusting thresholds multiple times to improve consistency  

The system required several iterations before it started giving stable results.

---

## What I learned
- How to work with real-time computer vision systems  
- The importance of breaking down a problem into smaller parts  
- How small design choices (like thresholds) affect output significantly  
- That implementing an idea is very different from just understanding it  

---

## Future scope
- Improve accuracy using trained ML models  
- Personalize posture detection for different users  
- Improve feedback and alert system  

---

## Author
Niharika Agarwal
