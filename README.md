---
title: AI Proctoring System
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# AI Proctoring System

An exam monitoring system that runs face recognition, gaze tracking, and object detection in real time.

## How to use

1. Go to **Register** — enter your student ID and take a photo to register your face
2. Go to **Verify** — verify your identity before the exam
3. Go to **Exam** — the system monitors your gaze, head pose, and detects prohibited objects during the exam

## Admin pages

- `/students` — see all registered students with their photos
- `/activity_log?student_id=YOUR_ID` — full second-by-second exam timeline
- `/access_log` — who accessed the site and when

## Notes

- This hosted version uses **MediaPipe** for gaze tracking
- Face encodings are stored per-session (not persistent between restarts on free tier)
- For full local deployment with custom MobileNet model, see the [GitHub repo](https://github.com/PavanKumarKillana/ai-proctoring-system)