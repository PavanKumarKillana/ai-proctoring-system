# AI Proctoring System

Final year project — an exam monitoring system that runs entirely on a local network. No cloud, no subscriptions. Built with Python, Flask, and React Native.

---

## What it does

Students connect to a WiFi hotspot from their phones or laptops. The system watches them through the camera during an exam and flags if they:
- look away from the screen for too long
- turn their head to the side
- have someone else in the frame
- pull out a phone or open a book

Five warnings and the exam session ends automatically.

---

## How it works

```
Android App  →  HTTP  →  Flask Server  →  MediaPipe + YOLO  →  SQLite
```

The phone app (React Native + Expo) runs a hidden native camera that captures frames every 100ms. These are sent to the Flask server over WiFi. The server runs MediaPipe face mesh and YOLOv8 on each frame and sends back the result. The app injects that result into a WebView showing the exam page.

We went with this approach instead of real WebRTC because Android blocks camera access inside WebViews on plain HTTP. The native camera workaround means it works on any Android phone without HTTPS setup.

---

## Project structure

```
├── src/
│   ├── face_module.py          main AI logic (face detection, gaze, DB)
│   ├── webrtc_processor.py     per-student exam session handler
│   └── custom_gaze_tracker.py  MobileNetV2 gaze model (alternative)
│
├── ui/
│   ├── app.py                  Flask server + all routes
│   ├── static/style.css
│   └── templates/
│       ├── register.html       student registration with quality meter
│       ├── verify.html         face verification
│       ├── exam.html           live exam monitoring UI
│       ├── students.html       admin - all registered students
│       ├── activity_log.html   admin - per-exam event timeline
│       └── access_log.html     admin - who accessed the site
│
├── tests/                      unit tests
└── requirements.txt
```

The mobile app is on the `mobile` branch.

---

## Stack

- **Flask** — web server
- **face_recognition** (dlib) — face encoding and comparison
- **MediaPipe** — 468-point face mesh for gaze + head pose
- **YOLOv8n** — object detection (phones, books etc.)
- **PyTorch MobileNetV2** — custom trained gaze classifier
- **SQLite** — exam logs and access logs
- **React Native + Expo** — Android app

---

## Setup

```bash
# clone and install deps
pip install -r requirements.txt

# run
cd ui
python -m ui.app
```

Server starts on port 7860. Connect a phone to your laptop's hotspot and open `http://192.168.137.1:7860`.

You'll need `yolov8n.pt` in the root directory (downloads automatically first run via ultralytics).

---

## Features added in this version

- Image quality check (brightness + sharpness) before face encoding runs — rejects bad photos with a clear reason
- Live quality bar in browser that blocks submission if photo is too dark or blurry
- Rate limiting on registration and verification endpoints (5 req/min per IP)
- HTTP security headers via flask-talisman
- Second-by-second exam activity log — not just violations, everything
- Each exam attempt tracked by unique session ID
- Admin dashboards: registered students with photos, exam timeline, site access log

---

## Admin pages

| Page | URL |
|---|---|
| Student registry | `/students` |
| Exam activity timeline | `/activity_log?student_id=X` |
| Site access log | `/access_log` |

---

## Team

B.Tech CSE — Amrita School of Engineering

Built for the AI/ML lab project submission.