import time
import sqlite3
import sys
import os
from flask import Flask, Response, render_template, request, redirect, url_for, jsonify
import cv2
from threading import Thread, Lock

# Ensure the root directory is in sys.path so 'src' can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.face_module import register_face, verify_face_with_gaze, monitor_exam, current_state

app = Flask(__name__)

# ===============================
# EXAM SETTINGS
# ===============================
EXAM_DURATION = 120  # 2 minutes

# ===============================
# DATABASE SETUP
# ===============================
def init_db():
    conn = sqlite3.connect("violations.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            violation_type TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_violation(student_id, violation_type):
    conn = sqlite3.connect("violations.db")
    c = conn.cursor()
    c.execute("INSERT INTO violations (student_id, violation_type) VALUES (?, ?)",
              (student_id, violation_type))
    conn.commit()
    conn.close()

# ===============================
# WEBCAM
# ===============================
video_capture = None
capture_lock = Lock()
monitoring_results = {'status': 'idle', 'message': '', 'warnings': []}

def initialize_video_capture():
    global video_capture
    with capture_lock:
        if video_capture is None or not video_capture.isOpened():
            video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if video_capture.isOpened():
                video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                video_capture.set(cv2.CAP_PROP_FPS, 15)
                print("Webcam initialized for streaming.")
            else:
                print("Error: Cannot initialize webcam.")
                return False
        return True

def release_video_capture():
    global video_capture
    with capture_lock:
        if video_capture is not None and video_capture.isOpened():
            video_capture.release()
            print("Webcam released.")

def gen_frames():
    if not initialize_video_capture():
        return
    last_time = time.time()
    while True:
        with capture_lock:
            if video_capture is None or not video_capture.isOpened():
                break
            ret, frame = video_capture.read()
            if not ret:
                break

        if time.time() - last_time < 0.1:
            continue
        last_time = time.time()

        cv2.putText(frame, f"Gaze: {current_state['gaze']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(frame, f"Head: {current_state['head_pose']}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        frame = cv2.resize(frame, (200, 150))
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ===============================
# MONITORING THREAD
# ===============================
def run_monitoring(student_id, model_choice):
    global monitoring_results
    monitoring_results['status'] = 'running'

    if not initialize_video_capture():
        monitoring_results['status'] = 'completed'
        monitoring_results['message'] = 'Webcam initialization failed.'
        return

    success, message, warnings = monitor_exam(
        student_id,
        video_capture=video_capture,
        duration_seconds=EXAM_DURATION,
        short_violation_duration=2,
        down_violation_duration=5,
        long_violation_duration=10,
        max_warnings=3,
        model_choice=model_choice
    )

    # Save violations to database
    for warning in warnings:
        save_violation(student_id, warning)

    monitoring_results['status'] = 'completed'
    monitoring_results['message'] = message
    monitoring_results['warnings'] = warnings

# ===============================
# ROUTES
# ===============================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/exam')
def exam():
    student_id = request.args.get('student_id', '')
    is_running = (monitoring_results['status'] == 'running')
    return render_template('exam.html', duration=EXAM_DURATION, student_id=student_id, is_running=is_running)

@app.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    global monitoring_results
    student_id = request.form['student_id']
    
    # Get model choice from the form (default to mediapipe)
    model_choice = request.form.get('model_choice', 'mediapipe')
    
    if not student_id:
        return redirect(url_for('result', message='Error: Student ID cannot be empty.'))

    monitoring_results = {'status': 'running', 'message': '', 'warnings': []}
    
    # 🔥 Clear global alerts from previous sessions before starting new thread
    current_state['alerts'].clear()
    current_state['gaze'] = 'forward'
    current_state['head_pose'] = 'forward'
    
    Thread(target=run_monitoring, args=(student_id, model_choice)).start()
    return redirect(url_for('exam'))

@app.route('/get_alerts')
def get_alerts():
    if monitoring_results['status'] == 'completed':
        return jsonify({'alert': monitoring_results['message']})

    if current_state['alerts']:
        # 🔥 Pop the alert so the frontend does not endlessly increment violations!
        return jsonify({'alert': current_state['alerts'].pop(0)})

    return jsonify({'alert': 'No alerts'})

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        student_id = request.form['student_id']
        success = register_face(student_id)
        if success:
            # Redirect directly to verify
            return redirect(url_for('verify', student_id=student_id))
        else:
            return redirect(url_for('result', message=f'Registration failed for {student_id}.'))
    return render_template('register.html')

@app.route('/verify', methods=['GET', 'POST'])
def verify():
    # Pre-fill student_id if passed from register
    student_id_get = request.args.get('student_id', '')
    if request.method == 'POST':
        student_id = request.form['student_id']
        success, message = verify_face_with_gaze(student_id)
        if success:
            return redirect(url_for('result', message=f'Verification successful for {student_id}! You may now proceed.', action_url=url_for('exam', student_id=student_id), action_text="Attempt Exam"))
        else:
            return redirect(url_for('result', message=f'Verification failed: {message}'))
    return render_template('verify.html', student_id=student_id_get)

@app.route('/result')
def result():
    message = request.args.get('message', 'No message provided.')
    action_url = request.args.get('action_url', '')
    action_text = request.args.get('action_text', '')
    return render_template('result.html', message=message, action_url=action_url, action_text=action_text)

if __name__ == '__main__':
    init_db()
    try:
        app.run(debug=True)
    finally:
        release_video_capture()