import face_recognition
import cv2
import pickle
import os
import time
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import sqlite3
import uuid
from contextlib import contextmanager

from src.custom_gaze_tracker import CustomPyTorchGazeTracker

# Global state for sharing with UI (gaze and alerts)
current_state = {'gaze': 'unknown', 'head_pose': 'forward', 'alerts': []}

def initialize_webcam(max_retries=3, delay=1):
    """Initialize webcam with retries and multiple indices."""
    for attempt in range(max_retries):
        for index in [0, 1]:
            video_capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if video_capture.isOpened():
                video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                video_capture.set(cv2.CAP_PROP_FPS, 15)
                print(f"Webcam initialized on index {index}.")
                return video_capture
            print(f"Webcam initialization failed on index {index}, attempt {attempt + 1}/{max_retries}...")
            time.sleep(delay)
    print("Error: Webcam access failed after retries.")
    return None

# ─────────────────────────────────────────────────────────────────
# IMAGE QUALITY CHECK  (NEW in Production_2)
# ─────────────────────────────────────────────────────────────────
def check_image_quality(image):
    """
    Check brightness and sharpness of an image before face encoding.

    Returns:
        (ok: bool, message: str, brightness: float, blur_score: float)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    blur_score  = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if brightness < 40:
        return False, f"Image too dark (brightness={brightness:.0f}/255). Move to better lighting.", brightness, blur_score
    if brightness > 235:
        return False, f"Image overexposed (brightness={brightness:.0f}/255). Reduce glare or bright background.", brightness, blur_score
    if blur_score < 30:
        return False, f"Image too blurry (sharpness={blur_score:.0f}). Hold still and try again.", brightness, blur_score

    return True, "Image quality OK.", brightness, blur_score


def detect_faces(frame, model="cnn", scale=0.5, retries=3):
    """Detect faces with retries and scale fallback."""
    for attempt in range(retries):
        try:
            small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model=model)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            if face_encodings:
                return face_locations, face_encodings, scale
            print(f"Face detection attempt {attempt + 1}/{retries} failed with {model} model at scale {scale}.")
            if attempt == retries - 1 and scale > 0.25:
                print(f"Reducing scale to 0.25 for better detection.")
                return detect_faces(frame, model=model, scale=0.25, retries=1)
        except Exception as e:
            print(f"Face detection error: {e}")
    return [], [], scale

def get_gaze_direction(face_landmarks, frame_width, frame_height, calibration_offset=(0, 0)):
    """Estimate gaze direction with calibration offset."""
    left_eye = face_landmarks.landmark[33]
    right_eye = face_landmarks.landmark[263]
    nose_tip = face_landmarks.landmark[1]

    eye_center_x = (left_eye.x + right_eye.x) / 2 * frame_width
    eye_center_y = (left_eye.y + right_eye.y) / 2 * frame_height
    nose_x = nose_tip.x * frame_width
    nose_y = nose_tip.y * frame_height

    x_diff = (eye_center_x - nose_x) - calibration_offset[0]
    y_diff = (eye_center_y - nose_y) - calibration_offset[1]

    print(f"Eye center: ({eye_center_x:.2f}, {eye_center_y:.2f}), Nose: ({nose_x:.2f}, {nose_y:.2f}), x_diff: {x_diff:.2f}, y_diff: {y_diff:.2f}")

    # Adjusted thresholds for 480x640 frames
    if y_diff > 10.0:
        return "down"
    elif y_diff < -15.0:
        return "up"
    
    if abs(x_diff) > 12.0:
        return "left" if x_diff > 0 else "right"
    return "forward"

def calibrate_gaze(video_capture, max_attempts=50, timeout_seconds=10):
    """Calibrate gaze by capturing forward gaze baseline, discarding outliers."""
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)
    start_time = time.time()
    x_diffs, y_diffs = [], []
    frame_count = 0

    print("Look straight at the camera for calibration (10 seconds). Keep head steady. Press 'q' to quit.")
    try:
        while (time.time() - start_time) < timeout_seconds and frame_count < max_attempts:
            ret, frame = video_capture.read()
            if not ret:
                print("Calibration: Failed to capture frame. Retrying...")
                continue
            frame_height, frame_width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    left_eye = face_landmarks.landmark[33]
                    right_eye = face_landmarks.landmark[263]
                    nose_tip = face_landmarks.landmark[1]
                    eye_center_x = (left_eye.x + right_eye.x) / 2 * frame_width
                    eye_center_y = (left_eye.y + right_eye.y) / 2 * frame_height
                    nose_x = nose_tip.x * frame_width
                    nose_y = nose_tip.y * frame_height
                    x_diff = eye_center_x - nose_x
                    y_diff = eye_center_y - nose_y
                    
                    x_diffs.append(x_diff)
                    y_diffs.append(y_diff)
                    
                    frame_count += 1
                    for idx in [33, 263, 1]:
                        x = int(face_landmarks.landmark[idx].x * frame_width)
                        y = int(face_landmarks.landmark[idx].y * frame_height)
                        color = (0, 255, 255) if idx in [33, 263] else (255, 0, 255)
                        cv2.circle(frame, (x, y), 3, color, -1)
            cv2.putText(frame, "Calibrating... Look forward, keep head steady", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow('Calibration', frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                print("Calibration cancelled.")
                break
        if frame_count > 0:
            cv2.imwrite(f'data/debug_calibration_{int(time.time())}.jpg', frame)
    finally:
        face_mesh.close()
        cv2.destroyAllWindows()

    if x_diffs and y_diffs:
        offset = (np.median(x_diffs), np.median(y_diffs))
        print(f"Calibration completed: offset x={offset[0]:.2f}, y={offset[1]:.2f}")
        return offset
    print("Calibration failed: No valid landmarks detected.")
    return 0, 0

def register_face(student_id, max_attempts=200, timeout_seconds=120, model="cnn"):
    """Registers student's face by capturing and encoding it."""
    video_capture = initialize_webcam()
    if not video_capture:
        print("Registration failed due to webcam error.")
        return False

    print(f"Look at the camera. Press 's' to save (one face only). Press 'q' to quit. Using {model} model.")
    start_time = time.time()
    attempts = 0
    last_debug_save = 0
    scale = 0.5

    try:
        while attempts < max_attempts and (time.time() - start_time) < timeout_seconds:
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Failed to capture frame. Retrying...")
                continue

            face_locations, face_encodings, used_scale = detect_faces(frame, model=model, scale=scale)
            print(f"Frame size: {frame.shape}, Faces detected: {len(face_encodings)}, Scale: {used_scale}")

            for i, (top, right, bottom, left) in enumerate(face_locations):
                top = int(top / used_scale)
                right = int(right / used_scale)
                bottom = int(bottom / used_scale)
                left = int(left / used_scale)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f"Face {i+1}", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            feedback = "Ensure bright lighting, face centered, 1-2 ft from camera, no glare on glasses, plain background"
            cv2.putText(frame, feedback, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow('Registration', frame)

            key = cv2.waitKey(20) & 0xFF
            
            if len(face_encodings) == 1:
                cv2.waitKey(500) 
                
                encoding = face_encodings[0]
                os.makedirs('data', exist_ok=True)
                try:
                    with open(f'data/{student_id}_encoding.pkl', 'wb') as f:
                        pickle.dump(encoding, f)
                    print(f"Registered {student_id} automatically.")
                    return True
                except Exception as e:
                    print(f"Error saving encoding: {e}")
                    return False
            
            if key == ord('q'):
                print("Registration cancelled by user.")
                return False
            elif len(face_encodings) != 1:
                print(f"Detected {len(face_encodings)} faces. Ensure exactly one face is visible.")
                current_time = time.time()
                if current_time - last_debug_save >= 3:
                    cv2.imwrite(f'data/debug_no_face_{int(current_time)}.jpg', frame)
                    last_debug_save = current_time

            attempts += 1

        print("Registration timed out or max attempts reached.")
        return False

    finally:
        video_capture.release()
        cv2.destroyAllWindows()

def verify_face_with_gaze(student_id, tolerance=0.5, max_attempts=300, timeout_seconds=60, model="cnn"):
    """Verify student's face and monitor gaze."""
    encoding_file = f'data/{student_id}_encoding.pkl'
    if not os.path.exists(encoding_file):
        print(f"Error: No encoding found for {student_id}.")
        return False, "No encoding found."

    video_capture = initialize_webcam()
    if not video_capture:
        print("Verification failed due to webcam error.")
        return False, "Webcam error."

    calibration_offset = calibrate_gaze(video_capture)
    print(f"Calibration offset: x={calibration_offset[0]:.2f}, y={calibration_offset[1]:.2f}")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)
    yolo_model = YOLO('yolov8n.pt')

    face_missing_start = None
    face_missing_count = 0
    multi_face_start = None
    gaze_violation_start = None
    gaze_violation_count = 0
    object_violation_start = None
    object_violation_count = 0
    last_gaze = None
    last_debug_save = 0
    start_time = time.time()
    warnings = []

    print(f"Look at the camera for verification (60 seconds). Press 'q' to quit. Using {model} model.")
    try:
        while (time.time() - start_time) < timeout_seconds:
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Failed to capture frame. Retrying...")
                continue

            frame_height, frame_width = frame.shape[:2]
            face_locations, face_encodings, used_scale = detect_faces(frame, model=model, scale=0.5)
            print(f"Frame size: {frame.shape}, Faces detected: {len(face_encodings)}, Scale: {used_scale}")

            if len(face_encodings) == 0:
                if face_missing_start is None:
                    face_missing_start = time.time()
                else:
                    face_missing_duration = time.time() - face_missing_start
                    if face_missing_duration > 10:
                        face_missing_count += 1
                        face_missing_start = time.time()
                        cv2.imwrite(f'data/debug_face_missing_{int(time.time())}.jpg', frame)
                        if face_missing_count >= 3:
                            return False, "Repeated face missing violations."
            else:
                face_missing_start = None

            yolo_results = yolo_model(frame)
            person_count = sum(1 for r in yolo_results for box in r.boxes if yolo_model.names[int(box.cls)] == 'person' and box.conf >= 0.5)
            if person_count > 1 or len(face_encodings) > 1:
                if multi_face_start is None:
                    multi_face_start = time.time()
                elif time.time() - multi_face_start > 2:
                    cv2.imwrite(f'data/debug_multi_face_{int(time.time())}.jpg', frame)
                    return False, "Multiple faces or persons detected."
            else:
                multi_face_start = None

            for i, (top, right, bottom, left) in enumerate(face_locations):
                top = int(top / used_scale)
                right = int(right / used_scale)
                bottom = int(bottom / used_scale)
                left = int(left / used_scale)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f"Face {i+1}", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            gaze_direction = "unknown"
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    gaze_direction = get_gaze_direction(face_landmarks, frame_width, frame_height, calibration_offset)
                    if gaze_direction != last_gaze:
                        last_gaze = gaze_direction
                    if gaze_direction in ["left", "right", "up", "down"]:
                        if gaze_violation_start is None:
                            gaze_violation_start = time.time()
                        gaze_duration = time.time() - gaze_violation_start
                        if gaze_duration > 5:
                            gaze_violation_count += 1
                            warning_message = f"Warning {gaze_violation_count}/3: Looked {gaze_direction} for {gaze_duration:.2f} seconds"
                            warnings.append(warning_message)
                            cv2.imwrite(f'data/debug_gaze_{gaze_direction}_{int(time.time())}.jpg', frame)
                            gaze_violation_start = time.time()
                            if gaze_violation_count >= 3:
                                return False, "Too many gaze violations."
                    else:
                        gaze_violation_start = None

            cv2.imshow('Verification', frame)
            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'):
                return False, "Verification cancelled."

            if len(face_encodings) == 1:
                with open(encoding_file, 'rb') as f:
                    known_encoding = pickle.load(f)
                match = face_recognition.compare_faces([known_encoding], face_encodings[0], tolerance=tolerance)
                if not match[0]:
                    cv2.imwrite(f'data/debug_no_face_{int(time.time())}.jpg', frame)
                    return False, "Face verification failed."
                else:
                    return True, "Verification successful."

            current_time = time.time()
            if current_time - last_debug_save >= 3:
                cv2.imwrite(f'data/debug_gaze_{gaze_direction}_{int(current_time)}.jpg', frame)
                last_debug_save = current_time

        return True, "Verification successful."

    finally:
        video_capture.release()
        face_mesh.close()
        cv2.destroyAllWindows()

# ─────────────────────────────────────────────────────────────────
# IMAGE-BASED REGISTRATION WITH QUALITY CHECK  (Enhanced)
# ─────────────────────────────────────────────────────────────────
def register_face_from_image(student_id, image, model="cnn"):
    """Registers student's face from an uploaded image with quality pre-check."""
    # Quality check FIRST before expensive CNN face detection
    ok, quality_msg, brightness, blur_score = check_image_quality(image)
    if not ok:
        return False, quality_msg

    # Aggressively resize massive mobile photos to prevent CNN lockups
    h, w = image.shape[:2]
    if w > 800:
        scale = 800.0 / w
        image = cv2.resize(image, (800, int(h * scale)))

    face_locations, face_encodings, used_scale = detect_faces(image, model=model, scale=1.0)
    
    if len(face_encodings) == 1:
        # Check face is reasonably centered (not at the very edges)
        top, right, bottom, left = face_locations[0]
        ih, iw = image.shape[:2]
        margin_x = iw * 0.05
        margin_y = ih * 0.05
        if left < margin_x or top < margin_y or right > iw - margin_x or bottom > ih - margin_y:
            return False, "Face is too close to the image edge. Center your face and try again."

        encoding = face_encodings[0]
        os.makedirs('data', exist_ok=True)
        try:
            with open(f'data/{student_id}_encoding.pkl', 'wb') as f:
                pickle.dump(encoding, f)
            # Save the registration photo so admins can verify the face visually
            photo_path = f'data/{student_id}_photo.jpg'
            cv2.imwrite(photo_path, image)
            print(f"Registered {student_id} from uploaded image (brightness={brightness:.0f}, blur={blur_score:.0f}).")
            return True, "Registration successful."
        except Exception as e:
            return False, f"Failed to save encoding: {e}"
    elif len(face_encodings) == 0:
        return False, "No face detected in the uploaded image. Ensure good lighting and try again."
    else:
        return False, f"Multiple faces ({len(face_encodings)}) detected. Ensure only you are in the frame."

# ─────────────────────────────────────────────────────────────────
# IMAGE-BASED VERIFICATION WITH QUALITY CHECK  (Enhanced)
# ─────────────────────────────────────────────────────────────────
def verify_face_from_image(student_id, image, tolerance=0.5, model="cnn"):
    """Verifies student's face from an uploaded image with quality pre-check."""
    encoding_file = f'data/{student_id}_encoding.pkl'
    if not os.path.exists(encoding_file):
        return False, f"No previous registration found for ID: {student_id}."

    # Quality check FIRST
    ok, quality_msg, brightness, blur_score = check_image_quality(image)
    if not ok:
        return False, quality_msg

    # Aggressively resize massive mobile photos to prevent CNN lockups
    h, w = image.shape[:2]
    if w > 800:
        scale = 800.0 / w
        image = cv2.resize(image, (800, int(h * scale)))

    face_locations, face_encodings, used_scale = detect_faces(image, model=model, scale=1.0)
    
    if len(face_encodings) == 1:
        with open(encoding_file, 'rb') as f:
            known_encoding = pickle.load(f)
        match = face_recognition.compare_faces([known_encoding], face_encodings[0], tolerance=tolerance)
        if match[0]:
            print(f"Verified {student_id} from uploaded image (brightness={brightness:.0f}, blur={blur_score:.0f}).")
            return True, "Verification successful."
        else:
            return False, "Face mismatch. Verification failed."
    elif len(face_encodings) == 0:
        return False, "No face detected in the uploaded image."
    else:
        return False, f"Multiple faces ({len(face_encodings)}) detected in the verification image."

# ─────────────────────────────────────────────────────────────────
# DATABASE  (Enhanced with exam_activity_log table)
# ─────────────────────────────────────────────────────────────────
@contextmanager
def get_db():
    conn = sqlite3.connect('data/proctoring_logs.db')
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    """Initialize all database tables including the new exam_activity_log."""
    with get_db() as conn:
        c = conn.cursor()

        # Existing violations/events log
        c.execute('''CREATE TABLE IF NOT EXISTS logs
                     (student_id TEXT, event_type TEXT, timestamp REAL, details TEXT)''')

        # NEW: Full exam activity log — records every event, not just violations
        c.execute('''CREATE TABLE IF NOT EXISTS exam_activity_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id      TEXT    NOT NULL,
            session_id      TEXT    NOT NULL,
            timestamp       REAL    NOT NULL,
            elapsed_seconds REAL    NOT NULL,
            event_type      TEXT    NOT NULL,
            value           TEXT,
            is_violation    INTEGER DEFAULT 0,
            details         TEXT
        )''')

        conn.commit()

def log_violation(student_id, event_type, details):
    """Log a violation event to the legacy logs table."""
    with get_db() as conn:
        c = conn.cursor()
        c.execute("INSERT INTO logs VALUES (?, ?, ?, ?)", 
                  (student_id, event_type, time.time(), details))
        conn.commit()

def log_activity(student_id, session_id, session_start_time, event_type, value, is_violation=False, details=None):
    """
    Log any exam activity event to exam_activity_log.
    
    event_type: 'gaze', 'head_pose', 'object_detected', 'face_status',
                'violation', 'session_start', 'session_end'
    value:      e.g. 'forward', 'left', 'right', 'up', 'down',
                     'face_present', 'face_missing', 'multiple_faces',
                     'cell phone', 'book', etc.
    """
    elapsed = time.time() - session_start_time
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""INSERT INTO exam_activity_log
                     (student_id, session_id, timestamp, elapsed_seconds,
                      event_type, value, is_violation, details)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                  (student_id, session_id, time.time(), round(elapsed, 2),
                   event_type, value, 1 if is_violation else 0, details))
        conn.commit()

def get_activity_log(student_id, session_id=None):
    """Retrieve full activity log for a student, optionally filtered by session."""
    with get_db() as conn:
        c = conn.cursor()
        if session_id:
            c.execute("""SELECT * FROM exam_activity_log
                         WHERE student_id=? AND session_id=?
                         ORDER BY timestamp ASC""", (student_id, session_id))
        else:
            c.execute("""SELECT * FROM exam_activity_log
                         WHERE student_id=?
                         ORDER BY timestamp DESC""", (student_id,))
        return c.fetchall()

def get_session_summary(student_id, session_id):
    """Generate a summary dict for a given exam session."""
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""SELECT event_type, value, is_violation, elapsed_seconds
                     FROM exam_activity_log
                     WHERE student_id=? AND session_id=?
                     ORDER BY timestamp ASC""", (student_id, session_id))
        rows = c.fetchall()

    if not rows:
        return None

    total_frames = len(rows)
    violations   = sum(1 for r in rows if r[2] == 1)
    gaze_counts  = {}
    for r in rows:
        if r[0] == 'gaze':
            gaze_counts[r[1]] = gaze_counts.get(r[1], 0) + 1

    forward_pct = round(gaze_counts.get('forward', 0) / max(total_frames, 1) * 100, 1)
    duration    = rows[-1][3] if rows else 0

    return {
        'total_events':  total_frames,
        'violations':    violations,
        'gaze_counts':   gaze_counts,
        'forward_pct':   forward_pct,
        'duration_secs': duration,
        'events':        rows
    }


def get_head_pose(face_landmarks, frame_width, frame_height):
    nose_tip = face_landmarks.landmark[1]
    yaw = nose_tip.x * frame_width - frame_width / 2
    if abs(yaw) > 0.1 * frame_width:
        return "turned_left" if yaw > 0 else "turned_right"
    return "forward"

def monitor_exam(student_id, video_capture, duration_seconds=20, max_warnings=3, short_violation_duration=2, down_violation_duration=5, long_violation_duration=10, model_choice='mediapipe'):
    """Monitor student activities with custom rules for exam."""
    global current_state

    # Reset state at exam START
    current_state['alerts'] = []
    current_state['gaze'] = 'forward'
    current_state['head_pose'] = 'forward'

    init_db()

    # Generate a unique session ID for this exam attempt
    session_id = str(uuid.uuid4())
    session_start = time.time()

    if not video_capture or not video_capture.isOpened():
        return False, "Webcam error", []

    log_activity(student_id, session_id, session_start, 'session_start', 'started')

    calibration_offset = calibrate_gaze(video_capture)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)
    yolo_model = YOLO('yolov8n.pt')
    
    custom_tracker = None
    if model_choice == 'custom_mobilenet':
        custom_tracker = CustomPyTorchGazeTracker()

    warnings = []
    interval_start = time.time()
    violation_start = {'gaze': None, 'head': None, 'object': None, 'multi_person': None}
    long_violation_start = {'gaze': None}
    last_gaze = "forward"
    last_head_pose = "forward"
    violation_count = 0
    last_process_time = 0
    prohibited_objects = ['cell phone', 'book', 'paper', 'remote', 'banana', 'bottle']
    last_activity_log_time = 0

    try:
        while time.time() - interval_start < duration_seconds:

            current_time = time.time()
            if current_time - last_process_time < 0.1:
                time.sleep(0.01)
                continue
            last_process_time = current_time

            ret, frame = video_capture.read()
            if not ret:
                log_violation(student_id, "webcam_error", "Failed to capture frame")
                return False, "Failed to capture frame", warnings

            frame_height, frame_width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            # ---------------- MULTI PERSON ----------------
            yolo_results = yolo_model(frame)
            person_count = sum(
                1 for r in yolo_results for box in r.boxes
                if yolo_model.names[int(box.cls)] == 'person' and box.conf >= 0.5
            )

            if person_count > 1:
                if violation_start['multi_person'] is None:
                    violation_start['multi_person'] = time.time()

                duration = time.time() - violation_start['multi_person']

                if duration > short_violation_duration:
                    violation_count += 1
                    warning_message = f"Warning {violation_count}/{max_warnings}: Multiple persons detected"
                    current_state['alerts'].append(warning_message)
                    log_violation(student_id, "multi_person_violation", warning_message)
                    log_activity(student_id, session_id, session_start, 'violation', 'multiple_persons', is_violation=True, details=warning_message)
                    warnings.append(warning_message)
                    violation_start['multi_person'] = time.time()

                    if violation_count >= max_warnings:
                        termination_msg = "Exam terminated: Too many violations"
                        current_state['alerts'].append(termination_msg)
                        log_violation(student_id, "exam_terminated", termination_msg)
                        log_activity(student_id, session_id, session_start, 'session_end', 'terminated', is_violation=True, details=termination_msg)
                        return False, termination_msg, warnings
            else:
                violation_start['multi_person'] = None

            # ---------------- GAZE & HEAD ----------------
            gaze_direction = "forward"
            head_pose = "forward"

            if model_choice == 'custom_mobilenet' and custom_tracker is not None:
                gaze_direction = custom_tracker.predict_gaze(frame)
                head_pose = "forward"
                if gaze_direction == "multiple_faces":
                    gaze_direction = "unknown"
                    
            elif results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    gaze_direction = get_gaze_direction(
                        face_landmarks, frame_width, frame_height, calibration_offset
                    )
                    head_pose = get_head_pose(
                        face_landmarks, frame_width, frame_height
                    )

            current_state['gaze'] = gaze_direction
            current_state['head_pose'] = head_pose

            # Log activity every second (not every frame, to avoid flooding the DB)
            if current_time - last_activity_log_time >= 1.0:
                log_activity(student_id, session_id, session_start, 'gaze', gaze_direction)
                log_activity(student_id, session_id, session_start, 'head_pose', head_pose)
                face_status = 'face_present' if results.multi_face_landmarks else 'face_missing'
                log_activity(student_id, session_id, session_start, 'face_status', face_status)
                last_activity_log_time = current_time

            # ---- GAZE VIOLATION ----
            if gaze_direction in ["left", "right", "up", "down"]:
                if long_violation_start['gaze'] is None:
                    long_violation_start['gaze'] = time.time()
                long_duration = time.time() - long_violation_start['gaze']
                if long_duration > long_violation_duration:
                    termination_msg = "Exam terminated: Prolonged gaze away"
                    current_state['alerts'].append(termination_msg)
                    log_violation(student_id, "long_gaze_violation", termination_msg)
                    log_activity(student_id, session_id, session_start, 'session_end', 'terminated', is_violation=True, details=termination_msg)
                    return False, termination_msg, warnings

                if violation_start['gaze'] is None:
                    violation_start['gaze'] = time.time()
                duration = time.time() - violation_start['gaze']
                threshold = down_violation_duration if gaze_direction == "down" else short_violation_duration
                if duration > threshold:
                    violation_count += 1
                    warning_message = f"Warning {violation_count}/{max_warnings}: Looked {gaze_direction}"
                    current_state['alerts'].append(warning_message)
                    log_violation(student_id, "gaze_violation", warning_message)
                    log_activity(student_id, session_id, session_start, 'violation', f'gaze_{gaze_direction}', is_violation=True, details=warning_message)
                    warnings.append(warning_message)
                    violation_start['gaze'] = time.time()
                    if violation_count >= max_warnings:
                        termination_msg = "Exam terminated: Too many violations"
                        current_state['alerts'].append(termination_msg)
                        log_violation(student_id, "exam_terminated", termination_msg)
                        log_activity(student_id, session_id, session_start, 'session_end', 'terminated', is_violation=True, details=termination_msg)
                        return False, termination_msg, warnings
            else:
                violation_start['gaze'] = None
                long_violation_start['gaze'] = None

            # ---- HEAD VIOLATION ----
            if head_pose in ["turned_left", "turned_right"]:
                if violation_start['head'] is None:
                    violation_start['head'] = time.time()
                duration = time.time() - violation_start['head']
                if duration > short_violation_duration:
                    violation_count += 1
                    warning_message = f"Warning {violation_count}/{max_warnings}: Head {head_pose}"
                    current_state['alerts'].append(warning_message)
                    log_violation(student_id, "head_pose_violation", warning_message)
                    log_activity(student_id, session_id, session_start, 'violation', head_pose, is_violation=True, details=warning_message)
                    warnings.append(warning_message)
                    violation_start['head'] = time.time()
                    if violation_count >= max_warnings:
                        termination_msg = "Exam terminated: Too many violations"
                        current_state['alerts'].append(termination_msg)
                        log_violation(student_id, "exam_terminated", termination_msg)
                        log_activity(student_id, session_id, session_start, 'session_end', 'terminated', is_violation=True, details=termination_msg)
                        return False, termination_msg, warnings
            else:
                violation_start['head'] = None

            # ---------------- OBJECT DETECTION ----------------
            for r in yolo_results:
                for box in r.boxes:
                    if box.conf < 0.5:
                        continue
                    obj_name = yolo_model.names[int(box.cls)]
                    if obj_name in prohibited_objects:
                        log_activity(student_id, session_id, session_start, 'object_detected', obj_name)
                        if violation_start['object'] is None:
                            violation_start['object'] = time.time()
                        duration = time.time() - violation_start['object']
                        if duration > short_violation_duration:
                            violation_count += 1
                            warning_message = f"Warning {violation_count}/{max_warnings}: {obj_name} detected"
                            current_state['alerts'].append(warning_message)
                            log_violation(student_id, "object_violation", warning_message)
                            log_activity(student_id, session_id, session_start, 'violation', obj_name, is_violation=True, details=warning_message)
                            warnings.append(warning_message)
                            violation_start['object'] = time.time()
                            if violation_count >= max_warnings:
                                termination_msg = "Exam terminated: Too many violations"
                                current_state['alerts'].append(termination_msg)
                                log_violation(student_id, "exam_terminated", termination_msg)
                                log_activity(student_id, session_id, session_start, 'session_end', 'terminated', is_violation=True, details=termination_msg)
                                return False, termination_msg, warnings
                else:
                    violation_start['object'] = None

        log_activity(student_id, session_id, session_start, 'session_end', 'completed')
        return True, "Monitoring completed successfully", warnings

    finally:
        face_mesh.close()
        video_capture.release()
        cv2.destroyAllWindows()
        print("Webcam released.")