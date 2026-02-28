import face_recognition
import cv2
import pickle
import os
import time
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import sqlite3
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
    # Make "up" more forgiving since looking at a standard monitor can tilt the face up
    if y_diff > 10.0:
        return "down"
    elif y_diff < -15.0:  # Much higher threshold for "up"
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
                    
                    # Store all coordinates, let median filter out outliers naturally
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
        # Use median to naturally discard blinking or head movement outliers
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
            
            # Auto-save once a single face is detected clearly
            if len(face_encodings) == 1:
                # Add a tiny delay to show the "Face 1" box before disappearing
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
            
            # Still allow manual quit
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
                    print(f"Face missing started at {time.time() - start_time:.2f} seconds")
                else:
                    face_missing_duration = time.time() - face_missing_start
                    if face_missing_duration < 10:
                        print(f"Face missing: {face_missing_duration:.2f} seconds (logging silently)")
                    else:
                        print(f"Warning: Please stay in front of camera ({face_missing_duration:.2f} seconds)")
                    if face_missing_duration > 10:
                        face_missing_count += 1
                        face_missing_start = time.time()
                        print(f"Face missing violation {face_missing_count}/3")
                        cv2.imwrite(f'data/debug_face_missing_{int(time.time())}.jpg', frame)
                        if face_missing_count >= 3:
                            print("Verification failed: Repeated face missing.")
                            return False, "Repeated face missing violations."
            else:
                face_missing_start = None

            yolo_results = yolo_model(frame)
            person_count = sum(1 for r in yolo_results for box in r.boxes if yolo_model.names[int(box.cls)] == 'person' and box.conf >= 0.5)
            if person_count > 1 or len(face_encodings) > 1:
                if multi_face_start is None:
                    multi_face_start = time.time()
                    print(f"Multiple faces/persons detected at {time.time() - start_time:.2f} seconds")
                elif time.time() - multi_face_start > 2:
                    print("Verification failed: Multiple faces/persons detected >2s.")
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
                    for idx in [33, 263, 1]:
                        x = int(face_landmarks.landmark[idx].x * frame_width)
                        y = int(face_landmarks.landmark[idx].y * frame_height)
                        color = (0, 255, 255) if idx in [33, 263] else (255, 0, 255)
                        cv2.circle(frame, (x, y), 3, color, -1)
                    if gaze_direction != last_gaze:
                        print(f"Gaze detected: {gaze_direction} at {time.time() - start_time:.2f} seconds")
                        last_gaze = gaze_direction
                    cv2.putText(frame, f"Gaze: {gaze_direction}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    if gaze_direction in ["left", "right", "up", "down"]:
                        if gaze_violation_start is None:
                            gaze_violation_start = time.time()
                            print(f"Violation started: Looking {gaze_direction} at {time.time() - start_time:.2f} seconds")
                        gaze_duration = time.time() - gaze_violation_start
                        if gaze_duration > 5:
                            gaze_violation_count += 1
                            warning_message = f"Warning {gaze_violation_count}/3: Looked {gaze_direction} for {gaze_duration:.2f} seconds"
                            print(warning_message)
                            warnings.append(warning_message)
                            cv2.imwrite(f'data/debug_gaze_{gaze_direction}_{int(time.time())}.jpg', frame)
                            gaze_violation_start = time.time()
                            if gaze_violation_count >= 3:
                                print("Verification failed: Too many gaze violations.")
                                return False, "Too many gaze violations."
                    else:
                        gaze_violation_start = None

            for r in yolo_results:
                for box in r.boxes:
                    if box.conf < 0.5:
                        continue
                    obj_name = yolo_model.names[int(box.cls)]
                    if obj_name in ['cell phone', 'book', 'paper', 'remote', 'banana', 'bottle']:
                        if object_violation_start is None:
                            object_violation_start = time.time()
                            print(f"Object detected: {obj_name} at {time.time() - start_time:.2f} seconds")
                        obj_duration = time.time() - object_violation_start
                        if obj_duration > 5:
                            object_violation_count += 1
                            warning_message = f"Warning {object_violation_count}/3: {obj_name} detected for {obj_duration:.2f} seconds"
                            print(warning_message)
                            warnings.append(warning_message)
                            cv2.imwrite(f'data/debug_object_{obj_name}_{int(time.time())}.jpg', frame)
                            object_violation_start = time.time()
                            if object_violation_count >= 3:
                                print("Verification failed: Too many object violations.")
                                return False, f"Too many object violations ({obj_name})."
            else:
                object_violation_start = None

            feedback = "Ensure bright lighting, face centered, 1-2 ft from camera, no glare on glasses, plain background"
            cv2.putText(frame, feedback, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow('Verification', frame)

            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'):
                print("Verification cancelled by user.")
                return False, "Verification cancelled."

            if len(face_encodings) == 1:
                with open(encoding_file, 'rb') as f:
                    known_encoding = pickle.load(f)
                match = face_recognition.compare_faces([known_encoding], face_encodings[0], tolerance=tolerance)
                if not match[0]:
                    print(f"Face verification failed for {student_id} at {time.time() - start_time:.2f} seconds")
                    cv2.imwrite(f'data/debug_no_face_{int(time.time())}.jpg', frame)
                    return False, "Face verification failed."
                else:
                    print(f"Verified {student_id} successfully.")
                    return True, "Verification successful."

            current_time = time.time()
            if current_time - last_debug_save >= 3:
                cv2.imwrite(f'data/debug_gaze_{gaze_direction}_{int(current_time)}.jpg', frame)
                last_debug_save = current_time

        print(f"Verified {student_id} successfully.")
        return True, "Verification successful."

    finally:
        video_capture.release()
        face_mesh.close()
        cv2.destroyAllWindows()

@contextmanager
def get_db():
    conn = sqlite3.connect('data/proctoring_logs.db')
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_db() as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS logs
                     (student_id TEXT, event_type TEXT, timestamp REAL, details TEXT)''')
        conn.commit()

def log_violation(student_id, event_type, details):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("INSERT INTO logs VALUES (?, ?, ?, ?)", 
                  (student_id, event_type, time.time(), details))
        conn.commit()

def get_head_pose(face_landmarks, frame_width, frame_height):
    nose_tip = face_landmarks.landmark[1]
    yaw = nose_tip.x * frame_width - frame_width / 2
    if abs(yaw) > 0.1 * frame_width:
        return "turned_left" if yaw > 0 else "turned_right"
    return "forward"

def monitor_exam(student_id, video_capture, duration_seconds=20, max_warnings=3, short_violation_duration=2, down_violation_duration=5, long_violation_duration=10, model_choice='mediapipe'):
    """Monitor student activities with custom rules for exam."""
    global current_state

    # 🔥 Reset state at exam START (NOT at end)
    current_state['alerts'] = []
    current_state['gaze'] = 'forward'
    current_state['head_pose'] = 'forward'

    init_db()

    if not video_capture or not video_capture.isOpened():
        return False, "Webcam error", []

    calibration_offset = calibrate_gaze(video_capture)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)
    yolo_model = YOLO('yolov8n.pt')
    
    # Initialize custom AI model if selected
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
                    print(warning_message)

                    current_state['alerts'].append(warning_message)
                    log_violation(student_id, "multi_person_violation", warning_message)
                    warnings.append(warning_message)

                    violation_start['multi_person'] = time.time()

                    if violation_count >= max_warnings:
                        termination_msg = "Exam terminated: Too many violations"
                        current_state['alerts'].append(termination_msg)
                        log_violation(student_id, "exam_terminated", termination_msg)
                        return False, termination_msg, warnings
            else:
                violation_start['multi_person'] = None

            # ---------------- GAZE & HEAD ----------------
            gaze_direction = "forward"
            head_pose = "forward"

            # Route calculation based on selected AI engine
            if model_choice == 'custom_mobilenet' and custom_tracker is not None:
                gaze_direction = custom_tracker.predict_gaze(frame)
                head_pose = "forward"  # Model handles only gaze, assume forward head
                
                # If model handles multi_person internally, sync violation clock here
                if gaze_direction == "multiple_faces": 
                    # Handled by YOLO above anyway, so we just treat as unknown to not break loop
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

            # ---- GAZE VIOLATION ----
            if gaze_direction in ["left", "right", "up", "down"]:

                if long_violation_start['gaze'] is None:
                    long_violation_start['gaze'] = time.time()

                long_duration = time.time() - long_violation_start['gaze']

                if long_duration > long_violation_duration:
                    termination_msg = "Exam terminated: Prolonged gaze away"
                    current_state['alerts'].append(termination_msg)
                    log_violation(student_id, "long_gaze_violation", termination_msg)
                    return False, termination_msg, warnings

                if violation_start['gaze'] is None:
                    violation_start['gaze'] = time.time()

                duration = time.time() - violation_start['gaze']
                threshold = down_violation_duration if gaze_direction == "down" else short_violation_duration

                if duration > threshold:
                    violation_count += 1
                    warning_message = f"Warning {violation_count}/{max_warnings}: Looked {gaze_direction}"
                    print(warning_message)

                    current_state['alerts'].append(warning_message)
                    log_violation(student_id, "gaze_violation", warning_message)
                    warnings.append(warning_message)

                    violation_start['gaze'] = time.time()

                    if violation_count >= max_warnings:
                        termination_msg = "Exam terminated: Too many violations"
                        current_state['alerts'].append(termination_msg)
                        log_violation(student_id, "exam_terminated", termination_msg)
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
                    print(warning_message)

                    current_state['alerts'].append(warning_message)
                    log_violation(student_id, "head_pose_violation", warning_message)
                    warnings.append(warning_message)

                    violation_start['head'] = time.time()

                    if violation_count >= max_warnings:
                        termination_msg = "Exam terminated: Too many violations"
                        current_state['alerts'].append(termination_msg)
                        log_violation(student_id, "exam_terminated", termination_msg)
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

                        if violation_start['object'] is None:
                            violation_start['object'] = time.time()

                        duration = time.time() - violation_start['object']

                        if duration > short_violation_duration:
                            violation_count += 1
                            warning_message = f"Warning {violation_count}/{max_warnings}: {obj_name} detected"
                            print(warning_message)

                            current_state['alerts'].append(warning_message)
                            log_violation(student_id, "object_violation", warning_message)
                            warnings.append(warning_message)

                            violation_start['object'] = time.time()

                            if violation_count >= max_warnings:
                                termination_msg = "Exam terminated: Too many violations"
                                current_state['alerts'].append(termination_msg)
                                log_violation(student_id, "exam_terminated", termination_msg)
                                return False, termination_msg, warnings
                else:
                    violation_start['object'] = None

        return True, "Monitoring completed successfully", warnings

    finally:
        face_mesh.close()
        video_capture.release()
        cv2.destroyAllWindows()
        print("Webcam released.")
        