import time
import cv2
import base64
import numpy as np
from src.face_module import get_gaze_direction, get_head_pose, current_state, log_violation
from src.custom_gaze_tracker import CustomPyTorchGazeTracker
import mediapipe as mp
from ultralytics import YOLO

# Global storage for stateless active sessions
active_sessions = {}

class ExamSession:
    def __init__(self, student_id, model_choice, max_warnings=5):
        self.student_id = student_id
        self.model_choice = model_choice
        self.max_warnings = max_warnings
        
        # Violation Duration Thresholds (increased for fair exams)
        self.short_violation_duration = 5
        self.down_violation_duration = 10
        self.long_violation_duration = 20
        
        # State Tracking
        self.violation_start = {'gaze': None, 'head': None, 'object': None, 'multi_person': None}
        self.long_violation_start = {'gaze': None}
        self.violation_count = 0
        self.alerts_queue = []
        self.is_terminated = False
        self.last_process_time = 0
        self.prohibited_objects = ['cell phone', 'book', 'paper', 'remote', 'banana', 'bottle']
        
        # AI Models Initialization
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)
        self.yolo_model = YOLO('yolov8n.pt')
        self.custom_tracker = None
        
        if self.model_choice == 'custom_mobilenet':
            self.custom_tracker = CustomPyTorchGazeTracker()

        # In-stream background calibration
        self.is_calibrated = False if model_choice == 'mediapipe' else True
        self.calibration_offset = (0, 0)
        self.calibration_frames = 0
        self.calibration_x_diffs = []
        self.calibration_y_diffs = []

    def process_base64_frame(self, base64_str):
        if self.is_terminated:
            return {"alert": "Exam Terminated", "violation_count": self.violation_count, "gaze": "terminated", "head": "terminated"}

        current_time = time.time()
        # Throttle to 10 FPS (100ms) maximum to save CPU
        if current_time - self.last_process_time < 0.1:
            return {"alert": "No alerts", "violation_count": self.violation_count, "gaze": "waiting", "head": "waiting"}
        self.last_process_time = current_time

        try:
            # Decode Base64
            encoded_data = base64_str.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return {"error": "Empty frame"}
            
            return self.analyze_frame(frame, current_time)
            
        except Exception as e:
            print(f"Frame Processing Error: {e}")
            return {"error": str(e)}

    def analyze_frame(self, frame, current_time):
        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(rgb_frame)
        
        # Initialize default state for early returns
        gaze_direction = "forward"
        head_pose = "forward"

        # ---------------- MULTI PERSON ----------------
        yolo_results = self.yolo_model(frame)
        person_count = sum(
            1 for r in yolo_results for box in r.boxes
            if self.yolo_model.names[int(box.cls)] == 'person' and box.conf >= 0.5
        )

        alert_message = "No alerts"

        if person_count > 1:
            if self.violation_start['multi_person'] is None:
                self.violation_start['multi_person'] = current_time

            duration = current_time - self.violation_start['multi_person']

            if duration > self.short_violation_duration:
                self.violation_count += 1
                alert_message = f"Warning {self.violation_count}/{self.max_warnings}: Multiple persons detected"
                log_violation(self.student_id, "multi_person_violation", alert_message)
                self.violation_start['multi_person'] = current_time

                if self.violation_count >= self.max_warnings:
                    self.is_terminated = True
                    alert_message = "Exam terminated: Too many violations"
                    log_violation(self.student_id, "exam_terminated", alert_message)
                    return {"alert": alert_message, "violation_count": self.violation_count, "gaze": gaze_direction, "head": head_pose}
        else:
            self.violation_start['multi_person'] = None

        # Return early if already warned this frame
        if alert_message != "No alerts":
            return {"alert": alert_message, "violation_count": self.violation_count, "gaze": gaze_direction, "head": head_pose}

        # ---------------- GAZE & HEAD ----------------
        # Route calculation based on selected AI engine
        if self.model_choice == 'custom_mobilenet' and self.custom_tracker is not None:
            gaze_direction = self.custom_tracker.predict_gaze(frame)
            head_pose = "forward"  # Model handles only gaze
        elif results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if not self.is_calibrated:
                    # Accumulate baseline offset during first ~2 seconds (20 frames)
                    left_eye = face_landmarks.landmark[33]
                    right_eye = face_landmarks.landmark[263]
                    nose_tip = face_landmarks.landmark[1]
                    
                    eye_center_x = (left_eye.x + right_eye.x) / 2 * frame_width
                    eye_center_y = (left_eye.y + right_eye.y) / 2 * frame_height
                    nose_x = nose_tip.x * frame_width
                    nose_y = nose_tip.y * frame_height
                    
                    self.calibration_x_diffs.append(eye_center_x - nose_x)
                    self.calibration_y_diffs.append(eye_center_y - nose_y)
                    self.calibration_frames += 1
                    
                    if self.calibration_frames >= 20:
                        self.calibration_offset = (
                            float(np.median(self.calibration_x_diffs)), 
                            float(np.median(self.calibration_y_diffs))
                        )
                        self.is_calibrated = True
                        print(f"Calibration complete for {self.student_id}: {self.calibration_offset}")
                    
                    gaze_direction = "calibrating"
                    head_pose = "calibrating"
                else:
                    gaze_direction = get_gaze_direction(
                        face_landmarks, frame_width, frame_height, self.calibration_offset
                    )
                    head_pose = get_head_pose(
                        face_landmarks, frame_width, frame_height
                    )

        # Global sync for side-effects (Not totally needed but keeps other UI parts happy if used)
        current_state['gaze'] = gaze_direction
        current_state['head_pose'] = head_pose

        # ---- GAZE VIOLATION ----
        if gaze_direction in ["left", "right", "up", "down"]:
            if self.long_violation_start['gaze'] is None:
                self.long_violation_start['gaze'] = current_time

            long_duration = current_time - self.long_violation_start['gaze']

            if long_duration > self.long_violation_duration:
                self.is_terminated = True
                alert_message = "Exam terminated: Prolonged gaze away"
                log_violation(self.student_id, "long_gaze_violation", alert_message)
                return {"alert": alert_message, "violation_count": self.violation_count, "gaze": gaze_direction, "head": head_pose}

            if self.violation_start['gaze'] is None:
                self.violation_start['gaze'] = current_time

            duration = current_time - self.violation_start['gaze']
            threshold = self.down_violation_duration if gaze_direction == "down" else self.short_violation_duration

            if duration > threshold:
                self.violation_count += 1
                alert_message = f"Warning {self.violation_count}/{self.max_warnings}: Looked {gaze_direction}"
                log_violation(self.student_id, "gaze_violation", alert_message)
                self.violation_start['gaze'] = current_time

                if self.violation_count >= self.max_warnings:
                    self.is_terminated = True
                    alert_message = "Exam terminated: Too many violations"
                    log_violation(self.student_id, "exam_terminated", alert_message)
                    return {"alert": alert_message, "violation_count": self.violation_count, "gaze": gaze_direction, "head": head_pose}
        else:
            self.violation_start['gaze'] = None
            self.long_violation_start['gaze'] = None

        if alert_message != "No alerts":
            return {"alert": alert_message, "violation_count": self.violation_count, "gaze": gaze_direction, "head": head_pose}

        # ---- HEAD VIOLATION ----
        if head_pose in ["turned_left", "turned_right"]:
            if self.violation_start['head'] is None:
                self.violation_start['head'] = current_time

            duration = current_time - self.violation_start['head']

            if duration > self.short_violation_duration:
                self.violation_count += 1
                alert_message = f"Warning {self.violation_count}/{self.max_warnings}: Head {head_pose}"
                log_violation(self.student_id, "head_pose_violation", alert_message)
                self.violation_start['head'] = current_time

                if self.violation_count >= self.max_warnings:
                    self.is_terminated = True
                    alert_message = "Exam terminated: Too many violations"
                    return {"alert": alert_message, "violation_count": self.violation_count, "gaze": gaze_direction, "head": head_pose}
        else:
            self.violation_start['head'] = None

        if alert_message != "No alerts":
            return {"alert": alert_message, "violation_count": self.violation_count, "gaze": gaze_direction, "head": head_pose}

        # ---------------- OBJECT DETECTION ----------------
        for r in yolo_results:
            for box in r.boxes:
                if box.conf < 0.5: continue
                obj_name = self.yolo_model.names[int(box.cls)]
                
                if obj_name in self.prohibited_objects:
                    if self.violation_start['object'] is None:
                        self.violation_start['object'] = current_time

                    duration = current_time - self.violation_start['object']

                    if duration > self.short_violation_duration:
                        self.violation_count += 1
                        alert_message = f"Warning {self.violation_count}/{self.max_warnings}: {obj_name} detected"
                        log_violation(self.student_id, "object_violation", alert_message)
                        self.violation_start['object'] = current_time

                        if self.violation_count >= self.max_warnings:
                            self.is_terminated = True
                            alert_message = "Exam terminated: Too many violations"
                            return {"alert": alert_message, "violation_count": self.violation_count, "gaze": gaze_direction, "head": head_pose}
                            
        # If loop finishes without returning, reset object timer
        if alert_message == "No alerts":
             self.violation_start['object'] = None

        return {"alert": alert_message, "violation_count": self.violation_count, "gaze": gaze_direction, "head": head_pose}
