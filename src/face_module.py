import face_recognition
import cv2
import pickle
import os
import time

def initialize_webcam(max_retries=3, delay=1):
    """Initialize webcam with retries and multiple indices."""
    for attempt in range(max_retries):
        for index in [0, 1]:
            video_capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if video_capture.isOpened():
                print(f"Webcam initialized on index {index}.")
                return video_capture
            print(f"Webcam initialization failed on index {index}, attempt {attempt + 1}/{max_retries}...")
            time.sleep(delay)
    print("Error: Webcam access failed after retries.")
    return None

def detect_faces(frame, model="hog", scale=0.5, retries=3):
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

def register_face(student_id, max_attempts=200, timeout_seconds=120, model="hog"):
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
            if key == ord('s') and len(face_encodings) == 1:
                encoding = face_encodings[0]
                os.makedirs('data', exist_ok=True)
                try:
                    with open(f'data/{student_id}_encoding.pkl', 'wb') as f:
                        pickle.dump(encoding, f)
                    print(f"Registered {student_id}.")
                    return True
                except Exception as e:
                    print(f"Error saving encoding: {e}")
                    return False
            elif key == ord('q'):
                print("Registration cancelled by user.")
                return False
            elif len(face_encodings) != 1:
                print(f"Detected {len(face_encodings)} faces. Ensure exactly one face is visible.")
                current_time = time.time()
                if current_time - last_debug_save >= 15:  # Save debug image every 15 seconds
                    cv2.imwrite(f'data/debug_no_face_{int(current_time)}.jpg', frame)
                    last_debug_save = current_time

            attempts += 1

        print("Registration timed out or max attempts reached.")
        return False

    finally:
        video_capture.release()
        cv2.destroyAllWindows()

def verify_face(student_id, tolerance=0.5, max_attempts=200, timeout_seconds=120, model="hog"):
    """Verifies face by comparing live capture to saved encoding."""
    encoding_file = f'data/{student_id}_encoding.pkl'
    if not os.path.exists(encoding_file):
        print(f"Error: No encoding found for {student_id}.")
        return False

    with open(encoding_file, 'rb') as f:
        known_encoding = pickle.load(f)

    video_capture = initialize_webcam()
    if not video_capture:
        print("Verification failed due to webcam error.")
        return False

    print(f"Look at the camera for verification. Press 'q' to quit. Using {model} model.")
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
            cv2.imshow('Verification', frame)

            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'):
                print("Verification cancelled by user.")
                return False

            if len(face_encodings) == 1:
                match = face_recognition.compare_faces([known_encoding], face_encodings[0], tolerance=tolerance)
                if match[0]:
                    print(f"Verified {student_id}.")
                    return True
                else:
                    print(f"Verification failed for {student_id}.")
                    current_time = time.time()
                    if current_time - last_debug_save >= 15:
                        cv2.imwrite(f'data/debug_no_face_{int(current_time)}.jpg', frame)
                        last_debug_save = current_time

            elif len(face_encodings) != 1:
                print(f"Detected {len(face_encodings)} faces. Ensure exactly one face is visible.")
                current_time = time.time()
                if current_time - last_debug_save >= 15:
                    cv2.imwrite(f'data/debug_no_face_{int(current_time)}.jpg', frame)
                    last_debug_save = current_time

            attempts += 1

        print("Verification timed out or max attempts reached.")
        return False

    finally:
        video_capture.release()
        cv2.destroyAllWindows()