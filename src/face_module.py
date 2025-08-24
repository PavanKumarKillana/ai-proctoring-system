import face_recognition
import cv2
import pickle
import os

def register_face(student_id):
    """Registers student's face by capturing and encoding it."""
    video_capture = cv2.VideoCapture(0)
    print("Look at the camera. Press 's' to save (one face only). Press 'q' to quit.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Webcam access failed.")
            return False

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        print(f"Frame size: {small_frame.shape}")
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for i, (top, right, bottom, left) in enumerate(face_locations):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.imshow('Registration', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and len(face_encodings) == 1:
            encoding = face_encodings[0]
            os.makedirs('data', exist_ok=True)
            with open(f'data/{student_id}_encoding.pkl', 'wb') as f:
                pickle.dump(encoding, f)
            print(f"Registered {student_id}.")
            video_capture.release()
            cv2.destroyAllWindows()
            return True
        elif key == ord('q'):
            break
        elif len(face_encodings) != 1:
            print("Ensure exactly one face is visible.")
            cv2.imwrite('data/debug_no_face.jpg', frame)

    video_capture.release()
    cv2.destroyAllWindows()
    return False

def verify_face(student_id, tolerance=0.6):
    """Verifies face by comparing live capture to saved encoding."""
    encoding_file = f'data/{student_id}_encoding.pkl'
    if not os.path.exists(encoding_file):
        print(f"Error: No encoding found for {student_id}.")
        return False

    with open(encoding_file, 'rb') as f:
        known_encoding = pickle.load(f)

    video_capture = cv2.VideoCapture(0)
    print("Look at the camera for verification. Press 'q' to quit.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Webcam access failed.")
            return False

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for i, (top, right, bottom, left) in enumerate(face_locations):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.imshow('Verification', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        if len(face_encodings) == 1:
            match = face_recognition.compare_faces([known_encoding], face_encodings[0], tolerance=tolerance)
            if match[0]:
                print(f"Verified {student_id}.")
                video_capture.release()
                cv2.destroyAllWindows()
                return True
            else:
                print(f"Verification failed for {student_id}.")

    video_capture.release()
    cv2.destroyAllWindows()
    return False
