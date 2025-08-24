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

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.imshow('Registration', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and len(face_encodings) == 1:
            encoding = face_encodings[0]
            os.makedirs('../data', exist_ok=True)
            with open(f'../data/{student_id}_encoding.pkl', 'wb') as f:
                pickle.dump(encoding, f)
            print(f"Registered {student_id}.")
            video_capture.release()
            cv2.destroyAllWindows()
            return True
        elif key == ord('q'):
            break
        elif len(face_encodings) != 1:
            print("Ensure exactly one face is visible.")

    video_capture.release()
    cv2.destroyAllWindows()
    return False

def verify_face(student_id, tolerance=0.6):
    """Placeholder: Verifies face continuously, returns True if verified, False on alert."""
    pass