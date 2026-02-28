from src.face_module import verify_face_with_gaze

def test_verify_face():
    student_id = "test_student"
    print(f"Starting face verification for {student_id}...")
    success, message = verify_face_with_gaze(student_id, tolerance=0.5, max_attempts=300, timeout_seconds=60, model="cnn")
    if success:
        print(f"Face verification successful for {student_id}!")
    else:
        print(f"Face verification failed for {student_id}: {message}")

if __name__ == "__main__":
    test_verify_face()