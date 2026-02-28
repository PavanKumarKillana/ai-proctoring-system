from src.face_module import register_face

def test_register_face():
    student_id = "test_student"
    print(f"Starting face registration for {student_id}...")
    success = register_face(student_id, max_attempts=200, timeout_seconds=120, model="cnn")
    if success:
        print(f"Face registration successful for {student_id}!")
    else:
        print(f"Face registration failed for {student_id}. Ensure one face is visible, use bright lighting, and center your face.")

if __name__ == "__main__":
    test_register_face()