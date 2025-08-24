from src.face_module import register_face

student_id = "test_student"
success = register_face(student_id)
if success:
    print("Face registration successful!")
else:
    print("Face registration failed.")