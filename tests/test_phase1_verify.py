from src.face_module import verify_face

student_id = "test_student"
success = verify_face(student_id)

if success:
    print("Face verification successful!")
else:
    print("Face verification failed.")
