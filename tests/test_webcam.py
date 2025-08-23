import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    print("Webcam working!")
else:
    print("Error: Check webcam permissions or connection.")
cap.release()