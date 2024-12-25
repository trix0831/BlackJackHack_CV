import cv2

for i in range(5):  # Test the first 5 indexes
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera index {i} is available")
        cap.release()
