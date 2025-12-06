import cv2

for i in range(6):
    cap = cv2.VideoCapture(i)
    print(i, "opened?" , cap.isOpened())
    cap.release()
