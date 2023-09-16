# Imports
import cv2
import dlib


# Configurando a camera
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


while True:
    isRead, frame = cam.read()
    frame = cv2.flip(frame, 1)
    cv2.imshow('img', frame)
    cv2.waitKey(1)
