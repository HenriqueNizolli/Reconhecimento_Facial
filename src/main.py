# Imports
import cv2
import dlib


# Iniciando a camera
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

faceClassifier = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

while True:
    isRead, frame = cam.read()

    if not isRead:
        print("Error !!!")
        break
        
    frame = cv2.flip(frame, 1)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = faceClassifier.detectMultiScale(gray_frame)

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('img', frame)
    cv2.waitKey(1)

# Liberando a camera
cam.release()

# Fechando todas as janelas
cv2.destroyAllWindows()
