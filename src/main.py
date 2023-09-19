# Imports
import cv2
import keyboard

# Iniciando a camera
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

faceClassifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

while True:
    isRead, frame = cam.read()

    if (not isRead) or (keyboard.is_pressed('Esc')):
        print("Shutdown.....")
        break

    frame = cv2.flip(frame, 1)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceClassifier.detectMultiScale(gray_frame)

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('img', frame)
    cv2.waitKey(1)

# Liberando a camera
cam.release()

# Fechando todas as janelas
cv2.destroyAllWindows()
