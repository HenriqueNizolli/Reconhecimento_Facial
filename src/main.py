# Imports
import cv2
import keyboard
import face_recognition
import numpy as np

#my imports
import encoder

#database faces
db_faces = encoder.list_imgs('../assets/img')
db_enc = encoder.encoding_imgs(db_faces)
print(db_faces)

# Iniciando a camera
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    isRead, frame = cam.read()

    if (not isRead) or (keyboard.is_pressed('Esc')):
        print("Shutdown.....")
        break

    frame = cv2.flip(frame, 1)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_recognition.face_locations(gray_frame)
    enc_faces = face_recognition.face_encodings(frame, faces)

    for enc, loc in zip(enc_faces, faces):
        match = face_recognition.compare_faces(db_enc, enc)
        dis = face_recognition.face_distance(db_enc, enc)
        gg = np.argmin(dis)

        print(match)
        print(dis)
        print(gg)

        if match[gg]:
            for x, y, w, h in faces:
                cv2.rectangle(frame, (y, x), (h, w), (0, 255, 0), 2)

    cv2.imshow('img', frame)
    cv2.waitKey(1)

# Liberando a camera
cam.release()

# Fechando todas as janelas
cv2.destroyAllWindows()
