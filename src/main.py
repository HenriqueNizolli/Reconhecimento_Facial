# Imports
import cv2
import keyboard
import face_recognition
import numpy as np

#my imports
import encoder
import utils

#database faces
db_faces, db_ids = encoder.list_imgs('../assets/img')
db_enc = encoder.encoding_imgs(db_faces)

# Iniciando a camera
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    isRead, frame = cam.read()

    if (not isRead) or (keyboard.is_pressed('Esc')):
        print("Shutdown.....")
        break

    faces = face_recognition.face_locations(frame)
    enc_faces = face_recognition.face_encodings(frame, faces)

    if faces:
        for enc, face_location in zip(enc_faces, faces):
            face_matchs = face_recognition.compare_faces(db_enc, enc)
            face_distances = face_recognition.face_distance(db_enc, enc)
            face_index = np.argmin(face_distances)

            if face_matchs[face_index]:
                frame = utils.drawRectagle(frame, face_location, (0, 255, 0))
                frame = utils.writeOnImg(frame, face_location, db_ids[face_index], (0, 255, 0))

            else:
                frame = utils.drawRectagle(frame, face_location, (0, 0, 255))
                frame = utils.writeOnImg(frame, face_location, 'unknown', (0, 0, 255))

    cv2.imshow('img', frame)
    cv2.waitKey(1)

# Liberando a camera
cam.release()

# Fechando todas as janelas
cv2.destroyAllWindows()
