import sys
import cv2
import time
import utils
import encoder
import keyboard
import numpy as np
import face_recognition

print("Load Faces :")
print("[Start]")
db_faces, db_ids = encoder.list_images('../assets/img')
db_enc = encoder.encoding_image(db_faces)
print("[Done]")

if len(db_enc) == 0:
    print("Image not found")
    print("Shutdown.....")
    sys.exit()

print("Starting Camera :")
print("[Start]")
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("[Done]")

print("Star Program :")
print("[Start]")
while True:
    start = time.time()
    isRead, frame = cam.read()

    if (not isRead) or (keyboard.is_pressed('Esc')):
        end = time.time()
        print(end - start)
        print("[Done]")
        print("Shutdown.....")
        break

    frame = cv2.medianBlur(frame, 3)

    faces = face_recognition.face_locations(frame)
    enc_faces = face_recognition.face_encodings(frame, faces)

    if faces:
        for enc, face_location in zip(enc_faces, faces):
            face_match = face_recognition.compare_faces(db_enc, enc)
            face_distances = face_recognition.face_distance(db_enc, enc)
            face_index = np.argmin(face_distances)

            if face_match[face_index]:
                frame = utils.draw_rectangle(frame, face_location, (0, 255, 0))
                frame = utils.write_on_image(frame, face_location, db_ids[face_index], (0, 255, 0))

            else:
                frame = utils.draw_rectangle(frame, face_location, (0, 0, 255))
                frame = utils.write_on_image(frame, face_location, 'unknown', (0, 0, 255))

    cv2.imshow('img', frame)
    cv2.waitKey(1)

    end = time.time()
    print(end - start)

# Releasing the camera
cam.release()

# close all windows
cv2.destroyAllWindows()
