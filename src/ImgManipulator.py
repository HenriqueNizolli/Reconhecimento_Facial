import cv2


def drawRectagle(frame, faces, color):
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    return frame

