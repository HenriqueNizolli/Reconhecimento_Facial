import cv2


def draw_rectangle(frame, face, color):
    x, y, w, h = face
    cv2.rectangle(frame, (y, x), (h, w), color, 2)

    return frame


def write_on_image(frame, location, message, color):
    x = location[3]
    y = location[2] + 20
    cv2.putText(frame, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    return frame
