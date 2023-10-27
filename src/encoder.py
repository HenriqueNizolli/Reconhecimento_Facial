import os
import cv2
import face_recognition


def encoding_image(img_list):
    encode_list = []
    for img in img_list:
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list


def list_images(path):
    image_list = []
    id_list = []
    for pessoa in os.listdir(path):
        for img_nome in os.listdir(path + '/' + pessoa):
            img = cv2.imread(path + '/' + pessoa + '/' + img_nome)
            image_list.append(img)
            id_list.append(pessoa)
    return [image_list, id_list]
