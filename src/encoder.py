import os
import cv2
import face_recognition


def encoding_imgs(img_list):
    encodelist = []
    for img in img_list:
        encode = face_recognition.face_encodings(img[0])
        encodelist.append(encode)
    return encodelist


def list_imgs(path):
    imglist = []
    for pessoa in os.listdir(path):
        for img_nome in os.listdir(path + '/' + pessoa):
            img = cv2.imread(path + '/' + pessoa + '/' + img_nome)
            imglist.append([img, pessoa])
    return imglist

