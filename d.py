import os
import cv2
import numpy as np
import json



cam = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
user1 = input("Kullanıcı adı giriniz: ")
#face_id = input('\n User id girin ==> ')
print("\n [BILGI] Kameraya bakin ve bekleyin..")
say = 0
os.mkdir('dataset/'+user1)

while (True):
    ret, cerceve = cam.read()
    cerceve = cv2.flip(cerceve, 1)
    gri = cv2.cvtColor(cerceve, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detecMultiScale(gri, 1.5, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(cerceve, (x,y), (x+w, y+h), (255, 0, 0), 2)
        say += 1
        path = "dataset/"+user1+"/"
        cv2.imwrite(path+str(say) + ".jpg", gri[y:y + h, x:x + w])
        cv2.imshow('DATA', cerceve)
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif say >= 50:
        break
cam.relase()
cv2.destroyAllWindows()

