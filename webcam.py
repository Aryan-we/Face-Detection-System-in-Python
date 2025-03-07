import cv2
from random import randrange as rr
dataset=cv2.CascadeClassifier("face.xml")
#img=cv2.imread("aryan2.webp")
webcam=cv2.VideoCapture(0)
while True:
    success,frame=webcam.read()
    grayimg=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faceCoordinate=dataset.detectMultiScale(grayimg)
    for i in range(len(faceCoordinate)):
        x,y,w,h=faceCoordinate[i]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(rr(0,255),rr(0,255),rr(0,255)),2)
    cv2.imshow("Face Detection System",frame)
    key=cv2.waitKey(1)
    if(key==81 or key==113):
        break
webcam.release()