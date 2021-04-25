import cv2
import numpy as np
import os 
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('D:/Machine Learning A-Z New/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/opencv/trainer.yml') 
face_cascade = cv2.CascadeClassifier("D:/Machine Learning A-Z New/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/opencv/haarcascade_frontalface_default.xml");
font = cv2.FONT_HERSHEY_SIMPLEX
id = 0
names = ['Aastik'] 
video = cv2.VideoCapture(0)
video.set(3, 640) # set video widht
video.set(4, 480) # set video height
minW = 0.1*video.get(3)
minH = 0.1*video.get(4)
while True:
    ret, frame = video.read()
    frame = cv2.flip(frame, 1) # Flip 
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale( 
        gray,
        scaleFactor = 1.05,
        minNeighbors = 10,
        minSize = (int(minW), int(minH)),
       )
    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = 'Unknown'
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(
                    frame, 
                    str(id), 
                    (x+5,y-5), 
                    font, 
                    1, 
                    (255,255,255), 
                    2
                   )
        cv2.putText(
                    frame, 
                    str(confidence), 
                    (x+5,y+h-5), 
                    font, 
                    1, 
                    (255,255,0), 
                    1
                   )  
    
    cv2.imshow('Recognizing',frame) 
    k = cv2.waitKey(10) & 0xff 
    if k == 27:
        break

print("\n [INFO] Exiting Program and cleanup stuff")
video.release()
cv2.destroyAllWindows()